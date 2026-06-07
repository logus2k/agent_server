# app/admin_api.py
"""
Admin API for agent_server - CRUD on agent presets + view/edit service config.

Mounted at /admin/api/* by main.py. No authentication: this is a
single-admin, trusted-network surface - keep it off any public proxy.

Agent presets HOT-RELOAD: a successful create/update/delete mutates the
live `AGENTS` registry in main.py in place, so the change takes effect
immediately with no restart.

agent_config.json is NOT hot-applied: PUT /config writes the file only.
The running process keeps the config it loaded at startup
(main.RAW_CONFIG); GET /config reports `restart_pending` when the on-disk
file differs from that startup snapshot.

See documents/plans/agent_server_admin_ux.md.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from pydantic import BaseModel


admin_router = APIRouter(prefix="/admin/api", tags=["admin"])

# Data dirs - resolved relative to this file: app/admin_api.py -> app/data/...
_APP_DIR = Path(__file__).resolve().parent
_DATA_DIR = _APP_DIR / "data"
_AGENTS_DIR = _DATA_DIR / "agents"
_PROMPTS_DIR = _DATA_DIR / "prompts"

# Agent names become filenames and registry keys - keep them safe.
_NAME_RE = re.compile(r"^[a-z0-9_]+$")

# Agents the server itself depends on - refuse to delete these.
_PROTECTED_AGENTS = {"router"}

_VALID_MEMORY_POLICIES = {"none", "thread_window"}


# ---------------------------------------------------------------------------
# Shared validation (also used by main.load_agent_presets at startup)
# ---------------------------------------------------------------------------
def validate_agent_dict(data: Any) -> List[str]:
    """Strict validation of an agent definition (the `.agent.json` dict).

    Shared by the startup loader and the admin API so the two can never
    disagree on what a valid preset is. Returns a list of human-readable
    errors; an empty list means valid. Here `system_prompt` is a file
    path (as stored on disk), not the prompt text.
    """
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["agent definition must be a JSON object"]

    name = (data.get("name") or "").strip()
    if not name:
        errors.append("missing required field 'name'")
    elif not _NAME_RE.match(name.lower()):
        errors.append("'name' may contain only lowercase letters, digits and underscore")

    if "grammar_path" in data and (data.get("grammar_path") or "").strip():
        errors.append("'grammar_path' is not allowed (grammar support was removed)")

    if "system_prompt_path" in data:
        errors.append("uses 'system_prompt_path'; the supported key is 'system_prompt'")

    if not (data.get("system_prompt") or "").strip():
        errors.append("missing required field 'system_prompt'")

    policy = (data.get("memory_policy") or "none").strip().lower()
    if policy not in _VALID_MEMORY_POLICIES:
        errors.append(f"invalid 'memory_policy' {policy!r}; allowed: {sorted(_VALID_MEMORY_POLICIES)}")

    po = data.get("params_override")
    if po is not None and not isinstance(po, dict):
        errors.append("'params_override' must be a JSON object")

    tf = data.get("tts_field")
    if tf is not None and not isinstance(tf, str):
        errors.append("'tts_field' must be a string")

    return errors


def validate_config(cfg: Any) -> List[str]:
    """Strict validation of agent_config.json, mirroring main.py's
    startup checks. Returns a list of errors; empty means valid."""
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["config must be a JSON object"]

    # `models` is grouped by task: { chat:[...], embedding:[...], reranking:[...] }.
    models = cfg.get("models")
    if not isinstance(models, dict):
        errors.append("'models' must be an object grouped by task "
                      "(chat / embedding / reranking)")
    else:
        chat = models.get("chat")
        if not isinstance(chat, list) or not chat:
            errors.append("'models.chat' must be a non-empty array")
            chat = []
        active = [m for m in chat if isinstance(m, dict) and m.get("active") is True]
        if len(active) != 1:
            errors.append(f'exactly one model in models.chat must have "active": true '
                          f"(found {len(active)})")
        # Validate every entry across all groups.
        for group, entries in models.items():
            if not isinstance(entries, list):
                errors.append(f"'models.{group}' must be an array")
                continue
            for m in entries:
                if not isinstance(m, dict):
                    errors.append(f"each entry in 'models.{group}' must be an object")
                    continue
                if not (m.get("model_id") or "").strip():
                    errors.append(f"every model in 'models.{group}' needs a non-empty 'model_id'")
                ab = (m.get("active_backend") or "").strip()
                if not ab:
                    errors.append(f"model {m.get('model_id')!r}: missing 'active_backend'")
                elif not isinstance(m.get("backends", {}).get(ab), dict):
                    errors.append(f"model {m.get('model_id')!r}: no backends.{ab} block "
                                  f"for active_backend {ab!r}")
                opts = m.get("backends", {}).get(ab, {}).get("options", {}) or {}
                if "grammar_path" in opts and str(opts.get("grammar_path") or "").strip():
                    errors.append(f"model {m.get('model_id')!r}: 'grammar_path' is not allowed")

    rt = cfg.get("runtime")
    if rt is not None and not isinstance(rt, dict):
        errors.append("'runtime' must be an object")
    elif isinstance(rt, dict):
        ps = rt.get("pool_size")
        if ps is not None and (isinstance(ps, bool) or not isinstance(ps, int) or ps < 1):
            errors.append("'runtime.pool_size' must be a positive integer")
        to = rt.get("per_request_timeout_s")
        if to is not None and (isinstance(to, bool) or not isinstance(to, int) or to < 0):
            errors.append("'runtime.per_request_timeout_s' must be a non-negative integer")

    mem = cfg.get("memory")
    if mem is not None and not isinstance(mem, dict):
        errors.append("'memory' must be an object")

    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _registry() -> Dict[str, Any]:
    """The live AGENTS dict from main.py. Mutating it in place hot-reloads
    the registry (the request handlers and the router share this object)."""
    from . import main as _main
    agents = getattr(_main, "AGENTS", None)
    if agents is None:
        raise HTTPException(status_code=503,
                            detail="agent registry not ready (server still starting)")
    return agents


def _atomic_write(path: Path, content: str) -> None:
    """Write via a temp file in the same directory, then os.replace - so a
    reader never sees a half-written file and a crash can't corrupt it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _config_path() -> Path:
    """Resolved path of agent_config.json (the file main.py loaded)."""
    from . import main as _main
    p = getattr(_main, "cfg_file", None)
    return Path(p) if p is not None else Path(os.getenv("AGENT_CONFIG", "agent_config.json"))


def _build_preset(agent_dict: Dict[str, Any]):
    """Construct a main.AgentPreset from a validated agent dict whose
    `system_prompt` is already an absolute path."""
    from . import main as _main
    return _main.AgentPreset(
        name=agent_dict["name"],
        system_prompt_path=agent_dict["system_prompt"],
        params_override=agent_dict.get("params_override") or {},
        memory_policy=(agent_dict.get("memory_policy") or "none").strip().lower(),
        tts_field=agent_dict.get("tts_field") or None,
    )


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class AgentBody(BaseModel):
    name: str
    system_prompt: str                       # the prompt TEXT (not a path)
    params_override: Dict[str, Any] = {}
    memory_policy: str = "none"
    tts_field: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent presets - CRUD
# ---------------------------------------------------------------------------
@admin_router.get("/agents")
def list_agents():
    """List every loaded agent preset (metadata only, no prompt text)."""
    agents = _registry()
    out = []
    for name in sorted(agents):
        p = agents[name]
        out.append({
            "name": p.name,
            "memory_policy": p.memory_policy,
            "params_override": dict(p.params_override or {}),
            "tts_field": p.tts_field,
            "protected": name in _PROTECTED_AGENTS,
        })
    return {"agents": out, "count": len(out)}


@admin_router.get("/agents/{name}")
def get_agent(name: str):
    """Full preset including the resolved system-prompt text."""
    agents = _registry()
    key = name.strip().lower()
    p = agents.get(key)
    if p is None:
        raise HTTPException(status_code=404, detail=f"unknown agent: {name!r}")
    text = ""
    if p.system_prompt_path:
        try:
            text = Path(p.system_prompt_path).read_text(encoding="utf-8")
        except OSError as e:
            raise HTTPException(status_code=500,
                                detail=f"system prompt file unreadable: {e}")
    return {
        "name": p.name,
        "system_prompt": text,
        "params_override": dict(p.params_override or {}),
        "memory_policy": p.memory_policy,
        "tts_field": p.tts_field,
        "protected": key in _PROTECTED_AGENTS,
    }


@admin_router.post("/agents", status_code=201)
def create_agent(body: AgentBody):
    return _upsert_agent(body, expect_exists=False)


@admin_router.put("/agents/{name}")
def update_agent(name: str, body: AgentBody):
    if name.strip().lower() != (body.name or "").strip().lower():
        raise HTTPException(status_code=400,
                            detail="URL name and body 'name' must match; 'name' is immutable")
    return _upsert_agent(body, expect_exists=True)


def _upsert_agent(body: AgentBody, *, expect_exists: bool) -> Dict[str, Any]:
    agents = _registry()
    name = (body.name or "").strip().lower()
    if not name or not _NAME_RE.match(name):
        raise HTTPException(status_code=422,
                            detail="'name' must be lowercase letters, digits and underscore")

    exists = name in agents
    if expect_exists and not exists:
        raise HTTPException(status_code=404, detail=f"unknown agent: {name!r}")
    if not expect_exists and exists:
        raise HTTPException(status_code=409, detail=f"agent {name!r} already exists")

    prompt_text = body.system_prompt or ""
    if not prompt_text.strip():
        raise HTTPException(status_code=422,
                            detail="'system_prompt' (the prompt text) must not be empty")

    prompt_path = _PROMPTS_DIR / f"{name}_system_prompt.txt"
    agent_path = _AGENTS_DIR / f"{name}.agent.json"

    # The on-disk .agent.json dict: system_prompt is the absolute path.
    agent_dict: Dict[str, Any] = {
        "name": name,
        "system_prompt": str(prompt_path),
        "params_override": dict(body.params_override or {}),
        "memory_policy": (body.memory_policy or "none").strip().lower(),
    }
    if body.tts_field:
        agent_dict["tts_field"] = body.tts_field

    errors = validate_agent_dict(agent_dict)
    if errors:
        raise HTTPException(status_code=422,
                            detail={"message": "invalid agent definition", "errors": errors})

    # Persist - prompt file then the .agent.json, each written atomically.
    try:
        _AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write(prompt_path, prompt_text)
        _atomic_write(agent_path, json.dumps(agent_dict, indent=2) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to write agent files: {e}")

    # Hot-reload: mutate the live registry in place.
    agents[name] = _build_preset(agent_dict)

    return {"status": "ok", "name": name, "created": not exists}


@admin_router.delete("/agents/{name}")
def delete_agent(name: str):
    agents = _registry()
    key = name.strip().lower()
    if key in _PROTECTED_AGENTS:
        raise HTTPException(status_code=409,
                            detail=f"agent {key!r} is required by the server and cannot be deleted")
    p = agents.get(key)
    if p is None:
        raise HTTPException(status_code=404, detail=f"unknown agent: {name!r}")

    agent_path = _AGENTS_DIR / f"{key}.agent.json"
    try:
        if agent_path.exists():
            agent_path.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to delete agent file: {e}")

    # Delete the prompt file too, but only if it lives in our prompts dir.
    if p.system_prompt_path:
        try:
            spp = Path(p.system_prompt_path).resolve()
            if spp.parent == _PROMPTS_DIR.resolve() and spp.exists():
                spp.unlink()
        except OSError:
            pass

    agents.pop(key, None)
    return {"status": "ok", "deleted": key}


# ---------------------------------------------------------------------------
# Service config - view / edit (restart-to-apply)
# ---------------------------------------------------------------------------
@admin_router.get("/config")
def get_config():
    """Return the live (startup-loaded) config, the on-disk config, and
    whether they differ (i.e. a restart is needed to apply on-disk edits)."""
    from . import main as _main
    live = getattr(_main, "RAW_CONFIG", None)
    cfg_path = _config_path()
    on_disk: Optional[Dict[str, Any]] = None
    read_error: Optional[str] = None
    try:
        on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        read_error = str(e)
    restart_pending = on_disk is not None and on_disk != live
    return {
        "live": live,
        "on_disk": on_disk,
        "restart_pending": restart_pending,
        "config_path": str(cfg_path),
        "read_error": read_error,
    }


@admin_router.put("/config")
def put_config(body: Dict[str, Any] = Body(...)):
    """Validate and write agent_config.json. Does NOT apply it - the
    running process keeps its startup config until agent_server restarts."""
    errors = validate_config(body)
    if errors:
        raise HTTPException(status_code=422,
                            detail={"message": "invalid agent_config.json", "errors": errors})
    cfg_path = _config_path()
    try:
        _atomic_write(cfg_path, json.dumps(body, indent=2) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to write config: {e}")

    from . import main as _main
    live = getattr(_main, "RAW_CONFIG", None)
    return {
        "status": "ok",
        "restart_pending": body != live,
        "note": "Config written. Restart agent_server (docker restart agent_server) to apply.",
    }


# ---------------------------------------------------------------------------
# Active-model switch (restart-based) - see documents/active_model_switching_sdk.md
# ---------------------------------------------------------------------------
# Container names from docker-compose.adapter.yml; overridable for other setups.
_LLAMA_CONTAINER = os.getenv("LLAMA_VISION_CONTAINER", "llama-vision")
_SELF_CONTAINER = os.getenv("AGENT_SERVER_CONTAINER", "agent_server")


class ActiveModelRequest(BaseModel):
    model_id: str
    category: str = "chat"


_SWITCHABLE_CATEGORIES = ("chat", "embedding", "reranking")


def _restart_for_switch() -> None:
    """Restart llama-vision (regenerates its preset with the new active model
    as the sole declared chat model) then agent_server (re-reads the config).
    Runs as a BackgroundTask AFTER the response is flushed. Restarting
    agent_server kills this process mid-call; the Docker daemon completes the
    restart regardless. Requires /var/run/docker.sock mounted (see compose)."""
    import docker  # lazy: only needed on a switch
    client = docker.from_env()
    # llama-vision first so agent_server comes back to a ready router.
    client.containers.get(_LLAMA_CONTAINER).restart(timeout=30)
    client.containers.get(_SELF_CONTAINER).restart(timeout=30)


@admin_router.post("/active-model")
def set_active_model(req: ActiveModelRequest, background: BackgroundTasks):
    """Switch the active local model in a given category (chat / embedding /
    reranking). Flips the `active` flags within that category's array in
    agent_config.json and restarts llama-vision + agent_server to apply (the
    adapter regenerates the preset declaring each category's active model on
    llama-vision boot). Embedding/reranking are always-resident; switching one
    briefly drops it for any dependent service (e.g. noted RAG) during the
    restart."""
    model_id = (req.model_id or "").strip()
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id is required")
    category = (req.category or "chat").strip().lower()
    if category not in _SWITCHABLE_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"invalid category {category!r}; allowed: {list(_SWITCHABLE_CATEGORIES)}")

    cfg_path = _config_path()
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"cannot read config: {e}")

    group = (((cfg.get("models") or {}).get(category)) or [])
    target = next((m for m in group if isinstance(m, dict)
                   and (m.get("model_id") or "").strip() == model_id), None)
    if target is None:
        available = [m.get("model_id") for m in group if isinstance(m, dict)]
        raise HTTPException(
            status_code=404,
            detail=f"unknown {category} model_id {model_id!r}; available: {available}")

    if target.get("active") is True:
        return {"status": "ok", "active_model": model_id,
                "category": category, "noop": True}

    for m in group:
        if isinstance(m, dict):
            m["active"] = (m.get("model_id") or "").strip() == model_id

    errors = validate_config(cfg)
    if errors:
        raise HTTPException(status_code=422,
                            detail={"message": "invalid agent_config.json after flip",
                                    "errors": errors})
    try:
        _atomic_write(cfg_path, json.dumps(cfg, indent=2) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to write config: {e}")

    resp = {
        "active_model": model_id,
        "category": category,
        "display_name": target.get("name", model_id),
        "family": target.get("family", ""),
    }

    # Verify Docker is reachable before promising a restart. If not, the
    # config is still written and applies on the next manual restart.
    try:
        import docker
        docker.from_env().ping()
    except Exception as e:  # noqa: BLE001 - any docker/connection error
        resp["status"] = "ok"
        resp["restart_pending"] = True
        resp["note"] = (f"config written but auto-restart unavailable ({e}); "
                        f"restart {_LLAMA_CONTAINER} + {_SELF_CONTAINER} manually to apply")
        return resp

    background.add_task(_restart_for_switch)
    resp["status"] = "switching"
    resp["note"] = (f"{_SELF_CONTAINER} is restarting (~10-20s); reconnect and "
                    f"re-fetch /v1/models to confirm")
    return resp


class ActiveContextRequest(BaseModel):
    context: int


@admin_router.post("/active-context")
def set_active_context(req: ActiveContextRequest, background: BackgroundTasks):
    """Change the context window (ctx-size) of the ACTIVE chat model without
    hand-editing agent_config.json. Updates the active model's `context`
    field and restarts llama-vision (the adapter regenerates its preset with
    the new `c = <context>`) + agent_server.

    VRAM scales with context (F16 KV): if the new size OOMs, llama-vision
    fails to load and the model stays down until a smaller value is chosen."""
    ctx = int(req.context or 0)
    if ctx < 512 or ctx > 1048576:
        raise HTTPException(status_code=422,
                            detail="context must be between 512 and 1048576 tokens")

    cfg_path = _config_path()
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"cannot read config: {e}")

    chat = (((cfg.get("models") or {}).get("chat")) or [])
    target = next((m for m in chat if isinstance(m, dict)
                   and m.get("active") is True), None)
    if target is None:
        raise HTTPException(status_code=404, detail="no active chat model in config")

    if int(target.get("context") or 0) == ctx:
        return {"status": "ok", "model_id": target.get("model_id"),
                "context": ctx, "noop": True}

    prev = target.get("context")
    target["context"] = ctx

    errors = validate_config(cfg)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "invalid agent_config.json after context change",
                    "errors": errors})
    try:
        _atomic_write(cfg_path, json.dumps(cfg, indent=2) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to write config: {e}")

    resp = {
        "model_id": target.get("model_id"),
        "display_name": target.get("name", target.get("model_id")),
        "context": ctx,
        "previous_context": prev,
    }

    try:
        import docker
        docker.from_env().ping()
    except Exception as e:  # noqa: BLE001
        resp["status"] = "ok"
        resp["restart_pending"] = True
        resp["note"] = (f"config written but auto-restart unavailable ({e}); "
                        f"restart {_LLAMA_CONTAINER} + {_SELF_CONTAINER} manually to apply")
        return resp

    background.add_task(_restart_for_switch)
    resp["status"] = "switching"
    resp["note"] = (f"{_LLAMA_CONTAINER} + {_SELF_CONTAINER} restarting (~40s) to apply "
                    f"ctx-size={ctx}; reconnect and re-fetch /v1/models to confirm")
    return resp


_PROJECTOR_PREFIX = "/agent_server_models"   # path prefix used by projector= in config


def _active_chat_model(cfg):
    chat = (((cfg.get("models") or {}).get("chat")) or [])
    return next((m for m in chat if isinstance(m, dict)
                 and m.get("active") is True), None)


def _model_projector(m):
    """Projector path on a chat model's active backend, or None."""
    if not m:
        return None
    bk = (m.get("backends") or {}).get(m.get("active_backend") or "") or {}
    return bk.get("projector")


@admin_router.get("/vision-adapters")
def list_vision_adapters():
    """Available mmproj/vision-projector GGUFs (scanned from data/models) plus
    the projector currently set on the active vision-capable chat model."""
    models_dir = Path(__file__).resolve().parent / "data" / "models"
    adapters = []
    try:
        for p in sorted(models_dir.glob("*.gguf")):
            if "mmproj" in p.name.lower():
                try:
                    size = p.stat().st_size
                except OSError:
                    size = None
                adapters.append({"file": p.name, "size": size})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"cannot scan models dir: {e}")

    try:
        cfg = json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"cannot read config: {e}")
    am = _active_chat_model(cfg)
    cur = _model_projector(am)
    return {
        "adapters": adapters,
        "active_model": (am or {}).get("model_id"),
        "active_model_vision": bool((am or {}).get("vision")),
        "current": (Path(cur).name if cur else None),
    }


class VisionAdapterRequest(BaseModel):
    file: str


@admin_router.post("/vision-adapter")
def set_vision_adapter(req: VisionAdapterRequest, background: BackgroundTasks):
    """Set which mmproj/projector the ACTIVE vision-capable chat model loads.
    Updates that model's backend `projector` field and restarts llama-vision +
    agent_server. The adapter must match the model's architecture or
    llama-server fails to load it (and the model stays down until reverted)."""
    fname = Path((req.file or "").strip()).name  # basename only — no path escape
    if not fname or not fname.lower().endswith(".gguf"):
        raise HTTPException(status_code=422, detail="file must be a .gguf filename")
    models_dir = Path(__file__).resolve().parent / "data" / "models"
    if not (models_dir / fname).exists():
        raise HTTPException(status_code=404, detail=f"no such adapter file: {fname!r}")

    cfg_path = _config_path()
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"cannot read config: {e}")

    am = _active_chat_model(cfg)
    if am is None:
        raise HTTPException(status_code=404, detail="no active chat model in config")
    if not am.get("vision"):
        raise HTTPException(
            status_code=409,
            detail=f"active model {am.get('model_id')!r} is not vision-capable")
    bk = (am.get("backends") or {}).get(am.get("active_backend") or "")
    if not isinstance(bk, dict):
        raise HTTPException(status_code=500, detail="active backend not found on active model")

    new_path = f"{_PROJECTOR_PREFIX}/{fname}"
    if (bk.get("projector") or "") == new_path:
        return {"status": "ok", "model_id": am.get("model_id"),
                "projector": fname, "noop": True}

    prev = bk.get("projector")
    bk["projector"] = new_path

    errors = validate_config(cfg)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "invalid agent_config.json after adapter change",
                    "errors": errors})
    try:
        _atomic_write(cfg_path, json.dumps(cfg, indent=2) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to write config: {e}")

    resp = {"model_id": am.get("model_id"), "projector": fname,
            "previous": (Path(prev).name if prev else None)}
    try:
        import docker
        docker.from_env().ping()
    except Exception as e:  # noqa: BLE001
        resp["status"] = "ok"
        resp["restart_pending"] = True
        resp["note"] = (f"config written but auto-restart unavailable ({e}); "
                        f"restart {_LLAMA_CONTAINER} + {_SELF_CONTAINER} manually to apply")
        return resp
    background.add_task(_restart_for_switch)
    resp["status"] = "switching"
    resp["note"] = (f"{_LLAMA_CONTAINER} + {_SELF_CONTAINER} restarting (~40s) to load "
                    f"adapter {fname}; reconnect and re-check to confirm")
    return resp


def _restart_self() -> None:
    """Restart ONLY agent_server (not llama-vision) so it re-reads the config
    and picks up a newly-registered model in its in-memory list. llama-vision
    keeps serving the still-active model — no multi-GB reload. Short stop
    timeout: agent_server is a stateless forwarder, so a fast SIGKILL (rather
    than the full 30s SIGTERM grace) makes the reload feel snappy."""
    import docker
    docker.from_env().containers.get(_SELF_CONTAINER).restart(timeout=3)


# arch (general.architecture) -> family used by the engine. Best-effort; the
# register form lets the user correct it.
_ARCH_FAMILY = {
    "gemma": "gemma", "gemma2": "gemma", "gemma3": "gemma", "gemma4": "gemma",
    "qwen2": "qwen", "qwen3": "qwen", "qwen35": "qwen", "qwen2vl": "qwen",
    "llama": "llama", "mistral": "mistral", "ministral": "mistral",
    "granite": "granite", "granitemoe": "granite",
    "smollm": "smollm", "smollm3": "smollm",
    "nemotron": "nemotron", "phi3": "phi",
    "bert": "bert", "nomic-bert": "bert", "xlm-roberta": "bert",
}
_EMBED_ARCHS = {"bert", "nomic-bert", "xlm-roberta", "jina-bert-v2"}


def _suggest_entry(fname: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    arch = (meta.get("architecture") or "").lower()
    family = _ARCH_FAMILY.get(arch, arch or "")
    stem = re.sub(r"\.gguf$", "", fname, flags=re.I)
    model_id = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:40] or "model"
    if arch in _EMBED_ARCHS or not meta.get("has_chat_template"):
        category = "embedding"
    else:
        category = "chat"
    ctx = meta.get("context_length")
    ctx_default = min(int(ctx), 32768) if ctx else 8192
    return {
        "category": category, "model_id": model_id,
        "name": meta.get("name") or stem, "family": family,
        "context": ctx_default, "max_context": (int(ctx) if ctx else None),
        "vision": False, "reasoning": bool(meta.get("reasoning_hint")),
    }


@admin_router.get("/discovered")
def list_discovered():
    """GGUF files present in data/models that are NOT yet registered in
    agent_config.json, each with auto-detected metadata + a suggested entry
    for the register form. mmproj/projector files are excluded (handled by
    the Vision Adapter tab)."""
    models_dir = Path(__file__).resolve().parent / "data" / "models"
    try:
        cfg = json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"cannot read config: {e}")

    registered = set()
    for group in _SWITCHABLE_CATEGORIES:
        for m in (((cfg.get("models") or {}).get(group)) or []):
            if not isinstance(m, dict):
                continue
            for bk in (m.get("backends") or {}).values():
                if isinstance(bk, dict):
                    for key in ("model_file", "projector"):
                        if bk.get(key):
                            registered.add(Path(bk[key]).name)

    from . import gguf_meta
    out = []
    for p in sorted(models_dir.glob("*.gguf")):
        if p.name in registered or "mmproj" in p.name.lower():
            continue
        try:
            meta = gguf_meta.summarize(str(p))
        except Exception as e:  # noqa: BLE001
            meta = {"architecture": None, "name": None, "context_length": None,
                    "has_chat_template": False, "reasoning_hint": False,
                    "is_vision_adapter": False, "error": str(e)}
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        out.append({"file": p.name, "size": size, **meta,
                    "suggestion": _suggest_entry(p.name, meta)})
    return {"discovered": out, "count": len(out)}


class RegisterRequest(BaseModel):
    file: str
    category: str = "chat"
    model_id: str
    name: str = ""
    family: str = ""
    context: int = 8192
    vision: bool = False
    reasoning: bool = False


@admin_router.post("/register")
def register_model(req: RegisterRequest, background: BackgroundTasks):
    """Create a new (inactive) model entry in agent_config.json from a
    discovered GGUF file, then restart agent_server so it shows up in the
    switch panel. Does NOT reload llama-vision — the model is inactive until
    you activate it from the panel."""
    category = (req.category or "chat").strip().lower()
    if category not in _SWITCHABLE_CATEGORIES:
        raise HTTPException(status_code=422,
                            detail=f"invalid category {category!r}; "
                                   f"allowed: {list(_SWITCHABLE_CATEGORIES)}")
    fname = Path((req.file or "").strip()).name      # basename only — no escape
    if not fname.lower().endswith(".gguf"):
        raise HTTPException(status_code=422, detail="file must be a .gguf filename")
    models_dir = Path(__file__).resolve().parent / "data" / "models"
    if not (models_dir / fname).exists():
        raise HTTPException(status_code=404, detail=f"no such file: {fname!r}")
    model_id = (req.model_id or "").strip()
    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", model_id):
        raise HTTPException(status_code=422,
                            detail="model_id must start alphanumeric and contain only "
                                   "lowercase letters, digits, '-', '.', '_'")

    cfg_path = _config_path()
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"cannot read config: {e}")
    cfg.setdefault("models", {}).setdefault(category, [])
    group = cfg["models"][category]
    if any(isinstance(m, dict) and (m.get("model_id") or "").strip() == model_id
           for m in group):
        raise HTTPException(status_code=409,
                            detail=f"model_id {model_id!r} already exists in {category}")

    if category == "chat":
        options = {"n-gpu-layers": -1, "flash-attn": "on", "jinja": True}
    elif category == "embedding":
        options = {"n-gpu-layers": -1, "flash-attn": "on",
                   "embedding": True, "pooling": "cls"}
    else:  # reranking
        options = {"n-gpu-layers": -1, "flash-attn": "on",
                   "embedding": True, "pooling": "rank", "reranking": True}

    entry = {
        "active": False,
        "name": (req.name or model_id),
        "model_id": model_id,
        "family": (req.family or "").strip().lower(),
        "active_backend": "llama_cpp",
        "context": int(req.context or 8192),
        "reasoning": bool(req.reasoning),
        "vision": bool(req.vision),
        "system_prompt": "",
        "sampling": {},
        "backends": {"llama_cpp": {
            "model_file": f"/agent_server_models/{fname}",
            "options": options,
        }},
    }
    group.append(entry)

    errors = validate_config(cfg)
    if errors:
        raise HTTPException(status_code=422,
                            detail={"message": "invalid agent_config.json after register",
                                    "errors": errors})
    try:
        _atomic_write(cfg_path, json.dumps(cfg, indent=2) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"failed to write config: {e}")

    resp = {"model_id": model_id, "category": category}
    try:
        import docker
        docker.from_env().ping()
    except Exception as e:  # noqa: BLE001
        resp["status"] = "ok"
        resp["restart_pending"] = True
        resp["note"] = (f"entry created but auto-restart unavailable ({e}); "
                        f"restart {_SELF_CONTAINER} to see it in the switch panel")
        return resp
    background.add_task(_restart_self)
    resp["status"] = "reloading"
    resp["note"] = (f"{_SELF_CONTAINER} reloading (~10s) to pick up {model_id!r}; "
                    f"it appears inactive in the {category} tab, ready to activate")
    return resp


# ---------------------------------------------------------------------------
# Dashboard: live status / recent calls / raw logs / memory inspector
# All endpoints below are READ-ONLY. None restart or reconfigure anything,
# so they cannot disrupt model serving on agent_server or llama-vision.
# ---------------------------------------------------------------------------
_ROUTER_URL = (os.getenv("LLAMA_SERVER_URL")
               or f"http://{_LLAMA_CONTAINER}:8500").rstrip("/")


@admin_router.get("/status")
def get_status():
    """Live operational snapshot for the dashboard: the active model, GPU
    VRAM (nvidia-smi exec'd read-only inside llama-vision), the router's
    resident models, and reachability. Pure reads - never restarts."""
    from . import main as _main
    active = getattr(_main, "ACTIVE_MODEL", {}) or {}
    out: Dict[str, Any] = {
        "active_model": {
            "model_id": active.get("model_id"),
            "display_name": active.get("name"),
            "family": active.get("family"),
            "vision": bool(active.get("vision")),
            "reasoning": bool(active.get("reasoning")),
            "context": active.get("context"),
        },
        "gpu": None,
        "resident": None,
        "router_reachable": False,
        "errors": [],
    }

    # GPU stats via nvidia-smi inside the llama-vision container. nvidia-smi
    # is a read-only query; it does not touch the running llama-server.
    try:
        import docker
        c = docker.from_env().containers.get(_LLAMA_CONTAINER)
        rc, raw = c.exec_run(
            "nvidia-smi --query-gpu=memory.total,memory.used,memory.free,"
            "utilization.gpu --format=csv,noheader,nounits")
        if rc == 0:
            line = raw.decode("utf-8", "replace").strip().splitlines()[0]
            tot, used, free, util = [p.strip() for p in line.split(",")]
            out["gpu"] = {
                "total_mb": int(tot), "used_mb": int(used),
                "free_mb": int(free), "util_pct": int(util),
            }
        else:
            out["errors"].append(f"nvidia-smi exited {rc}")
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"gpu: {e}")

    # Resident/declared models from the llama-server router (read-only GET).
    try:
        import urllib.request
        with urllib.request.urlopen(f"{_ROUTER_URL}/models", timeout=3) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        models = body.get("data") or body.get("models") or []
        resident = []
        models_dir = Path(__file__).resolve().parent / "data" / "models"

        def _size_by_name(p):
            try:
                return (models_dir / Path(p).name).stat().st_size
            except Exception:  # noqa: BLE001
                return None

        for m in models:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("name")
            # Skip the inert "[default]" placeholder preset: it declares no
            # --model (just cache defaults), is never loaded, and only
            # confuses the panel.
            if mid == "default":
                continue
            # llama-server reports load state under status.value
            # ("loaded"/"unloaded"); keep older keys as a fallback.
            state = ((m.get("status") or {}).get("value")
                     or m.get("state")
                     or ("loaded" if m.get("loaded") else None))
            # Classify role from launch flags: --pooling rank = reranking,
            # --pooling/--embeddings = embedding, --mmproj = a vision model.
            args = (m.get("status") or {}).get("args") or []
            pooling = (args[args.index("--pooling") + 1]
                       if "--pooling" in args
                       and args.index("--pooling") + 1 < len(args) else None)
            low = (mid or "").lower()
            if pooling == "rank" or "rerank" in low:
                role = "reranking"
            elif pooling or "--embeddings" in args:
                role = "embedding"
            else:
                role = "chat"
            vision = "--mmproj" in args
            resident.append({
                "id": mid, "state": state, "role": role, "vision": vision,
                "size": (m.get("meta") or {}).get("size"),
            })
            # Surface the vision adapter (mmproj) loaded alongside this model
            # as its own labelled entry, with its on-disk size.
            if vision:
                i = args.index("--mmproj")
                mmproj = args[i + 1] if i + 1 < len(args) else None
                if mmproj:
                    resident.append({
                        "id": Path(mmproj).name, "state": state,
                        "role": "vision adapter", "vision": True,
                        "size": _size_by_name(mmproj),
                    })
        out["resident"] = resident
        out["router_reachable"] = True
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"router: {e}")

    return out


@admin_router.get("/calls")
def get_calls(limit: int = 50):
    """Recent /v1/chat/completions calls (in-process ring buffer, newest
    first). Live-only: cleared on restart / model switch."""
    from . import call_log
    n = max(1, min(int(limit or 50), 200))
    return {"calls": call_log.recent(n), "stats": call_log.stats()}


@admin_router.get("/logs")
def get_logs(tail: int = 200, container: str = "agent_server"):
    """Tail recent stdout/stderr of a stack container (raw-logs escape
    hatch). Restricted to the two known containers."""
    allowed = {_SELF_CONTAINER, _LLAMA_CONTAINER, "agent_server", "llama-vision"}
    name = container if container in allowed else _SELF_CONTAINER
    n = max(1, min(int(tail or 200), 2000))
    try:
        import docker
        c = docker.from_env().containers.get(name)
        raw = c.logs(tail=n, timestamps=False)
        return {"container": name, "tail": n,
                "logs": raw.decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"cannot read logs for {name!r}: {e}")


def _thread_window_strategy():
    """The live ThreadWindowMemory instance, or None if memory is off."""
    from . import main as _main
    reg = getattr(_main, "MEMORY", None)
    if reg is None:
        return None
    try:
        return reg.get("thread_window")
    except Exception:
        return None


@admin_router.get("/memory")
def list_memory():
    """List thread_window memory threads (in-process; cleared on restart /
    model switch)."""
    strat = _thread_window_strategy()
    if strat is None or not hasattr(strat, "stats"):
        return {"policy": None, "threads": [], "count": 0,
                "note": "thread_window memory is not configured"}
    threads = strat.stats()
    threads.sort(key=lambda t: t.get("messages", 0), reverse=True)
    return {"policy": "thread_window", "threads": threads, "count": len(threads),
            "note": "in-process; cleared on restart / model switch"}


@admin_router.get("/memory/{thread_id}")
def get_memory_thread(thread_id: str):
    """Full transcript for one thread_window thread."""
    strat = _thread_window_strategy()
    if strat is None or not hasattr(strat, "transcript"):
        raise HTTPException(status_code=404,
                            detail="thread_window memory is not configured")
    msgs = strat.transcript(thread_id)
    if not msgs:
        raise HTTPException(status_code=404, detail=f"no such thread: {thread_id!r}")
    return {"thread_id": thread_id, "messages": msgs, "count": len(msgs)}


# ---------------------------------------------------------------------------
# Clients: a unified view of who is talking to agent_server, with optional
# GeoLite2 geolocation. Two sources merged into one table (a "kind" column
# distinguishes them):
#   - "socket": live Socket.IO sessions (browsers on the STT/voice path);
#     IP captured at connect time in main.py.
#   - "http":   recent /v1/chat/completions callers, aggregated by IP from
#     the call_log ring buffer (usually noted/cv backends on the Docker
#     network -> private IPs, so geo is null for them).
# Read-only; never restarts anything.
# ---------------------------------------------------------------------------
@admin_router.get("/clients")
def get_clients():
    import time as _time
    from . import geoip
    now = _time.time()
    out: List[Dict[str, Any]] = []

    # --- live Socket.IO sessions -------------------------------------------
    try:
        from . import main as _main
        sessions = getattr(_main, "_sessions", {}) or {}
        for sid, st in list(sessions.items()):
            ip = getattr(st, "ip", None) or "?"
            ca = getattr(st, "connected_at", None)
            la = getattr(st, "last_activity", None)
            out.append({
                "kind": "socket",
                "id": getattr(st, "client_id", None) or sid,
                "client_id": getattr(st, "client_id", None),
                "sid": sid,
                "ip": ip,
                "geo": geoip.lookup(ip),
                "connected_for_s": (round(now - ca, 1) if ca else None),
                "idle_for_s": (round(now - la, 1) if la else None),
                "calls": None,
                "last_model": None,
                "_sort": ca or 0,
            })
    except Exception as e:  # noqa: BLE001
        out.append({"kind": "socket", "error": str(e)})

    # --- recent HTTP /v1 callers (aggregated by IP) ------------------------
    try:
        from . import call_log
        agg: Dict[str, Dict[str, Any]] = {}
        for c in call_log.recent(200):
            ip = c.get("client") or "?"
            ts = c.get("ts")
            a = agg.get(ip)
            if a is None:
                agg[ip] = a = {"ip": ip, "calls": 0,
                               "first": ts, "last": ts, "last_model": c.get("model")}
            a["calls"] += 1
            # recent() is newest-first, so the first row per IP is the latest.
            if ts is not None:
                if a["last"] is None or ts > a["last"]:
                    a["last"] = ts
                    a["last_model"] = c.get("model")
                if a["first"] is None or ts < a["first"]:
                    a["first"] = ts
        for ip, a in agg.items():
            out.append({
                "kind": "http",
                "id": ip,
                "client_id": None,
                "sid": None,
                "ip": ip,
                "geo": geoip.lookup(ip),
                "connected_for_s": (round(now - a["first"], 1) if a["first"] else None),
                "idle_for_s": (round(now - a["last"], 1) if a["last"] else None),
                "calls": a["calls"],
                "last_model": a["last_model"],
                "_sort": a["last"] or 0,
            })
    except Exception as e:  # noqa: BLE001
        out.append({"kind": "http", "error": str(e)})

    # Most recently active first; drop the private sort key from the payload.
    out.sort(key=lambda x: x.get("_sort", 0), reverse=True)
    for r in out:
        r.pop("_sort", None)

    dbp = geoip.db_present()
    return {
        "clients": out,
        "count": len(out),
        "geoip_db_present": dbp["any"],
        "geoip_ipv4_present": dbp["v4"],
        "geoip_ipv6_present": dbp["v6"],
    }
