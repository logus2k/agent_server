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

from fastapi import APIRouter, Body, HTTPException
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

    models = cfg.get("models")
    if not isinstance(models, list) or not models:
        errors.append("'models' must be a non-empty array")
    else:
        active = [m for m in models if isinstance(m, dict) and m.get("active") is True]
        if len(active) != 1:
            errors.append(f'exactly one model must have "active": true (found {len(active)})')
        for m in models:
            if not isinstance(m, dict):
                errors.append("each entry in 'models' must be an object")
                continue
            if not (m.get("name") or "").strip():
                errors.append("every model needs a non-empty 'name'")
            if "grammar_path" in m and (m.get("grammar_path") or "").strip():
                errors.append(f"model {m.get('name')!r}: 'grammar_path' is not allowed")

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
