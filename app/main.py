# app/main.py
from __future__ import annotations

import os
import uuid
import json
import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import socketio  # python-socketio (ASGI)

from .llm_engine import LLMEngine, LlamaCppEngine
from .worker_pool import WorkerPool


# -----------------------------------
# Load model config
# -----------------------------------
CONFIG_PATH = os.getenv("AGENT_CONFIG", "agent_config.json")
cfg_file = Path(CONFIG_PATH)
if not cfg_file.exists():
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}")

RAW_CONFIG: Dict[str, Any] = json.loads(cfg_file.read_text(encoding="utf-8"))

RUNTIME: Dict[str, Any] = RAW_CONFIG.get("runtime", {}) or {}
POOL_SIZE: int = int(RUNTIME.get("pool_size", 1))
REQ_TIMEOUT_S: Optional[int] = int(RUNTIME.get("per_request_timeout_s", 0)) or None

MODELS: List[Dict[str, Any]] = list(RAW_CONFIG.get("models", []))
ACTIVE = [m for m in MODELS if m.get("active") is True]
if len(ACTIVE) != 1:
    raise RuntimeError('agent_config.json must have exactly one model with "active": true')

ACTIVE_MODEL = ACTIVE[0]
MODEL_PATH: str = ACTIVE_MODEL.get("path", "")
MODEL_DEFAULT_SYSTEM_PROMPT: str = ACTIVE_MODEL.get("system_prompt", "") or ""
PARAMS: Dict[str, Any] = ACTIVE_MODEL.get("params", {})

# Fail fast if someone left grammar keys around (we removed grammar support)
if "grammar_path" in ACTIVE_MODEL and (ACTIVE_MODEL.get("grammar_path") or "").strip():
    raise RuntimeError("Grammar support is disabled. Remove 'grammar_path' from agent_config.json.")


# -----------------------------------
# Agent presets (JSON files)
# -----------------------------------
@dataclass(frozen=True)
class AgentPreset:
    name: str
    system_prompt_path: str
    params_override: Dict[str, Any]
    memory_policy: str  # "none" | "topic" | "stateless"

def _resolve_relative(base: Path, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)

def load_agent_presets(dir_path: str) -> Dict[str, AgentPreset]:
    root = Path(dir_path).resolve()
    if not root.exists():
        raise RuntimeError(f"[agents] directory not found: {root}")

    presets: Dict[str, AgentPreset] = {}

    for fp in sorted(root.glob("*.agent.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"[agents] {fp.name}: invalid JSON: {e}") from e

        # required: name
        name = (data.get("name") or "").strip().lower()
        if not name:
            raise RuntimeError(f"[agents] {fp.name}: missing required field 'name'")
        if name in presets:
            raise RuntimeError(f"[agents] duplicate agent name '{name}' in {fp.name}")

        # required: system_prompt (path) — ONLY this key is supported
        raw_prompt = data.get("system_prompt")
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise RuntimeError(f"[agents] {fp.name}: missing required 'system_prompt' (path)")

        # resolve relative to the preset file folder (app/agents)
        sys_prompt_abs = (fp.parent / raw_prompt).resolve()
        if not sys_prompt_abs.exists():
            raise RuntimeError(f"[agents] {fp.name}: system prompt not found: {sys_prompt_abs}")

        # optional: params override must be a dict
        params = data.get("params_override") or {}
        if not isinstance(params, dict):
            raise RuntimeError(f"[agents] {fp.name}: 'params_override' must be an object")

        # memory policy (strict set)
        policy = (data.get("memory_policy") or "none").lower()
        if policy not in {"none", "topic", "stateless"}:
            raise RuntimeError(f"[agents] {fp.name}: invalid memory_policy '{policy}'")

        presets[name] = AgentPreset(
            name=name,
            system_prompt_path=str(sys_prompt_abs),
            params_override=params,
            memory_policy=policy,
        )
        print(f"[agents] loaded '{name}' (prompt={sys_prompt_abs})")

    if not presets:
        raise RuntimeError(f"[agents] no presets found in {root}")

    return presets


# -----------------------------------
# FastAPI app + static at root
# -----------------------------------
app = FastAPI(title="Assistant v2 (Python, llama.cpp) — Socket.IO")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# -----------------------------------
# Engine factory (used by worker pool)
# -----------------------------------
def build_engine_or_raise() -> LLMEngine:
    engine = LlamaCppEngine(
        model_path=MODEL_PATH,
        system_prompt=MODEL_DEFAULT_SYSTEM_PROMPT,  # may be empty; agents override per-call
        params=PARAMS,
    )
    print(f"[debug] built engine: {engine!r} (type={type(engine)})")
    return engine


# -----------------------------------
# Session state (per Socket.IO client)
# -----------------------------------
class SessionState:
    def __init__(self):
        self.current_task: Optional[asyncio.Task] = None
        self.cancel_event = threading.Event()
        self.lock = asyncio.Lock()

_sessions: Dict[str, SessionState] = {}

# -----------------------------------
# Worker pool + agent registry (globals)
# -----------------------------------
POOL: Optional[WorkerPool] = None
AGENTS: Dict[str, AgentPreset] = {}

@app.on_event("startup")
async def on_startup():
    global POOL, AGENTS
    print(f"Starting worker pool (size={POOL_SIZE}) for active model: {ACTIVE_MODEL.get('name')}")
    POOL = WorkerPool(factory=build_engine_or_raise, size=POOL_SIZE)
    AGENTS = load_agent_presets(str((Path(__file__).parent / "agents").resolve()))
    print("Worker pool ready.")


# -----------------------------------
# Socket.IO (ASGI)
# -----------------------------------
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="socket.io")

@sio.event
async def connect(sid, environ, auth):
    _sessions[sid] = SessionState()
    print(f"[sio] connected {sid}")

@sio.event
async def disconnect(sid):
    print(f"[sio] disconnected {sid}")
    state = _sessions.pop(sid, None)
    if not state:
        return
    state.cancel_event.set()
    task = state.current_task
    if task:
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except Exception:
            task.cancel()

def _require_agent(data: Any) -> AgentPreset:
    if not isinstance(data, dict):
        raise ValueError("Payload must be an object")
    agent_name = (data.get("agent") or "").strip().lower()
    if not agent_name:
        raise ValueError("Missing 'agent' in payload")
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent '{agent_name}'. Available: {sorted(AGENTS.keys())}")
    return AGENTS[agent_name]

@sio.event
async def Chat(sid, data):
    state = _sessions.get(sid)
    if not state:
        return await sio.emit("Error", {"code": "NO_SESSION", "message": "No session."}, to=sid)

    try:
        preset = _require_agent(data)
    except Exception as e:
        return await sio.emit("Error", {"code": "AGENT_INVALID", "message": str(e)}, to=sid)

    text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
    if not text:
        return await sio.emit("Error", {"code": "EMPTY", "message": "Text is empty."}, to=sid)

    if state.current_task and not state.current_task.done():
        return await sio.emit("Error", {"code": "BUSY", "message": "A run is already active."}, to=sid)

    async with state.lock:
        if state.current_task and not state.current_task.done():
            return await sio.emit("Error", {"code": "BUSY", "message": "A run is already active."}, to=sid)

        state.cancel_event.clear()
        run_id = str(uuid.uuid4())
        await sio.emit("RunStarted", {"runId": run_id, "agent": preset.name}, to=sid)

        async def runner():
            try:
                assert POOL is not None, "Worker pool not initialized"
                async with POOL.acquire() as worker:
                    async def _stream():
                        async for chunk in worker.engine.generate_stream(
                            text,
                            cancel=state.cancel_event,
                            system_prompt_path=preset.system_prompt_path,
                            sampling_overrides=preset.params_override,
                            preamble=None,  # later: memory snippet if policy != "none"
                        ):
                            await sio.emit("ChatChunk", {"runId": run_id, "chunk": chunk}, to=sid)

                    if REQ_TIMEOUT_S:
                        await asyncio.wait_for(_stream(), timeout=REQ_TIMEOUT_S)
                    else:
                        await _stream()

                if state.cancel_event.is_set():
                    await sio.emit("Interrupted", {"runId": run_id}, to=sid)
                else:
                    await sio.emit("ChatDone", {"runId": run_id}, to=sid)

            except asyncio.TimeoutError:
                state.cancel_event.set()
                await sio.emit("Error", {"runId": run_id, "message": f"Timeout after {REQ_TIMEOUT_S}s"}, to=sid)
            except Exception as e:
                await sio.emit("Error", {"runId": run_id, "message": str(e)}, to=sid)
            finally:
                state.current_task = None

        state.current_task = asyncio.create_task(runner())

@sio.event
async def Interrupt(sid):
    state = _sessions.get(sid)
    if not state:
        return
    state.cancel_event.set()
    if state.current_task:
        try:
            await asyncio.wait_for(state.current_task, timeout=2.0)
        except Exception:
            pass
    await sio.emit("Interrupted", {"runId": None}, to=sid)
