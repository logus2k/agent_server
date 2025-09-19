# app/main.py
from __future__ import annotations

import os
import uuid
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import socketio  # python-socketio (ASGI)

from .llm_engine import LLMEngine, LlamaCppEngine

CONFIG_PATH = os.getenv("AGENT_CONFIG", "agent_config.json")
cfg_file = Path(CONFIG_PATH)
if not cfg_file.exists():
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}")

with cfg_file.open("r", encoding="utf-8") as f:
    RAW_CONFIG: Dict[str, Any] = json.load(f)

MODELS: List[Dict[str, Any]] = list(RAW_CONFIG.get("models", []))
ACTIVE = [m for m in MODELS if m.get("active") is True]
if len(ACTIVE) != 1:
    raise RuntimeError('agent_config.json must have exactly one model with "active": true')

ACTIVE_MODEL = ACTIVE[0]
MODEL_PATH: str = ACTIVE_MODEL.get("path", "")
SYSTEM_PROMPT_PATH: str = ACTIVE_MODEL.get("system_prompt", "")
GRAMMAR_PATH: str = ACTIVE_MODEL.get("grammar_path", "")
PARAMS: Dict[str, Any] = ACTIVE_MODEL.get("params", {})

app = FastAPI(title="Assistant v2 (Python, llama.cpp) — Socket.IO")

@app.get("/health")
async def health():
    exists = Path(MODEL_PATH).exists()
    return JSONResponse({
        "ok": exists,
        "active_model": ACTIVE_MODEL.get("name"),
        "model_path": MODEL_PATH,
        "static_root": str((Path(__file__).parent / "static").resolve()),
        "chat_params": PARAMS,
        "note": "Sessions refused if model init fails."
    })

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

class SessionState:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.current_task: Optional[asyncio.Task] = None
        self.cancel_event = threading.Event()
        self.lock = asyncio.Lock()

_sessions: Dict[str, SessionState] = {}

def build_engine_or_raise() -> LLMEngine:
    engine = LlamaCppEngine(
        model_path=MODEL_PATH,
        system_prompt=SYSTEM_PROMPT_PATH,
        params=PARAMS,
        grammar_path=GRAMMAR_PATH or None,
    )
    print(f"[debug] built engine: {engine!r} (type={type(engine)})")
    return engine

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="socket.io")

@sio.event
async def connect(sid, environ, auth):
    try:
        engine = build_engine_or_raise()
        if not isinstance(engine, LlamaCppEngine):
            raise TypeError(f"Engine is not an instance of LlamaCppEngine (got {type(engine)})")
    except Exception as e:
        await sio.emit("Error", {"code": "ENGINE_INIT", "message": str(e)}, to=sid)
        await sio.disconnect(sid)
        return

    _sessions[sid] = SessionState(engine)
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

@sio.event
async def Chat(sid, data):
    state = _sessions.get(sid)
    if not state:
        return await sio.emit("Error", {"code": "NO_SESSION", "message": "Engine was not initialized."}, to=sid)

    if not isinstance(state.engine, LlamaCppEngine):
        return await sio.emit("Error", {"code": "BAD_ENGINE", "message": f"Invalid engine in session: {type(state.engine)}"}, to=sid)

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
        await sio.emit("RunStarted", {"runId": run_id}, to=sid)

        async def runner():
            try:
                async for chunk in state.engine.generate_stream(text, cancel=state.cancel_event):
                    await sio.emit("ChatChunk", {"runId": run_id, "chunk": chunk}, to=sid)
                if state.cancel_event.is_set():
                    await sio.emit("Interrupted", {"runId": run_id}, to=sid)
                else:
                    await sio.emit("ChatDone", {"runId": run_id}, to=sid)
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
