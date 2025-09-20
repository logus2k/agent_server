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
from .memory import MemoryRegistry, build_registry_from_config


# -----------------------------------
# Load model + runtime config
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
# allow empty model-level system prompt; agents will provide theirs:
MODEL_DEFAULT_SYSTEM_PROMPT: str = ACTIVE_MODEL.get("system_prompt", "") or ""
PARAMS: Dict[str, Any] = ACTIVE_MODEL.get("params", {})

# Grammar support is removed; fail fast if present
if "grammar_path" in ACTIVE_MODEL and (ACTIVE_MODEL.get("grammar_path") or "").strip():
	raise RuntimeError("Grammar support is disabled. Remove 'grammar_path' from agent_config.json.")


# -----------------------------------
# Agent presets (JSON files)
# -----------------------------------
@dataclass(frozen=True)
class AgentPreset:
	name: str
	system_prompt_path: Optional[str]
	params_override: Dict[str, Any]
	memory_policy: str  # "none" | "thread_window" (more later)

def _resolve_relative(base: Path, path: Optional[str]) -> Optional[str]:
	if not path:
		return None
	p = Path(path)
	if not p.is_absolute():
		p = (base / p).resolve()
	return str(p)

def load_agent_presets(dir_path: str) -> Dict[str, AgentPreset]:
	root = Path(dir_path)
	if not root.exists():
		return {}

	presets: Dict[str, AgentPreset] = {}
	for fp in sorted(root.glob("*.agent.json")):
		data = json.loads(fp.read_text(encoding="utf-8"))
		name = (data.get("name") or "").strip().lower()
		if not name:
			raise RuntimeError(f"[agents] {fp.name} missing required 'name'")

		# grammar is not supported
		if "grammar_path" in data and (data.get("grammar_path") or "").strip():
			raise RuntimeError(f"[agents] {fp.name} contains 'grammar_path' but grammar is disabled.")

		# Only 'system_prompt' (file path) is supported to avoid ambiguity
		if "system_prompt_path" in data:
			raise RuntimeError(f"[agents] {fp.name} uses 'system_prompt_path'. Use 'system_prompt' only.")

		sys_prompt = _resolve_relative(fp.parent, data.get("system_prompt"))
		params = data.get("params_override") or {}
		policy = (data.get("memory_policy") or "none").lower()

		presets[name] = AgentPreset(
			name=name,
			system_prompt_path=sys_prompt,
			params_override=params,
			memory_policy=policy,
		)
		print(f"[agents] loaded '{name}' from {fp.name}")
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
		system_prompt=MODEL_DEFAULT_SYSTEM_PROMPT,  # agents override per-call
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
# Worker pool + agent registry + memory registry (globals)
# -----------------------------------
POOL: Optional[WorkerPool] = None
AGENTS: Dict[str, AgentPreset] = {}
MEMORY: Optional[MemoryRegistry] = None


@app.on_event("startup")
async def on_startup():
	global POOL, AGENTS, MEMORY
	print(f"Starting worker pool (size={POOL_SIZE}) for active model: {ACTIVE_MODEL.get('name')}")
	POOL = WorkerPool(factory=build_engine_or_raise, size=POOL_SIZE)

	agents_dir = (Path(__file__).parent / "agents").resolve()
	AGENTS = load_agent_presets(str(agents_dir))

	MEMORY = build_registry_from_config(RAW_CONFIG.get("memory", {}))
	print(
		f"Worker pool ready. Agents: {sorted(AGENTS.keys())} | "
		f"Memory strategies: {MEMORY.available() if MEMORY else []}"
	)


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

	# Agent preset (server-controlled)
	try:
		preset = _require_agent(data)
	except Exception as e:
		return await sio.emit("Error", {"code": "AGENT_INVALID", "message": str(e)}, to=sid)

	# User text
	text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
	if not text:
		return await sio.emit("Error", {"code": "EMPTY", "message": "Text is empty."}, to=sid)

	# Thread id (client provides; required by some server-side memory modes)
	thread_id: Optional[str] = None
	if isinstance(data, dict):
		tid = data.get("thread_id")
		if isinstance(tid, str):
			tid = tid.strip()
			if tid:
				thread_id = tid

	# Resolve memory strictly from the preset (server authority)
	mem_mode = (preset.memory_policy or "none").lower()
	mem_strategy = None
	if mem_mode != "none":
		if not MEMORY:
			return await sio.emit("Error", {"code": "MEM_DISABLED", "message": "Memory registry not initialized."}, to=sid)
		mem_strategy = MEMORY.get(mem_mode)
		if not mem_strategy:
			return await sio.emit("Error", {"code": "MEM_UNKNOWN", "message": f"Unknown memory mode '{mem_mode}' configured for agent."}, to=sid)
		if not thread_id:
			return await sio.emit("Error", {"code": "MEM_THREAD_REQUIRED", "message": "thread_id is required for this memory mode."}, to=sid)

	# Concurrency guard
	if state.current_task and not state.current_task.done():
		return await sio.emit("Error", {"code": "BUSY", "message": "A run is already active."}, to=sid)

	async with state.lock:
		if state.current_task and not state.current_task.done():
			return await sio.emit("Error", {"code": "BUSY", "message": "A run is already active."}, to=sid)

		state.cancel_event.clear()
		run_id = str(uuid.uuid4())
		await sio.emit("RunStarted", {"runId": run_id}, to=sid)

		async def runner():
			assistant_out: List[str] = []

			try:
				assert POOL is not None, "Worker pool not initialized"

				# Build memory preamble (if any) and record user turn
				preamble = None
				if mem_strategy and thread_id:
					preamble = await mem_strategy.preamble(thread_id)
					await mem_strategy.on_user_message(thread_id, text)

				async with POOL.acquire() as worker:
					async def _stream():
						async for chunk in worker.engine.generate_stream(
							text,
							cancel=state.cancel_event,
							system_prompt_path=preset.system_prompt_path,
							sampling_overrides=preset.params_override,
							preamble=preamble,
						):
							assistant_out.append(chunk)
							await sio.emit("ChatChunk", {"runId": run_id, "chunk": chunk}, to=sid)

					if REQ_TIMEOUT_S:
						await asyncio.wait_for(_stream(), timeout=REQ_TIMEOUT_S)
					else:
						await _stream()

				if state.cancel_event.is_set():
					await sio.emit("Interrupted", {"runId": run_id}, to=sid)
				else:
					# Persist assistant only if not interrupted
					if mem_strategy and thread_id:
						await mem_strategy.on_assistant_message(thread_id, "".join(assistant_out))
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
