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
from .stt_manager import STTManager


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
	memory_policy: str  # "none" | "thread_window" | future

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

		if "grammar_path" in data and (data.get("grammar_path") or "").strip():
			raise RuntimeError(f"[agents] {fp.name} contains 'grammar_path' but grammar is disabled.")

		# We ONLY support 'system_prompt' (file path). No aliases.
		if "system_prompt_path" in data:
			raise RuntimeError(f"[agents] {fp.name} uses 'system_prompt_path'. Use 'system_prompt' only.")

		sys_prompt = _resolve_relative(fp.parent, data.get("system_prompt"))
		params     = data.get("params_override") or {}
		policy     = (data.get("memory_policy") or "none").strip().lower()

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
		# (Legacy fields left in place; the multiplexed STT path does not use them)
		self.stt_client: Optional[socketio.AsyncClient] = None
		self.stt_url: Optional[str] = None
		self.stt_client_id: Optional[str] = None
		self.stt_agent_name: Optional[str] = None
		self.stt_thread_id: Optional[str] = None

_sessions: Dict[str, SessionState] = {}


# -----------------------------------
# Worker pool + agent registry + memory registry (globals)
# + STT multiplexing manager and index
# -----------------------------------
POOL: Optional[WorkerPool] = None
AGENTS: Dict[str, AgentPreset] = {}
MEMORY: Optional[MemoryRegistry] = None

@dataclass
class _SttSubscription:
	client_id: str
	sid: str
	agent: str
	thread_id: Optional[str]
	stt_url: str

STT: Optional[STTManager] = None
CLIENT_INDEX: Dict[str, _SttSubscription] = {}


@app.on_event("startup")
async def on_startup():
	global POOL, AGENTS, MEMORY, STT
	print(f"Starting worker pool (size={POOL_SIZE}) for active model: {ACTIVE_MODEL.get('name')}")
	POOL = WorkerPool(factory=build_engine_or_raise, size=POOL_SIZE)

	agents_dir = (Path(__file__).parent / "agents").resolve()
	AGENTS = load_agent_presets(str(agents_dir))

	MEMORY = build_registry_from_config(RAW_CONFIG.get("memory", {}))

	# --- STT Manager: one connection per STT URL, many room subscriptions ---
	async def _on_stt_transcript(client_id: str, text: str, duration: float, stt_url: str):
		sub = CLIENT_INDEX.get(client_id)
		if not sub:
			# Unknown or already unsubscribed — ignore silently
			return
		try:
			# Resolve preset + mem policy from agent name stored in the mapping
			preset = _require_agent_by_name(sub.agent)
			mem_mode = (preset.memory_policy or "none").strip().lower()
			# Log once to be sure we are getting transcripts server-side (not proxied by the browser)
			print(f"[stt→agent_server] {client_id} ({duration:.1f}s) @ {stt_url}: {text!r}")
			await _run_text_with_preset_and_mem(
				sid=sub.sid,
				text=text,
				preset=preset,
				mem_mode=mem_mode,
				thread_id=sub.thread_id if mem_mode != "none" else None,
			)
		except Exception as e:
			# Surface as an error to that same browser sid
			await sio.emit("Error", {"code": "STT_ROUTE_ERROR", "message": str(e)}, to=sub.sid)

	STT = STTManager(on_transcript=_on_stt_transcript)

	print(f"Worker pool ready. Agents: {sorted(AGENTS.keys())} | Memory strategies: {MEMORY.available() if MEMORY else []}")


@app.on_event("shutdown")
async def on_shutdown():
	global STT
	try:
		if STT is not None:
			await STT.aclose()
	except Exception:
		pass


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
	if state:
		# cancel active run
		state.cancel_event.set()
		task = state.current_task
		if task:
			try:
				await asyncio.wait_for(task, timeout=1.0)
			except Exception:
				task.cancel()
		# (legacy) per-session stt_client disconnect if present
		if state.stt_client is not None:
			try:
				await state.stt_client.disconnect()
			except Exception:
				pass

	# Unsubscribe any STT rooms owned by this sid
	for cid, sub in list(CLIENT_INDEX.items()):
		if sub.sid == sid:
			try:
				assert STT is not None
				await STT.unsubscribe(sub.stt_url, cid)
			except Exception:
				pass
			CLIENT_INDEX.pop(cid, None)


def _require_agent_by_name(agent_name: str) -> AgentPreset:
	name = (agent_name or "").strip().lower()
	if not name:
		raise ValueError("Missing 'agent' in payload")
	if name not in AGENTS:
		raise ValueError(f"Unknown agent '{name}'. Available: {sorted(AGENTS.keys())}")
	return AGENTS[name]

def _require_agent(data: Any) -> AgentPreset:
	if not isinstance(data, dict):
		raise ValueError("Payload must be an object")
	return _require_agent_by_name(data.get("agent"))


def _parse_memory_request(data: Any):
	"""
	Accepts either:
	  - memory: "none" | "thread_window"
	  - memory: { mode: "none" | "thread_window", thread_window?: { max_context_tokens?: int } }
	Also parses: thread_id: str
	Returns: (mem_mode, thread_id, thread_window_cfg_dict)
	"""
	mem_mode = "none"
	thread_id = None
	thread_window = {}

	if isinstance(data, dict):
		raw_mem = data.get("memory")
		if isinstance(raw_mem, str):
			mem_mode = (raw_mem or "none").strip().lower() or "none"
		elif isinstance(raw_mem, dict):
			mode_val = raw_mem.get("mode")
			if isinstance(mode_val, str):
				mem_mode = (mode_val or "none").strip().lower() or "none"
			tw = raw_mem.get("thread_window")
			if isinstance(tw, dict):
				mct = tw.get("max_context_tokens")
				if isinstance(mct, int) and mct > 0:
					thread_window["max_context_tokens"] = mct

		tid = data.get("thread_id")
		if isinstance(tid, str):
			tid = tid.strip()
			if tid:
				thread_id = tid

	# normalize
	if mem_mode not in ("none", "thread_window"):
		mem_mode = "none"

	return mem_mode, thread_id, thread_window


async def _run_text_with_preset_and_mem(
	sid: str,
	text: str,
	preset: AgentPreset,
	mem_mode: str,
	thread_id: Optional[str],
) -> None:
	"""
	Shared run path used by both Chat (typed input) and STT transcript events.
	"""
	state = _sessions.get(sid)
	if not state:
		await sio.emit("Error", {"code": "NO_SESSION", "message": "No session."}, to=sid)
		return

	# resolve memory strategy
	mem_strategy = None
	if mem_mode != "none":
		if not MEMORY:
			await sio.emit("Error", {"code": "MEM_DISABLED", "message": "Memory registry not initialized."}, to=sid)
			return
		mem_strategy = MEMORY.get(mem_mode)
		if not mem_strategy:
			await sio.emit("Error", {"code": "MEM_UNKNOWN", "message": f"Unknown memory mode '{mem_mode}'."}, to=sid)
			return
		if not thread_id:
			await sio.emit("Error", {"code": "MEM_THREAD_REQUIRED", "message": "thread_id is required for this memory mode."}, to=sid)
			return

	# busy guard
	if state.current_task and not state.current_task.done():
		await sio.emit("Error", {"code": "BUSY", "message": "A run is already active."}, to=sid)
		return

	async with state.lock:
		if state.current_task and not state.current_task.done():
			await sio.emit("Error", {"code": "BUSY", "message": "A run is already active."}, to=sid)
			return

		state.cancel_event.clear()
		run_id = str(uuid.uuid4())
		await sio.emit("RunStarted", {"runId": run_id}, to=sid)

		async def runner():
			assistant_out: List[str] = []
			try:
				assert POOL is not None, "Worker pool not initialized"

				# Build preamble from memory (if any)
				preamble = None
				if mem_strategy and thread_id:
					preamble = await mem_strategy.preamble(thread_id)
					# record user message first
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
					# Persist assistant message only if not interrupted
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


# -----------------------------------
# Events: Chat / Interrupt (typed path)
# -----------------------------------
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

	# Keep existing behavior: client can still request memory mode for typed runs
	mem_mode, thread_id, _thread_window_cfg = _parse_memory_request(data)
	await _run_text_with_preset_and_mem(sid, text, preset, mem_mode, thread_id)


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


# -----------------------------------
# Event: JoinSTT — subscribe agent_server to STT transcripts for a clientId
# (multiplexed via STTManager: one connection per URL, many rooms)
# -----------------------------------
@sio.event
async def JoinSTT(sid, data):
	"""
	Payload:
	{
		"sttUrl": "http://localhost:2700",
		"clientId": "<uuid used by the browser with STT>",
		"agent": "router" | "topic",
		"threadId": "<optional/required if agent has memory>"
	}
	"""
	state = _sessions.get(sid)
	if not state:
		return await sio.emit("Error", {"code": "NO_SESSION", "message": "No session."}, to=sid)

	if not isinstance(data, dict):
		return await sio.emit("Error", {"code": "BAD_REQUEST", "message": "Payload must be an object"}, to=sid)

	stt_url = (data.get("sttUrl") or "").strip()
	client_id = (data.get("clientId") or "").strip()
	agent_name = (data.get("agent") or "").strip().lower()
	thread_id = (data.get("threadId") or "").strip() or None

	if not stt_url or not client_id or not agent_name:
		return await sio.emit("Error", {"code": "MISSING_PARAMS", "message": "sttUrl, clientId and agent are required"}, to=sid)

	# resolve preset & memory requirement
	try:
		preset = _require_agent_by_name(agent_name)
	except Exception as e:
		return await sio.emit("Error", {"code": "AGENT_INVALID", "message": str(e)}, to=sid)

	mem_mode = (preset.memory_policy or "none").strip().lower()
	if mem_mode not in ("none", "thread_window"):
		return await sio.emit("Error", {"code": "MEM_UNKNOWN", "message": f"Unknown memory mode '{mem_mode}' in preset"}, to=sid)

	if mem_mode != "none" and not thread_id:
		return await sio.emit("Error", {"code": "THREAD_REQUIRED", "message": "thread_id is required for this agent"}, to=sid)

	# Register mapping and subscribe on shared STT link
	try:
		assert STT is not None, "STT manager not initialized"
		CLIENT_INDEX[client_id] = _SttSubscription(
			client_id=client_id,
			sid=sid,
			agent=agent_name,
			thread_id=thread_id,
			stt_url=stt_url,
		)
		await STT.subscribe(stt_url, client_id)
	except Exception as e:
		# cleanup mapping if subscribe fails
		CLIENT_INDEX.pop(client_id, None)
		return await sio.emit("Error", {"code": "STT_CONNECT", "message": str(e)}, to=sid)

	await sio.emit("STTSubscribed", {"clientId": client_id, "sttUrl": stt_url, "agent": agent_name}, to=sid)


@sio.event
async def LeaveSTT(sid, data):
	client_id = (data.get("clientId") or "").strip() if isinstance(data, dict) else ""
	sub = CLIENT_INDEX.get(client_id)
	if not sub or sub.sid != sid:
		return
	try:
		assert STT is not None
		await STT.unsubscribe(sub.stt_url, client_id)
	except Exception:
		pass
	CLIENT_INDEX.pop(client_id, None)
	await sio.emit("STTUnsubscribed", {"clientId": client_id}, to=sid)
