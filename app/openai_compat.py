# app/openai_compat.py
"""
OpenAI-compatible REST API layer.

Provides:
  POST /v1/chat/completions  (streaming + non-streaming)
  GET  /v1/models

Zero coupling to Socket.IO, STT, TTS, memory, or router subsystems.
All state access goes through the shared WorkerPool.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Router (collected by main.py via app.include_router(openai_router))
# ---------------------------------------------------------------------------
openai_router = APIRouter(tags=["OpenAI-compatible"])


# ---------------------------------------------------------------------------
# Bearer token auth (optional)
# ---------------------------------------------------------------------------
_API_KEY: Optional[str] = (os.getenv("OPENAI_API_KEY") or "").strip() or None


async def _check_auth(authorization: Optional[str] = Header(None)):
	if _API_KEY is None:
		return
	if not authorization:
		raise HTTPException(status_code=401, detail=_oai_error(
			"Missing Authorization header", "invalid_request_error", 401))
	scheme, _, token = authorization.partition(" ")
	if scheme.lower() != "bearer" or token != _API_KEY:
		raise HTTPException(status_code=401, detail=_oai_error(
			"Invalid API key", "invalid_request_error", 401))


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
	role: str
	content: str


class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[ChatMessage]
	stream: bool = False
	temperature: Optional[float] = None
	top_p: Optional[float] = None
	top_k: Optional[int] = None
	min_p: Optional[float] = None
	max_tokens: Optional[int] = None
	stop: Optional[Union[str, List[str]]] = None
	tools: Optional[List[Dict[str, Any]]] = None
	# Accepted for compatibility, not acted upon:
	frequency_penalty: Optional[float] = None
	presence_penalty: Optional[float] = None
	n: Optional[int] = None
	user: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _oai_error(message: str, error_type: str, code: Union[int, str]) -> dict:
	return {"error": {"message": message, "type": error_type, "code": code}}


def _get_globals():
	"""Late import to avoid circular dependency with main.py."""
	from .main import POOL, AGENTS, ACTIVE_MODEL, MODELS
	return POOL, AGENTS, ACTIVE_MODEL, MODELS


def _make_model_id(name: str) -> str:
	return name.strip().lower().replace(" ", "-")


def _resolve_model(model_field: str):
	"""
	Resolve the ``model`` field from the request.

	Returns (preset_or_none, system_prompt_path, sampling_overrides, model_id).
	"""
	POOL, AGENTS, ACTIVE_MODEL, MODELS = _get_globals()
	key = model_field.strip().lower()

	# 1) Agent preset match
	if key in AGENTS:
		preset = AGENTS[key]
		return preset, preset.system_prompt_path, dict(preset.params_override), key

	# 2) Active model slug match
	active_slug = _make_model_id(ACTIVE_MODEL.get("name", ""))
	if key == active_slug or key == ACTIVE_MODEL.get("name", "").strip().lower():
		return None, None, {}, active_slug

	# 3) Inactive model match -> informative error
	for m in MODELS:
		slug = _make_model_id(m.get("name", ""))
		if key == slug or key == m.get("name", "").strip().lower():
			if not m.get("active"):
				raise HTTPException(status_code=400, detail=_oai_error(
					f"Model '{model_field}' is configured but not active. "
					f"Active model: '{ACTIVE_MODEL.get('name')}'",
					"invalid_request_error", 400))
			return None, None, {}, slug

	# 4) Not found
	available = sorted(list(AGENTS.keys()) + [active_slug])
	raise HTTPException(status_code=404, detail=_oai_error(
		f"Model '{model_field}' not found. Available: {available}",
		"model_not_found", 404))


def _merge_request_params(
	engine_defaults: dict,
	preset_overrides: dict,
	request: ChatCompletionRequest,
) -> dict:
	"""Three-tier: engine defaults < preset overrides < explicit request fields."""
	merged = dict(engine_defaults)
	for k, v in preset_overrides.items():
		if k in ("max_tokens", "temperature", "top_k", "top_p", "min_p", "stop"):
			merged[k] = v
	if request.temperature is not None:
		merged["temperature"] = request.temperature
	if request.top_p is not None:
		merged["top_p"] = request.top_p
	if request.top_k is not None:
		merged["top_k"] = request.top_k
	if request.min_p is not None:
		merged["min_p"] = request.min_p
	if request.max_tokens is not None:
		merged["max_tokens"] = request.max_tokens
	if request.stop is not None:
		merged["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]
	return merged


def _build_messages(
	request_messages: List[ChatMessage],
	system_prompt_path: Optional[str],
) -> List[Dict[str, str]]:
	"""
	Build the messages array for create_chat_completion.

	If resolving to an agent preset, prepend the agent's system prompt
	from its file. Client messages follow unmodified.
	"""
	messages: List[Dict[str, str]] = []
	if system_prompt_path:
		p = Path(system_prompt_path)
		if p.exists():
			sys_text = p.read_text(encoding="utf-8").strip()
			if sys_text:
				messages.append({"role": "system", "content": sys_text})
	for msg in request_messages:
		messages.append({"role": msg.role, "content": msg.content})
	return messages


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------
@openai_router.post("/v1/chat/completions", dependencies=[Depends(_check_auth)])
async def chat_completions(body: ChatCompletionRequest):
	POOL, AGENTS, ACTIVE_MODEL, MODELS = _get_globals()

	if POOL is None:
		raise HTTPException(status_code=503, detail=_oai_error(
			"Server is starting up, worker pool not ready",
			"server_error", 503))

	preset, system_prompt_path, preset_overrides, model_id = _resolve_model(body.model)
	messages = _build_messages(body.messages, system_prompt_path)
	if not messages:
		raise HTTPException(status_code=400, detail=_oai_error(
			"messages array is empty", "invalid_request_error", 400))

	if body.stream:
		# Streaming: worker acquired inside the async generator
		return _streaming_response(POOL, messages, preset_overrides, body, model_id)
	else:
		# Non-streaming: worker acquired and released in this scope
		async with POOL.acquire() as worker:
			gen_params = _merge_request_params(
				worker.engine.default_gen, preset_overrides, body)
			return await _non_streaming_response(
				worker.engine, messages, gen_params, model_id, tools=body.tools)


async def _non_streaming_response(engine, messages, gen_params, model_id, tools=None):
	loop = asyncio.get_running_loop()

	def _call():
		kwargs = dict(messages=messages, stream=False, **gen_params)
		if tools:
			kwargs["tools"] = tools
		return engine.llm.create_chat_completion(**kwargs)

	try:
		result = await loop.run_in_executor(None, _call)
	except Exception as e:
		raise HTTPException(status_code=500, detail=_oai_error(
			f"LLM inference error: {e}", "server_error", 500))

	result["model"] = model_id
	return JSONResponse(content=result)


def _streaming_response(pool, messages, preset_overrides, body, model_id):
	async def event_generator():
		async with pool.acquire() as worker:
			engine = worker.engine
			gen_params = _merge_request_params(
				engine.default_gen, preset_overrides, body)
			loop = asyncio.get_running_loop()

			stream_kwargs = dict(messages=messages, stream=True, **gen_params)
			if body.tools:
				stream_kwargs["tools"] = body.tools
			stream = engine.llm.create_chat_completion(**stream_kwargs)

			def _next():
				try:
					return next(stream, None)
				except StopIteration:
					return None

			try:
				while True:
					chunk = await loop.run_in_executor(None, _next)
					if chunk is None:
						break
					chunk["model"] = model_id
					yield f"data: {json.dumps(chunk)}\n\n"
			except Exception as e:
				err = _oai_error(f"Stream error: {e}", "server_error", 500)
				yield f"data: {json.dumps(err)}\n\n"
			finally:
				yield "data: [DONE]\n\n"

	return StreamingResponse(
		event_generator(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no",
		},
	)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@openai_router.get("/v1/models", dependencies=[Depends(_check_auth)])
async def list_models():
	POOL, AGENTS, ACTIVE_MODEL, MODELS = _get_globals()

	now = int(time.time())
	data = []

	# Active model as a "real" model
	active_slug = _make_model_id(ACTIVE_MODEL.get("name", "unknown"))
	data.append({
		"id": active_slug,
		"object": "model",
		"created": now,
		"owned_by": "local",
		"display_name": ACTIVE_MODEL.get("name", "unknown"),
	})

	# Each agent preset as a virtual model
	for name in sorted(AGENTS.keys()):
		data.append({
			"id": name,
			"object": "model",
			"created": now,
			"owned_by": "local",
		})

	return JSONResponse(content={"object": "list", "data": data})
