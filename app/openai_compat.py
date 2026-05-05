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

from fastapi import APIRouter, Depends, Header, HTTPException, Request
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
	# Accept either a plain string (text-only chat — the common case) or
	# a list of OpenAI-style content blocks for multimodal input. Each
	# block is a dict like
	#   {"type": "text", "text": "..."}
	# or
	#   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
	# llama-cpp-python passes the list straight to the active chat handler;
	# the vision handler (Gemma4VisionChatHandler) renders image URLs in
	# the prompt where mtmd substitutes embedding tokens.
	#
	# Optional in OpenAI's spec for assistant messages that ONLY carry
	# tool_calls (content can be null/absent). Default to empty string so
	# downstream code that assumes a string still works.
	content: Optional[Union[str, List[Dict[str, Any]]]] = ""
	# Native tool-calling fields (OpenAI standard). Required for multi-turn
	# tool flows — without these, the asf0 chat template can't render prior
	# assistant.tool_calls or match tool messages back to their calls, and
	# the model falls into infinite re-call loops because it can't see its
	# own prior actions in history.
	tool_calls: Optional[List[Dict[str, Any]]] = None
	tool_call_id: Optional[str] = None
	name: Optional[str] = None


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
	# Forwarded as-is to the engine. llama-server uses this to thread
	# template-level switches like `{"enable_thinking": false}` to the Jinja
	# template, which is the supported path to suppress Gemma 4's reasoning
	# channel for structured-output callers (noted-graph's chat_json).
	# Without this field declared, Pydantic silently dropped it.
	chat_template_kwargs: Optional[Dict[str, Any]] = None
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
	"""Three-tier: engine defaults < preset overrides < explicit request fields.

	Always injects '<eos>' as a stop string. The Gemma chat_format only
	stops on '<end_of_turn>\\n', but when tools=[...] is passed the model
	often emits the literal text '<eos>' after a tool call. Without that
	stop, generation runs past it and produces malformed leftover tokens
	(observed: '<eos><eos>voice>...' where the '<voice>' opening got
	consumed by the post-stop decode). Confirmed root-cause repro 2026-04-27.
	"""
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
	# Always include '<eos>' (literal text) in stop, regardless of source.
	# Append (don't replace) any user/preset-provided stops.
	user_stops = merged.get("stop") or []
	if isinstance(user_stops, str):
		user_stops = [user_stops]
	if "<eos>" not in user_stops:
		user_stops = list(user_stops) + ["<eos>"]
	merged["stop"] = user_stops
	# Pass-through for template-level switches (enable_thinking, etc.).
	# Used by noted-graph's chat_json to disable the reasoning channel for
	# structured-output calls; llama-server forwards this dict to the
	# Jinja template's render context.
	if request.chat_template_kwargs is not None:
		merged["chat_template_kwargs"] = request.chat_template_kwargs
	return merged


def _build_messages(
	request_messages: List[ChatMessage],
	system_prompt_path: Optional[str],
) -> List[Dict[str, Any]]:
	"""
	Build the messages array for create_chat_completion.

	If resolving to an agent preset, prepend the agent's system prompt
	from its file. Client messages follow unmodified.

	Preserves native tool-calling fields (`tool_calls`, `tool_call_id`,
	`name`) so the chat template can render prior assistant tool_calls and
	match tool responses back to their originating calls. Without this,
	multi-turn tool flows degrade into infinite re-call loops.
	"""
	messages: List[Dict[str, Any]] = []
	if system_prompt_path:
		p = Path(system_prompt_path)
		if p.exists():
			sys_text = p.read_text(encoding="utf-8").strip()
			if sys_text:
				messages.append({"role": "system", "content": sys_text})
	for msg in request_messages:
		out: Dict[str, Any] = {"role": msg.role, "content": msg.content}
		if msg.tool_calls is not None:
			out["tool_calls"] = msg.tool_calls
		if msg.tool_call_id is not None:
			out["tool_call_id"] = msg.tool_call_id
		if msg.name is not None:
			out["name"] = msg.name
		messages.append(out)
	return messages


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------
@openai_router.post("/v1/chat/completions", dependencies=[Depends(_check_auth)])
async def chat_completions(request: Request, body: ChatCompletionRequest):
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

	# DEBUG: log the actual tool names received from upstream (e.g. noted's
	# llm.py). This is the canonical view of what Gemma will see — useful
	# when the model claims a tool is "not available" despite the upstream
	# gating log saying it is. Print to stderr so it's visible in
	# `docker logs agent_server`.
	try:
		_tool_names = [t.get("function", {}).get("name") or t.get("name") for t in (body.tools or [])]
		import sys as _sys
		print(f"[INCOMING_TOOLS] count={len(_tool_names)} names={_tool_names}", file=_sys.stderr, flush=True)
	except Exception:
		pass

	if body.stream:
		# Streaming: worker acquired inside the async generator. Pass `request`
		# so the generator can poll request.is_disconnected() per token and
		# break out instead of generating until EOS / max_tokens after the
		# upstream client (e.g. noted) drops. Without this, a cancelled chat
		# keeps Gemma running on the GPU until natural completion.
		return _streaming_response(POOL, messages, preset_overrides, body, model_id, request)
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


def _streaming_response(pool, messages, preset_overrides, body, model_id, request=None):
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

			# Track whether we exited via client disconnect so we can close
			# the underlying llama_cpp generator and stop GPU work ASAP.
			# Without this, the model keeps generating until EOS/max_tokens
			# even when noted (or any other client) has dropped the SSE
			# connection — pinning GPU at 100 % well past the user giving up.
			client_dropped = False
			try:
				while True:
					if request is not None:
						try:
							if await request.is_disconnected():
								client_dropped = True
								break
						except Exception:
							pass
					chunk = await loop.run_in_executor(None, _next)
					if chunk is None:
						break
					chunk["model"] = model_id
					yield f"data: {json.dumps(chunk)}\n\n"
			except Exception as e:
				err = _oai_error(f"Stream error: {e}", "server_error", 500)
				yield f"data: {json.dumps(err)}\n\n"
			finally:
				# Close the llama_cpp stream generator. Calling .close() on
				# the generator raises GeneratorExit at its current yield
				# point, which causes its surrounding try/finally in
				# create_chat_completion to free the eval state. Wrapped
				# defensively in case the underlying object isn't a true
				# generator.
				try:
					if hasattr(stream, "close"):
						stream.close()
				except Exception:
					pass
				if not client_dropped:
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
