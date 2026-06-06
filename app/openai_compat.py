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
	# OpenAI-compatible structured-output hint. Forwarded verbatim to
	# llama-server. Two real shapes:
	#   {"type": "json_object"}                                          (loose)
	#   {"type": "json_schema", "json_schema": {"name": .., "schema": {...}}}  (strict)
	# llama-server's strict schema mode enforces JSON at sampler level,
	# including proper backslash escaping inside string values — the only
	# reliable cure for the `\lambda`/`\partial` etc. JSON-escape failures
	# that bit noted-graph's community summarizer on academic content.
	# Without this field declared, Pydantic silently dropped it (same
	# class of bug as chat_template_kwargs above).
	response_format: Optional[Dict[str, Any]] = None
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

	# 2) Active chat model match (by model_id — the neutral forward id)
	active_id = (ACTIVE_MODEL.get("model_id") or "").strip().lower()
	if key == active_id:
		return None, None, {}, active_id

	# 3) Any configured chat model (active OR not). llama-server runs in
	#    router mode, so a non-active model is loaded on demand (autoload,
	#    evicting LRU per --models-max) — it need not be the active one. Return
	#    the model's OWN sampling as overrides so each model uses its own
	#    defaults regardless of which one is currently active. This is what
	#    lets a notebook drive a different LLM per pipeline phase (e.g.
	#    generate with one model, judge with another).
	for m in MODELS:
		mid = (m.get("model_id") or "").strip().lower()
		if key == mid:
			return None, None, dict(m.get("sampling") or {}), mid

	# 4) Not found
	available = sorted(list(AGENTS.keys()) + [(m.get("model_id") or "").strip().lower() for m in MODELS])
	raise HTTPException(status_code=404, detail=_oai_error(
		f"Model '{model_field}' not found. Available: {available}",
		"model_not_found", 404))


def _merge_request_params(
	engine_defaults: dict,
	preset_overrides: dict,
	request: ChatCompletionRequest,
	model_family: str = "gemma",
) -> dict:
	"""Three-tier: engine defaults < preset overrides < explicit request fields.

	For the Gemma family, injects '<eos>' as a stop string. The Gemma
	chat_format only stops on '<end_of_turn>\\n', but when tools=[...] is
	passed the model often emits the literal text '<eos>' after a tool call.
	Without that stop, generation runs past it and produces malformed
	leftover tokens (observed: '<eos><eos>voice>...' where the '<voice>'
	opening got consumed by the post-stop decode). Confirmed root-cause
	repro 2026-04-27. Qwen/Phi stop on their own EOS tokens, so the literal
	'<eos>' injection is gated on the active family.
	"""
	merged = dict(engine_defaults)
	for k, v in preset_overrides.items():
		if k in ("max_tokens", "temperature", "top_k", "top_p", "min_p", "stop",
				 "thinking_budget_tokens"):
			merged[k] = v
		elif k == "chat_template_kwargs" and isinstance(v, dict):
			# Allow presets to set template-level switches (e.g.
			# {"enable_thinking": false} for tight structured-output presets).
			# Per-request kwargs below override these if present.
			merged.setdefault("chat_template_kwargs", {}).update(v)
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
	# Gemma-only: include '<eos>' (literal text) in stop, appending (not
	# replacing) any user/preset-provided stops. See the docstring for why.
	user_stops = merged.get("stop") or []
	if isinstance(user_stops, str):
		user_stops = [user_stops]
	if model_family == "gemma" and "<eos>" not in user_stops:
		user_stops = list(user_stops) + ["<eos>"]
	merged["stop"] = user_stops
	# Pass-through for template-level switches (enable_thinking, etc.).
	# Used by noted-graph's chat_json to disable the reasoning channel for
	# structured-output calls; llama-server forwards this dict to the
	# Jinja template's render context.
	if request.chat_template_kwargs is not None:
		merged["chat_template_kwargs"] = request.chat_template_kwargs
	if request.response_format is not None:
		merged["response_format"] = request.response_format
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
				# Dynamic placeholders. {{today_utc}} explicitly inserts
				# the current UTC datetime where the prompt author placed
				# it. ALSO, every system prompt gets a small preamble
				# carrying today's date automatically — so future agents
				# don't have to remember to opt in and stale-date bugs
				# (e.g. researcher choosing 2023 dates in 2026) become
				# impossible by default. The explicit placeholder is
				# preserved for prompts that want today's date inlined
				# at a specific spot in their instructions.
				from datetime import datetime, timezone
				now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
				if "{{today_utc}}" in sys_text:
					sys_text = sys_text.replace("{{today_utc}}", now_utc)
				preamble = f"Today's UTC date and time: {now_utc}."
				sys_text = f"{preamble}\n\n{sys_text}"
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
	try:
		import sys as _sys
		print(f"[MODEL_REQ] client={request.client.host if request.client else '?'} "
		      f"requested={body.model!r} -> resolved={model_id!r}", file=_sys.stderr, flush=True)
	except Exception:
		pass
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
			family = getattr(worker.engine, "model_family", "gemma")
			gen_params = _merge_request_params(
				worker.engine.default_gen, preset_overrides, body, family)
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
			family = getattr(engine, "model_family", "gemma")
			gen_params = _merge_request_params(
				engine.default_gen, preset_overrides, body, family)
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

	# All configured chat models. The one with active:true is the model the
	# router currently serves; the others are switchable via
	# POST /admin/api/active-model (a client cannot request them directly —
	# only the active chat model is declared/loadable in the router preset).
	active_id = (ACTIVE_MODEL.get("model_id") or "unknown")
	for m in (MODELS or []):
		mid = (m.get("model_id") or "").strip()
		if not mid:
			continue
		data.append({
			"id": mid,
			"object": "model",
			"created": now,
			"owned_by": "local",
			"display_name": m.get("name", mid),
			"family": m.get("family", ""),
			"active": bool(m.get("active")) or mid == active_id,
			"kind": "chat",
		})

	# Each agent preset as a virtual model (always resolves to the active chat
	# model server-side); kind:"agent" so model pickers can ignore these.
	for name in sorted(AGENTS.keys()):
		data.append({
			"id": name,
			"object": "model",
			"created": now,
			"owned_by": "local",
			"kind": "agent",
		})

	return JSONResponse(content={"object": "list", "data": data})
