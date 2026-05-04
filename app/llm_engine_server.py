"""HTTP-forwarding LLM engine that proxies all requests to a `llama-server`
sidecar. PoC for the v2 -> llama-server migration: agent_server's API surface
(REST + Socket.IO) and clients (noted backend, etc.) stay unchanged; the
LLM call internally forwards over HTTP instead of running through the
in-process `Llama()` instance.

Wired in by `main.build_engine_or_raise()` when env var `LLAMA_SERVER_URL`
is set. When unset, the factory keeps building `LlamaCppEngine` as before,
so this module is purely additive — disabling forwarding restores prior
behaviour for instant rollback.

Per session memory `feedback_async_proxy_no_sync_io.md` and
`feedback_no_localhost_loopback_in_async_handler.md`:
- The Socket.IO chat path (`generate_stream`) is async and uses
  `httpx.AsyncClient` so it never blocks the event loop.
- The OpenAI REST path calls `engine.llm.create_chat_completion(...)` from
  inside `loop.run_in_executor(...)` — that runs in a worker thread, so the
  proxy's sync `httpx.Client` is safe there.

Thinking-stream rewrite:
  llama-server's peg-gemma4 chat template extracts Gemma's thinking phase
  into `delta.reasoning_content`, while noted's existing UI parser only
  recognises thinking when it appears inline as `<think>...</think>` text
  in `delta.content` (matches v1's llama-cpp-python output). To keep the
  noted UI working without rebuilding the noted backend, the streaming
  paths here splice `reasoning_content` chunks back into `content` chunks
  wrapped in `<think>...</think>` tags. Temporary translator until noted's
  UI learns to render `reasoning_content` natively (deferred — Launchpad
  outage blocks noted backend rebuild).

  TODO: remove `_ThinkingSplice` after Phase 6 of the migration plan
  (noted UI updated to render `reasoning_content` natively). Alternatively,
  test llama-server's `--reasoning-format none` flag — if it forces all
  reasoning into `content`, the splice is no longer needed even in the
  current shape. See migration plan Phase 3 / `feedback_llama_server_forwarding_poc_validated`.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Union

import httpx


# Feature flag — server-side voice injection fallback (Option 3 / Path C).
# When enabled, _stream_iter watches for `<voice>` in streamed content; if the
# model finishes a turn without emitting a voice block AND there is visible
# answer content to summarise, a focused secondary llama-server call generates
# a 1-2 sentence spoken summary which is wrapped in <voice>...</voice> and
# injected into the stream just before [DONE]. Default OFF so behaviour is
# byte-identical to today; flip to "true" via env var to enable.
# See: project_voice_injection_fallback.md
_FORCE_VOICE_INJECTION = os.environ.get("FORCE_VOICE_INJECTION", "false").lower() == "true"


# Matches a single <think>...</think> block (non-greedy across newlines)
# plus any trailing whitespace, so stripped messages don't end up with
# orphan blank lines. Used to enforce Google's Gemma 4 multi-turn rule:
# "only keep the final visible answer in chat history. Do not feed prior
# thought blocks back into the next turn." (Unsloth Gemma 4 docs.)
#
# noted's own pipeline already strips <think> before persisting to memory,
# so this is a defensive single-pass safety net — keeps history clean if
# any other client forwards unstripped reasoning.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_assistant_thinking(content: Any) -> Any:
    """Remove <think>...</think> blocks from a message's `content` field.
    Returns the value unchanged if it isn't a string (multimodal content
    arrays are passed through)."""
    if not isinstance(content, str):
        return content
    stripped = _THINK_BLOCK_RE.sub("", content)
    return stripped.strip() if stripped != content else content


def _expand_tool_call_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert assistant.tool_calls[].function.arguments from JSON string
    (OpenAI standard, what noted/openai-clients send) to a dict, in-place
    on a shallow copy of each message. The asf0 Gemma 4 chat template's
    rendering branches at chat_template_gemma-4.jinja:246-255 — only the
    `is mapping` branch produces the native `{key:<|"|>value<|"|>,...}`
    pipe-marker format Gemma was trained on. The `is string` branch emits
    the literal arguments string inside outer braces, producing
    `{{"key": "value"}}` (double-braced JSON) which is off-distribution.

    Bad arguments JSON is left as-is (the model's own emitted shape; if
    Gemma can't parse its prior call we let llama-server's parser cope or
    not — a malformed call shouldn't cascade into a forwarding crash)."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            out.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            out.append(msg)
            continue
        new_tool_calls = []
        mutated = False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_tool_calls.append(tc)
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                new_tool_calls.append(tc)
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    new_fn = {**fn, "arguments": parsed}
                    new_tool_calls.append({**tc, "function": new_fn})
                    mutated = True
                    continue
            new_tool_calls.append(tc)
        if mutated:
            out.append({**msg, "tool_calls": new_tool_calls})
        else:
            out.append(msg)
    return out


def _strip_history_thinking(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per Google's Gemma 4 multi-turn rule, strip the model's prior
    thoughts before re-feeding history to the model. The exception (also
    from the spec) is "if a single model turn involves function or tool
    calls, thoughts must NOT be removed between the function calls" — so
    we leave the CURRENT turn untouched.

    Heuristic for "current turn": everything from the last user message
    onward. Any assistant message BEFORE the last user message belongs
    to a prior turn; strip its <think>...</think> blocks. Assistant
    messages after the last user message are part of the in-flight
    tool-calling sequence; leave their content alone.
    """
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx <= 0:
        return messages  # No prior turns to strip
    out: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i < last_user_idx and msg.get("role") in ("assistant", "model"):
            new_content = _strip_assistant_thinking(msg.get("content"))
            if new_content is not msg.get("content"):
                msg = {**msg, "content": new_content}
        out.append(msg)
    return out

from app.llm_engine import LLMEngine


# ---------------------------------------------------------------------------
# Thinking-tag splice helper. Stateful per-stream — caller creates one and
# feeds each chunk's delta dict to it; gets back the translated string to
# emit as the next content delta (or None to skip).
# ---------------------------------------------------------------------------
class _ThinkingSplice:
	"""Stateful translator: maps `reasoning_content` chunks back into
	`<think>...</think>`-wrapped `content` chunks for legacy parsers."""

	def __init__(self) -> None:
		# State machine: 'NEUTRAL' -> 'THINKING' -> 'CONTENT'
		self._state = "NEUTRAL"

	def feed(self, delta: Dict[str, Any]) -> str:
		"""Return the text to emit as `delta.content` for this chunk.
		Empty string if there's nothing user-visible in this chunk
		(e.g., tool_calls-only chunk, role-only chunk)."""
		rc = delta.get("reasoning_content") or ""
		ct = delta.get("content") or ""
		if not rc and not ct:
			return ""
		out = ""
		# Reasoning text first (chunks usually carry one or the other,
		# but handle both-in-one-chunk just in case).
		if rc:
			if self._state == "NEUTRAL":
				out += "<think>" + rc
				self._state = "THINKING"
			else:
				out += rc
		if ct:
			if self._state == "THINKING":
				out += "</think>" + ct
				self._state = "CONTENT"
			else:
				out += ct
				self._state = "CONTENT"
		return out

	def finalise(self) -> str:
		"""Call when the stream ends. Closes `<think>` if it was left open."""
		if self._state == "THINKING":
			self._state = "CONTENT"
			return "</think>"
		return ""


# ---------------------------------------------------------------------------
# Sync proxy — exposed as `engine.llm` so openai_compat.py works unchanged
# ---------------------------------------------------------------------------
class _LlamaServerProxy:
	"""Mimics the subset of `llama_cpp.Llama` that agent_server's OpenAI REST
	path calls. Specifically:
	  - `create_chat_completion(messages=..., stream=..., **kwargs)` (sync)
	  - `set_cache(...)`  -> no-op (llama-server has its own prompt cache)

	Called from inside `loop.run_in_executor(...)` so sync HTTP is fine.
	"""

	def __init__(self, base_url: str, default_model: str = "gemma-4", request_timeout_s: float = 600.0) -> None:
		self._base_url = base_url.rstrip("/")
		self._default_model = default_model
		self._timeout = request_timeout_s

	def set_cache(self, *_args, **_kwargs) -> None:
		# No-op: llama-server manages its own prompt cache and slot reuse.
		return None

	def create_chat_completion(
		self,
		*,
		messages,
		stream: bool = False,
		**kwargs,
	) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
		# Strip prior-turn thinking so the model never sees its own past
		# thoughts in multi-turn history. Google/Unsloth: "only keep the
		# final visible answer in chat history." Without this, Gemma's
		# round-N+1 prompt contains the round-N <think>...</think> block,
		# which corrupts channel boundaries and produces the
		# "answer-inside-thinking" bug we hit. noted strips before
		# persisting; this is the defensive safety net.
		messages = _strip_history_thinking(list(messages))
		# Convert assistant.tool_calls[].function.arguments from JSON string
		# (OpenAI standard) to dict so the asf0 chat template renders them
		# in Gemma's NATIVE pipe-marker format `{key:<|"|>value<|"|>,...}`
		# instead of the string-fallback `{{"key": "value"}}` (double-braced
		# JSON). The string-fallback is off-distribution for Gemma — the
		# model fails to recognize its own prior tool calls in history and
		# loops calling the same tool repeatedly. Template branching at
		# chat_template_gemma-4.jinja:245-256.
		messages = _expand_tool_call_arguments(messages)
		payload: Dict[str, Any] = {"messages": messages, "stream": bool(stream), **kwargs}
		# Set model field so router-mode llama-server can route the request.
		# kwargs may already contain a model name (preset overrides etc.) —
		# don't clobber that.
		payload.setdefault("model", self._default_model)
		url = f"{self._base_url}/v1/chat/completions"
		if stream:
			return self._stream_iter(url, payload)
		# Non-streaming: one-shot request, return parsed dict. Splice
		# reasoning_content into content with <think>...</think> tags so the
		# response shape matches what noted (and other clients on legacy
		# parsers) expect.
		with httpx.Client(timeout=self._timeout) as client:
			r = client.post(url, json=payload)
			r.raise_for_status()
			data = r.json()
			self._splice_nonstreaming(data)
			return data

	@staticmethod
	def _splice_nonstreaming(data: Dict[str, Any]) -> None:
		"""For non-streaming responses, fold reasoning_content into content
		with <think>...</think> wrapping. Mutates `data` in place."""
		for choice in (data.get("choices") or []):
			msg = choice.get("message") or {}
			rc = msg.get("reasoning_content") or ""
			ct = msg.get("content") or ""
			if rc:
				msg["content"] = f"<think>{rc}</think>" + ct
				msg.pop("reasoning_content", None)

	def _stream_iter(self, url: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
		"""Sync generator yielding parsed SSE chunks. The `with` blocks ensure
		the connection is closed when the generator is exhausted, closed by
		the caller, or garbage-collected mid-stream (e.g., on client cancel).
		Splices reasoning_content into <think>...</think>-wrapped content.

		When _FORCE_VOICE_INJECTION is enabled (env flag, default OFF), this
		also tracks whether a `<voice>` block was emitted by the model. If the
		stream ends without one AND visible content exists, fires a focused
		secondary llama-server call to generate a 1-2 sentence spoken summary
		and injects it as a synthetic <voice>...</voice> chunk before [DONE].
		See project_voice_injection_fallback.md for the full design rationale.
		"""
		splice = _ThinkingSplice()
		# Voice-injection state (only used when _FORCE_VOICE_INJECTION).
		spliced_content_buffer: List[str] = []  # all spliced output we emitted to client
		voice_seen = False
		tool_call_seen = False
		with httpx.Client(timeout=self._timeout) as client:
			with client.stream("POST", url, json=payload) as r:
				r.raise_for_status()
				for line in r.iter_lines():
					if not line.startswith("data: "):
						continue
					body = line[6:]
					if body.strip() == "[DONE]":
						# If <think> was left open by a stream that ended
						# without ever emitting content, close it now.
						tail = splice.finalise()
						if tail:
							spliced_content_buffer.append(tail)
							if "<voice>" in tail:
								voice_seen = True
							yield self._wrap_content_chunk(tail)
						# Voice-injection fallback: if the model never emitted
						# <voice>, generate a spoken summary now and inject it
						# before [DONE]. Strict zero-regression: any failure
						# silently falls back to today's behaviour (no voice).
						if _FORCE_VOICE_INJECTION and not voice_seen:
							injected = self._maybe_inject_voice_chunk(
								spliced_content_buffer, tool_call_seen, payload
							)
							if injected:
								yield injected
						return
					try:
						chunk = json.loads(body)
					except json.JSONDecodeError:
						continue
					choices = chunk.get("choices") or []
					if not choices:
						# Pass through (usage stats, etc.)
						yield chunk
						continue
					delta = choices[0].get("delta") or {}
					# Track tool_calls deltas (separate channel from content).
					if delta.get("tool_calls"):
						tool_call_seen = True
					# tool_calls-only chunks have no content/reasoning_content;
					# pass them through unmodified so noted's tool dispatcher
					# sees the standard OpenAI shape.
					if not (delta.get("content") or delta.get("reasoning_content")):
						yield chunk
						continue
					new_text = splice.feed(delta)
					if not new_text:
						yield chunk
						continue
					# Track spliced content + voice presence for injection logic.
					# Substring check must span chunk boundaries — a single
					# delta can be smaller than the literal "<voice>" tag
					# (8 chars) when the model emits one or two characters
					# at a time. Joining the cumulative buffer is O(n) per
					# chunk but stays cheap in practice (~few thousand chars
					# per turn). Tail-only optimisation possible later.
					spliced_content_buffer.append(new_text)
					if not voice_seen and "<voice>" in "".join(spliced_content_buffer):
						voice_seen = True
					new_delta = {k: v for k, v in delta.items() if k != "reasoning_content"}
					new_delta["content"] = new_text
					new_choice = dict(choices[0])
					new_choice["delta"] = new_delta
					new_chunk = dict(chunk)
					new_chunk["choices"] = [new_choice]
					yield new_chunk

	def _maybe_inject_voice_chunk(
		self,
		spliced_buffer: List[str],
		tool_call_seen: bool,
		original_payload: Dict[str, Any],
	) -> Optional[Dict[str, Any]]:
		"""Generate and wrap a voice block for injection.

		Called only when _FORCE_VOICE_INJECTION is enabled AND the model
		finished a stream without emitting <voice>. Returns a synthetic
		content chunk to yield before [DONE], or None if injection should
		be skipped (empty body, generation failed, etc.). NEVER raises —
		any failure logs and returns None so the stream completes cleanly.

		Logs every decision to stderr (visible in `docker logs agent_server`)
		per feedback_no_silent_degradation.md — injection MUST be observable.
		"""
		try:
			full_spliced = "".join(spliced_buffer)
			# Strip <think>...</think> blocks so we summarise visible content
			# only (reasoning isn't user-facing).
			visible = re.sub(r"<think>[\s\S]*?</think>\s*", "", full_spliced).strip()
			body_chars = len(visible)
			print(
				f"[VOICE_INJECTION_TRIGGERED] body_chars={body_chars} "
				f"has_tool_call={tool_call_seen}",
				flush=True,
			)
			if not visible:
				# Nothing to summarise. This includes tool-call rounds where
				# the model emitted only `<|tool_call>...<tool_call|>` markers
				# with no visible body — they have no answer to recap. Sending
				# an empty string to the secondary llama-server call produces
				# a meta-response like "Please provide the answer you would
				# like me to summarize" which then gets played as voice.
				# The synthesis round (after the tool result returns) is the
				# turn that needs voice injection, and it ALWAYS has visible
				# content. Gate purely on `visible`, not on tool_call_seen.
				print("[VOICE_INJECTION_RESULT] status=skip_empty_body", flush=True)
				return None
			voice_text = self._generate_voice_summary(visible, original_payload)
			if not voice_text:
				print("[VOICE_INJECTION_RESULT] status=empty_response", flush=True)
				return None
			# Wrap and emit. Leading newlines so the voice block is visually
			# separated from any answer-body content already streamed (the
			# frontend's <voice> parser doesn't care about whitespace).
			print(
				f"[VOICE_INJECTION_RESULT] status=success "
				f"generated_text={voice_text!r}",
				flush=True,
			)
			return self._wrap_content_chunk(f"\n<voice>{voice_text}</voice>")
		except Exception as e:
			print(f"[VOICE_INJECTION_RESULT] status=error err={e!r}", flush=True)
			return None

	def _generate_voice_summary(
		self,
		visible_content: str,
		original_payload: Dict[str, Any],
	) -> str:
		"""Fire a focused chat completion to llama-server asking for a 1-2
		sentence spoken summary of the given visible content. Used as the
		recovery path when the model skipped its <voice> block.

		Returns the cleaned voice text (no markup) or empty string on any
		failure. Hard read timeout (5s) caps worst-case latency added to
		the user's perceived turn end. Strips defensive markup the model
		might add (`<voice>`, surrounding quotes).
		"""
		# Truncate input to keep the focused call fast even for long answers.
		truncated = visible_content[:3000]
		messages = [
			{
				"role": "system",
				"content": (
					"You are a text-to-speech summarizer. Produce ONE to TWO "
					"short, natural-sounding sentences (max 280 characters total) "
					"that summarise the assistant's answer for spoken playback. "
					"Plain prose only — no markdown, no code, no lists, no "
					"citation tags, no quotation marks. Output ONLY the summary "
					"text — no <voice> tags, no labels, no explanation."
				),
			},
			{
				"role": "user",
				"content": (
					"Summarise the following answer in 1-2 spoken sentences for "
					f"text-to-speech:\n\n{truncated}"
				),
			},
		]
		payload = {
			"model": original_payload.get("model", self._default_model),
			"messages": messages,
			"stream": False,
			"temperature": 0.4,
			"max_tokens": 120,
			# Disable thinking on this focused secondary call. With reasoning
			# enabled, Gemma's reasoning_content channel consumes the entire
			# max_tokens budget before any visible content is emitted —
			# leaving an empty `content` field. We don't need reasoning for
			# a one-off voice summary; disabling it returns the full budget
			# to actual generation. Probed empirically 2026-05-04.
			"chat_template_kwargs": {"enable_thinking": False},
		}
		# Tight timeout so a hung secondary call can't stall the primary
		# stream's [DONE] indefinitely. 5s read covers normal generation
		# of ~100 tokens with healthy headroom on the local Gemma 4.
		timeout = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
		try:
			with httpx.Client(timeout=timeout) as client:
				r = client.post(f"{self._base_url}/v1/chat/completions", json=payload)
				r.raise_for_status()
				data = r.json()
			choices = data.get("choices") or []
			if not choices:
				return ""
			text = (choices[0].get("message", {}).get("content") or "").strip()
			# Defensive cleanup: strip any <voice> tags the model added,
			# leading/trailing quotes, and stray markdown.
			text = re.sub(r"</?voice>", "", text).strip()
			text = text.strip("\"'")
			# Cap at 280 chars on a sentence boundary if possible.
			if len(text) > 280:
				cut = text[:280]
				period = cut.rfind(".")
				text = cut[: period + 1] if period > 50 else cut
			return text
		except Exception:
			return ""

	@staticmethod
	def _wrap_content_chunk(text: str) -> Dict[str, Any]:
		"""Synthesise a delta.content chunk for the splice's tail close-tag."""
		return {
			"object": "chat.completion.chunk",
			"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
		}


# ---------------------------------------------------------------------------
# Async engine — used by Socket.IO chat path via `engine.generate_stream(...)`
# ---------------------------------------------------------------------------
class LlamaServerEngine(LLMEngine):
	def __init__(
		self,
		*,
		base_url: str,
		system_prompt: str = "",
		params: Optional[Dict[str, Any]] = None,
	) -> None:
		self.base_url = base_url.rstrip("/")
		self.default_system_prompt = system_prompt or ""
		self.params = params or {}
		# Model name used for forwarding when llama-server is in router mode
		# (multiple models hosted in one process; request must include `model`
		# field to select). Single-model llama-server ignores the field.
		# Read from agent_config.json's params, default to a v2-friendly name.
		self._llama_server_model = str(self.params.get("llama_server_model", "gemma-4"))

		# Generation defaults — same shape LlamaCppEngine produces so that
		# openai_compat.py's `_merge_request_params(engine.default_gen, ...)`
		# and the Socket.IO path's `_merge_sampling(...)` keep working.
		self.default_gen: Dict[str, Any] = {
			"max_tokens": int(self.params.get("max_tokens", 512)),
			"temperature": float(self.params.get("temperature", 0.7)),
			"top_k": int(self.params.get("top_k", 40)),
			"top_p": float(self.params.get("top_p", 0.95)),
			"min_p": float(self.params.get("min_p", 0.0)),
		}
		stops = self.params.get("stop") or []
		if isinstance(stops, (list, tuple)):
			self.default_gen["stop"] = list(stops)

		# Sync proxy used by the OpenAI REST path (called from a thread executor).
		self.llm = _LlamaServerProxy(base_url=self.base_url, default_model=self._llama_server_model)

	# --- internal helpers (mirror LlamaCppEngine) -----------------------
	def _read_text_file(self, p: Optional[str]) -> str:
		if not p:
			return ""
		path = Path(p)
		return path.read_text(encoding="utf-8")

	def _merge_sampling(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		merged = dict(self.default_gen)
		if overrides:
			merged.update(overrides)
		return merged

	# --- async streaming path used by Socket.IO Chat events -------------
	async def generate_stream(
		self,
		prompt: str,
		*,
		cancel: threading.Event,
		system_prompt_path: Optional[str] = None,
		sampling_overrides: Optional[Dict[str, Any]] = None,
		preamble: Optional[str] = None,
	) -> AsyncGenerator[str, None]:
		# Build messages identically to LlamaCppEngine — this is what
		# agent_server has shipped to the model for the Socket.IO chat path
		# all along. No semantic change.
		system_text = (
			self._read_text_file(system_prompt_path) if system_prompt_path
			else (self.default_system_prompt or "")
		)
		messages: List[Dict[str, str]] = []
		if system_text.strip():
			messages.append({"role": "system", "content": system_text})
		if preamble:
			preamble_str = str(preamble).strip()
			if preamble_str:
				messages.append({"role": "system", "content": f"Conversation context:\n{preamble_str}"})
		messages.append({"role": "user", "content": prompt})

		gen = self._merge_sampling(sampling_overrides)
		payload: Dict[str, Any] = {"messages": messages, "stream": True, "model": self._llama_server_model, **gen}

		# Async http stream — never blocks the event loop. Connect fast,
		# read with generous timeout (long generations can take minutes).
		timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
		url = f"{self.base_url}/v1/chat/completions"
		splice = _ThinkingSplice()
		async with httpx.AsyncClient(timeout=timeout) as client:
			async with client.stream("POST", url, json=payload) as resp:
				resp.raise_for_status()
				async for line in resp.aiter_lines():
					if cancel.is_set():
						return
					if not line.startswith("data: "):
						continue
					body = line[6:]
					if body.strip() == "[DONE]":
						tail = splice.finalise()
						if tail:
							yield tail
						return
					try:
						chunk = json.loads(body)
					except json.JSONDecodeError:
						continue
					if not chunk.get("choices"):
						continue
					delta = chunk["choices"][0].get("delta") or {}
					piece = splice.feed(delta)
					if piece:
						yield piece

	def __repr__(self) -> str:
		return f"<LlamaServerEngine base_url={self.base_url!r} max_tokens={self.default_gen.get('max_tokens', '?')}>"
