# app/llm_engine.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncio

# llama_cpp is lazy-imported inside LlamaCppEngine.__init__ so this module
# stays importable in slim agent_server images that ship without
# llama-cpp-python (forwarding-only mode via LlamaServerEngine).


@dataclass
class LLMEngine:
	async def generate_stream(
		self,
		prompt: str,
		*,
		cancel: threading.Event,
		system_prompt_path: Optional[str] = None,
		sampling_overrides: Optional[Dict[str, Any]] = None,
		preamble: Optional[str] = None,
	) -> AsyncGenerator[str, None]:
		raise NotImplementedError


class LlamaCppEngine(LLMEngine):
	def __init__(
		self,
		*,
		model_path: str,
		system_prompt: str = "",
		params: Optional[Dict[str, Any]] = None,
	) -> None:
		# Lazy import — slim agent_server images don't ship llama-cpp-python.
		# This raises a clear error if someone tries to use in-process mode
		# in a forwarding-only image.
		try:
			from llama_cpp import Llama
			from llama_cpp.llama_cache import LlamaRAMCache
		except ImportError as e:
			raise RuntimeError(
				"LlamaCppEngine requires llama-cpp-python, which is not installed "
				"in this image (slim/forwarding-only build). Set LLAMA_SERVER_URL "
				"to use the LlamaServerEngine instead, or rebuild from "
				"Dockerfile.v2.fat for in-process mode."
			) from e
		self._Llama = Llama
		self._LlamaRAMCache = LlamaRAMCache

		self.model_path = model_path
		self.default_system_prompt = system_prompt or ""
		self.params = params or {}

		if not Path(self.model_path).exists():
			raise FileNotFoundError(f"Model not found: {self.model_path}")

		# Core model load params
		n_ctx = int(self.params.get("n_ctx", 4096))
		n_threads_cfg = int(self.params.get("n_threads", 0))
		n_threads = n_threads_cfg if n_threads_cfg > 0 else None  # None = auto
		n_gpu_layers = int(self.params.get("n_gpu_layers", 0))
		# FlashAttention: opt-in per model in agent_config.json's params.
		# Composes with the KV cache (different optimization: FA reshapes
		# how a single attention pass is computed; KV cache reuses K/V
		# across decode steps). On long-context decoder LLMs (e.g. Gemma
		# at n_ctx=131072) this halves attention memory and speeds up
		# prefill ~1.5-2x. Default off so models that don't play well
		# with FA fall back to the standard kernel.
		flash_attn = bool(self.params.get("flash_attn", False))

		# Optional explicit chat_format override (keeps auto-detect by default)
		chat_format_cfg: Optional[str] = self.params.get("chat_format")

		# Optional vision support. When `mmproj_path` is configured for the
		# active model, build a vision-capable chat handler that loads the
		# matching multimodal projector and routes image content blocks
		# through llama.cpp's mtmd C-API. The handler ALSO supplies the
		# chat template, so any explicit `chat_format` override is ignored
		# when vision is on (the two are mutually exclusive in llama-cpp-python).
		mmproj_path: Optional[str] = self.params.get("mmproj_path")
		chat_handler = None
		if mmproj_path:
			from app.chat_handlers.gemma4_vision import Gemma4VisionChatHandler
			chat_handler = Gemma4VisionChatHandler(
				clip_model_path=mmproj_path,
				verbose=False,
			)

		# Create llama.cpp instance
		# NOTE: Prefer auto-detect from GGUF metadata. If an override is provided,
		# we use it. If load fails for any reason, fall back to a known format.
		try:
			llama_kwargs = dict(
				model_path=self.model_path,
				n_ctx=n_ctx,
				n_threads=n_threads,
				n_gpu_layers=n_gpu_layers,
				flash_attn=flash_attn,
				logits_all=False,
				verbose=False,
			)
			if chat_handler is not None:
				llama_kwargs["chat_handler"] = chat_handler
			elif chat_format_cfg:
				llama_kwargs["chat_format"] = chat_format_cfg
			self.llm = self._Llama(**llama_kwargs)
		except Exception:
			# Fallback: explicit handler for Qwen-family models. Skipped
			# entirely when vision is on — re-raising preserves the real
			# error rather than masking it with a wrong-format retry.
			if chat_handler is not None:
				raise
			self.llm = self._Llama(
				model_path=self.model_path,
				n_ctx=n_ctx,
				n_threads=n_threads,
				n_gpu_layers=n_gpu_layers,
				flash_attn=flash_attn,
				logits_all=False,
				verbose=True,
				chat_format="qwen",
			)

		self.llm.set_cache(self._LlamaRAMCache(capacity_bytes=2 << 30))

		# Defaults for generation (can be overridden per-call)
		self.default_gen = {
			"max_tokens": int(self.params.get("max_tokens", 512)),
			"temperature": float(self.params.get("temperature", 0.7)),
			"top_k": int(self.params.get("top_k", 40)),
			"top_p": float(self.params.get("top_p", 0.95)),
			"min_p": float(self.params.get("min_p", 0.0)),
		}
		stops = self.params.get("stop") or []
		if isinstance(stops, (list, tuple)):
			self.default_gen["stop"] = list(stops)

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

	async def generate_stream(
		self,
		prompt: str,
		*,
		cancel: threading.Event,
		system_prompt_path: Optional[str] = None,
		sampling_overrides: Optional[Dict[str, Any]] = None,
		preamble: Optional[str] = None,
	) -> AsyncGenerator[str, None]:
		# Build messages per chat template
		system_text = (
			self._read_text_file(system_prompt_path)
			if system_prompt_path
			else (self.default_system_prompt or "")
		)

		messages: List[Dict[str, str]] = []

		# Primary system prompt
		if system_text.strip():
			messages.append({"role": "system", "content": system_text})

		# Place memory/context as a separate system turn (prevents role confusion)
		if preamble:
			preamble_str = str(preamble).strip()
			if preamble_str:
				messages.append({"role": "system", "content": f"Conversation context:\n{preamble_str}"})

		# User turn is just the current prompt (no preamble concatenation)
		messages.append({"role": "user", "content": prompt})

		gen = self._merge_sampling(sampling_overrides)

		stream = self.llm.create_chat_completion(
			messages=messages,
			stream=True,
			**gen,
		)

		loop = asyncio.get_running_loop()

		def _next_chunk():
			try:
				return next(stream, None)
			except StopIteration:
				return None

		while not cancel.is_set():
			chunk = await loop.run_in_executor(None, _next_chunk)
			if chunk is None:
				break
			try:
				choices = chunk.get("choices") or []
				if not choices:
					continue
				delta = choices[0].get("delta") or {}
				piece = delta.get("content")
				if piece:
					yield piece
			except Exception as e:
				raise RuntimeError(f"Stream decode error: {e}")

		if cancel.is_set():
			return

	def __repr__(self) -> str:
		return f"<LlamaCppEngine model_path=? max_tokens={self.default_gen.get('max_tokens', '?')}>"
