# app/llm_engine.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncio
from llama_cpp import Llama


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

        # Create llama.cpp instance
        # NOTE: Do NOT pass chat_format="gguf". Let llama auto-detect from GGUF metadata.
        # If that ever fails on a particular build, fall back to "qwen" (your model family).
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                logits_all=False,
                verbose=False,
            )
        except Exception:
            # Fallback: explicit handler for Qwen-family models
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                logits_all=False,
                verbose=True,
                chat_format="qwen",
            )

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
        user_text = prompt if preamble is None else f"{preamble}\n{prompt}"

        messages: List[Dict[str, str]] = []
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

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
