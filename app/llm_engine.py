# app/llm_engine.py
from __future__ import annotations

import os
import asyncio
import threading
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=os.environ.get("PY_LOGLEVEL", "INFO"))

try:
    from llama_cpp import Llama  # type: ignore
    _HAVE_LLAMA = True
except Exception:
    Llama = None  # type: ignore
    _HAVE_LLAMA = False


__all__ = ["LLMEngine", "LlamaCppEngine"]


class LLMEngine:
    """Abstract streaming engine interface."""
    async def generate_stream(
        self,
        user_text: str,
        *,
        cancel: threading.Event,
        system_prompt_path: Optional[str] = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
        preamble: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class LlamaCppEngine(LLMEngine):
    """
    Chat-completion engine using llama-cpp-python (no grammar).
    - Constructor takes model + baseline params.
    - Per-request you may override system prompt and sampling knobs.
    """

    def __init__(
        self,
        *,
        model_path: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> None:
        if not _HAVE_LLAMA:
            raise RuntimeError("llama-cpp-python is not installed")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load baseline (optional) system prompt text (can be empty)
        self._default_system: str = ""
        try:
            if system_prompt:
                sp = Path(system_prompt)
                if sp.exists():
                    self._default_system = sp.read_text(encoding="utf-8").strip()
                else:
                    # allow literal text too (but we generally pass file paths)
                    self._default_system = system_prompt.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to load system prompt '{system_prompt}': {e}") from e

        # llama.cpp constructor params
        try:
            n_threads    = int(params.get("n_threads") or 0) or (os.cpu_count() or 4)
            n_gpu_layers = int(params.get("n_gpu_layers") or 0)
            n_ctx        = int(params.get("n_ctx") or 0)
            n_batch      = params.get("n_batch")
            n_ubatch     = params.get("n_ubatch")
            flash_attn   = params.get("flash_attn")
            chat_format  = params.get("chat_format")

            cfg_verbose  = bool(params.get("verbose", False))
            env_verbose  = os.environ.get("LLAMA_VERBOSE", "").lower() in ("1", "true", "yes")
            verbose      = cfg_verbose or env_verbose

            llama_kwargs: Dict[str, Any] = dict(
                model_path=model_path,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            if n_ctx and n_ctx > 0: llama_kwargs["n_ctx"] = n_ctx
            if isinstance(n_batch, int) and n_batch > 0: llama_kwargs["n_batch"] = n_batch
            if isinstance(n_ubatch, int) and n_ubatch > 0: llama_kwargs["n_ubatch"] = n_ubatch
            if isinstance(flash_attn, bool): llama_kwargs["flash_attn"] = flash_attn
            if isinstance(chat_format, str) and chat_format: llama_kwargs["chat_format"] = chat_format

            log.debug(f"[llm] Llama kwargs: {{k: v for k, v in llama_kwargs.items() if k != 'model_path'}}")
            self._llama = Llama(**llama_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Llama model: {e}") from e

        # Baseline generation params (can be overridden per-call)
        stop = params.get("stop")
        if stop is None:
            stop = ["</s>"]
        self._base_gen: Dict[str, Any] = {
            "temperature": float(params.get("temperature", 0.6)),
            "top_k": int(params.get("top_k", 40)),
            "top_p": float(params.get("top_p", 0.9)),
            "min_p": float(params.get("min_p", 0.0)),
            "max_tokens": int(params.get("max_tokens", 512)),
            "stop": stop,
            "stream": True,
        }

    def __repr__(self) -> str:
        return f"<LlamaCppEngine model_path=? max_tokens={self._base_gen.get('max_tokens','?')}>"

    # ---------- internal sync generator ----------

    def _sync_stream(
        self,
        user_text: str,
        cancel: threading.Event,
        *,
        system_prompt_path: Optional[str],
        sampling_overrides: Optional[Dict[str, Any]],
        preamble: Optional[str],
    ):
        # System prompt resolution
        system_text = self._default_system
        if system_prompt_path:
            p = Path(system_prompt_path)
            if p.exists():
                try:
                    system_text = p.read_text(encoding="utf-8").strip()
                except Exception as e:
                    raise RuntimeError(f"Failed to read system prompt at '{p}': {e}") from e

        # Optional preface (e.g., memory snippet)
        user_payload = f"{preamble}\n\n{user_text}" if preamble else user_text

        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_payload})

        # Merge generation params
        call_params = dict(self._base_gen)
        if sampling_overrides:
            for k, v in sampling_overrides.items():
                if v is not None:
                    call_params[k] = v

        try:
            stream = self._llama.create_chat_completion(messages=messages, **call_params)
        except Exception as e:
            raise RuntimeError(f"Llama.create_chat_completion failed: {e}") from e

        for part in stream:
            if cancel.is_set():
                break
            try:
                ch0 = part["choices"][0]
                delta = ch0.get("delta", {}).get("content") or ch0.get("message", {}).get("content") or ""
            except Exception as e:
                raise RuntimeError(f"Malformed streamed part: {e} | part={part!r}") from e
            if delta:
                yield delta

    # ---------- public async generator ----------

    async def generate_stream(
        self,
        user_text: str,
        *,
        cancel: threading.Event,
        system_prompt_path: Optional[str] = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
        preamble: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Async bridge with backpressure — yields tokens immediately."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)

        def safe_put(item: Optional[str]) -> None:
            fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
            try:
                fut.result()
            except Exception:
                pass

        def producer():
            try:
                for chunk in self._sync_stream(
                    user_text,
                    cancel,
                    system_prompt_path=system_prompt_path,
                    sampling_overrides=sampling_overrides,
                    preamble=preamble,
                ):
                    if cancel.is_set():
                        break
                    safe_put(chunk)
            except Exception as e:
                safe_put(f"__ERROR__:{e}")
            finally:
                safe_put(None)

        fut = loop.run_in_executor(None, producer)

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, str) and item.startswith("__ERROR__:"):
                    raise RuntimeError(item.split(":", 1)[1])
                if cancel.is_set():
                    break
                yield item
        finally:
            try:
                await asyncio.wait_for(asyncio.shield(fut), timeout=0.2)
            except Exception:
                pass
