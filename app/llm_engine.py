# app/llm_engine.py
from __future__ import annotations

import os
import asyncio
import threading
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List

# ---- logging ----
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=os.environ.get("PY_LOGLEVEL", "INFO"))


try:
    from llama_cpp import Llama, LlamaGrammar  # type: ignore
    _HAVE_LLAMA = True
except Exception:
    Llama = None  # type: ignore
    LlamaGrammar = None  # type: ignore
    _HAVE_LLAMA = False


__all__ = ["LLMEngine", "LlamaCppEngine"]


class LLMEngine:
    """Abstract streaming engine interface."""
    async def generate_stream(self, prompt: str, *, cancel: threading.Event) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class LlamaCppEngine(LLMEngine):
    """
    Chat-completion engine using llama-cpp-python with optional GBNF grammar.

    Supported chat params (generation): temperature, top_k, top_p, min_p, max_tokens, stop, stream.
    Supported model params (constructor): n_ctx, n_threads, n_gpu_layers, n_batch, n_ubatch, flash_attn, verbose.
    """

    def __init__(
        self,
        *,
        model_path: str,
        system_prompt: str,                 # path to .txt OR literal string
        params: Dict[str, Any],             # includes n_ctx, n_threads, n_gpu_layers + sampling knobs
        grammar: Optional[str] = None,      # inline GBNF (optional)
        grammar_path: Optional[str] = None, # path to .gbnf (optional)
    ) -> None:
        if not _HAVE_LLAMA:
            raise RuntimeError("llama-cpp-python is not installed")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # ----- System prompt -----
        try:
            if os.path.exists(system_prompt):
                with open(system_prompt, "r", encoding="utf-8") as f:
                    self._system = f.read().strip()
            else:
                self._system = (system_prompt or "").strip()
            if not self._system:
                raise ValueError("System prompt is empty after loading.")
        except Exception as e:
            raise RuntimeError(f"Failed to load system prompt '{system_prompt}': {e}") from e

        # ----- Build constructor kwargs for Llama -----
        try:
            n_threads    = int(params.get("n_threads") or 0) or (os.cpu_count() or 4)
            n_gpu_layers = int(params.get("n_gpu_layers") or 0)
            n_ctx        = int(params.get("n_ctx") or 0)

            # Optional GPU-utilization knobs
            n_batch      = params.get("n_batch")
            n_ubatch     = params.get("n_ubatch")
            flash_attn   = params.get("flash_attn")
            chat_format  = params.get("chat_format")  # rarely needed; GGUF template usually auto-detected
            # Verbose control (either via config or env)
            cfg_verbose  = bool(params.get("verbose", False))
            env_verbose  = os.environ.get("LLAMA_VERBOSE", "").lower() in ("1", "true", "yes")
            verbose      = cfg_verbose or env_verbose

            llama_kwargs: Dict[str, Any] = dict(
                model_path=model_path,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            if n_ctx and n_ctx > 0:
                llama_kwargs["n_ctx"] = n_ctx
            if isinstance(n_batch, int) and n_batch > 0:
                llama_kwargs["n_batch"] = n_batch
            if isinstance(n_ubatch, int) and n_ubatch > 0:
                llama_kwargs["n_ubatch"] = n_ubatch
            if isinstance(flash_attn, bool):
                llama_kwargs["flash_attn"] = flash_attn
            if isinstance(chat_format, str) and chat_format:
                llama_kwargs["chat_format"] = chat_format

            log.debug(f"[llm] Llama kwargs: { {k: v for k, v in llama_kwargs.items() if k != 'model_path'} }")

            self._llama = Llama(**llama_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Llama model: {e}") from e

        # ----- Chat generation params -----
        # Keep these conservative to avoid API mismatches; expand only with supported args.
        try:
            stop = params.get("stop")
            if stop is None:
                stop = ["</s>"]  # can override in config; for Qwen3 chat, prefer ["<|im_end|>"]

            self._gen: Dict[str, Any] = {
                "temperature": float(params.get("temperature", 0.6)),
                "top_k": int(params.get("top_k", 40)),
                "top_p": float(params.get("top_p", 0.9)),
                "min_p": float(params.get("min_p", 0.0)),
                "max_tokens": int(params.get("max_tokens", 512)),
                "stop": stop,
                "stream": True,
            }
        except Exception as e:
            raise RuntimeError(f"Invalid generation params: {e}") from e

        # ----- Grammar handling -----
        self._grammar_obj = None
        try:
            gbnf_text: Optional[str] = None
            if grammar_path and isinstance(grammar_path, str) and grammar_path.strip():
                if os.path.exists(grammar_path):
                    with open(grammar_path, "r", encoding="utf-8") as gf:
                        gbnf_text = gf.read()
                else:
                    # If a non-empty path was configured but not found, surface it clearly
                    raise FileNotFoundError(f"grammar_path does not exist: {grammar_path}")
            elif grammar is not None and isinstance(grammar, str) and grammar.strip():
                gbnf_text = grammar

            if gbnf_text:
                if LlamaGrammar is None:
                    raise RuntimeError("This llama-cpp-python build lacks LlamaGrammar support.")
                self._grammar_obj = LlamaGrammar.from_string(gbnf_text)
                log.info("[llm] Loaded GBNF grammar from %s",
                         grammar_path if grammar_path else "inline string")
        except Exception as e:
            # Surface grammar errors at runtime as a generation error message
            raise RuntimeError(f"Failed to prepare grammar: {e}") from e

        log.debug(f"[llm] Built engine: {self!r}")

    def __repr__(self) -> str:
        return f"<LlamaCppEngine model_path=? max_tokens={self._gen.get('max_tokens','?')}>"

    # ---------- internal sync generator ----------

    def _sync_stream(self, user_prompt: str, cancel: threading.Event):
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = dict(self._gen)
        if self._grammar_obj is not None:
            kwargs["grammar"] = self._grammar_obj  # pass the compiled LlamaGrammar

        try:
            stream = self._llama.create_chat_completion(messages=messages, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Llama.create_chat_completion failed: {e}") from e

        for part in stream:
            if cancel.is_set():
                break
            try:
                ch0 = part["choices"][0]
                # streaming delta for new tokens; some builds fall back to "message"
                delta = ch0.get("delta", {}).get("content") or ch0.get("message", {}).get("content") or ""
            except Exception as e:
                raise RuntimeError(f"Malformed streamed part: {e} | part={part!r}") from e
            if delta:
                yield delta

    # ---------- public async generator ----------

    async def generate_stream(self, prompt: str, *, cancel: threading.Event) -> AsyncGenerator[str, None]:
        """
        Async wrapper around the blocking llama.cpp stream.
        Bridges a producer thread to an asyncio.Queue using run_coroutine_threadsafe
        so we have real backpressure and no 'coroutine was never awaited' warnings.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)

        def safe_put(item: Optional[str]) -> None:
            """Thread-safe, blocking put into an asyncio.Queue (backpressure)."""
            fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
            try:
                fut.result()  # block this producer thread until enqueued
            except Exception:
                # If loop is closing or cancelled, just stop producing
                pass

        def producer():
            try:
                for chunk in self._sync_stream(prompt, cancel):
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
