# app/memory.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, List, Any


# ---------------------------
# Base strategy interface
# ---------------------------
class MemoryStrategy:
    """
    Stateless interface; implementations typically maintain their own state
    keyed by thread_id.
    """

    name: str

    async def preamble(self, thread_id: str) -> Optional[str]:
        """Return a textual preamble to prepend before the user prompt."""
        return None

    async def on_user_message(self, thread_id: str, text: str) -> None:
        """Record a user message for the given thread."""
        return None

    async def on_assistant_message(self, thread_id: str, text: str) -> None:
        """Record an assistant message for the given thread."""
        return None


# ---------------------------
# Utilities
# ---------------------------

def _char_budget_from_tokens(tokens: int) -> int:
    # Rough heuristic: ~4 chars per token (safe, conservative).
    # We keep it simple for now; can wire a real tokenizer later.
    return max(64, tokens * 4)


@dataclass
class ThreadWindowConfig:
    max_context_tokens: int = 1024  # Only knob we agreed to keep


class ThreadWindowMemory(MemoryStrategy):
    """
    Simple, in-process memory that keeps a rolling window of messages per thread.

    Storage shape:
      _store[thread_id] = List[{"role": "user"|"assistant", "content": str}]

    Preamble is created by concatenating "ROLE: content" lines and trimming to
    character budget derived from max_context_tokens.
    """

    name = "thread_window"

    def __init__(self, cfg: ThreadWindowConfig):
        self._cfg = cfg
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._lock = asyncio.Lock()

    async def preamble(self, thread_id: str) -> Optional[str]:
        async with self._lock:
            msgs = list(self._store.get(thread_id, []))

        if not msgs:
            return None

        # Build a simple transcript text
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)

        # Trim by char budget derived from tokens
        budget = _char_budget_from_tokens(self._cfg.max_context_tokens)
        if len(transcript) <= budget:
            return transcript

        # Keep tail-most characters (most recent messages weigh more)
        return transcript[-budget:]

    async def on_user_message(self, thread_id: str, text: str) -> None:
        if not thread_id:
            return
        entry = {"role": "user", "content": text}
        async with self._lock:
            self._store.setdefault(thread_id, []).append(entry)

    async def on_assistant_message(self, thread_id: str, text: str) -> None:
        if not thread_id:
            return
        entry = {"role": "assistant", "content": text}
        async with self._lock:
            self._store.setdefault(thread_id, []).append(entry)


# ---------------------------
# Registry / factory
# ---------------------------

class MemoryRegistry:
    """
    Holds instantiated strategies configured at startup.
    """

    def __init__(self):
        self._strategies: Dict[str, MemoryStrategy] = {}

    def register(self, strategy: MemoryStrategy) -> None:
        key = strategy.name.strip().lower()
        self._strategies[key] = strategy

    def get(self, name: str) -> Optional[MemoryStrategy]:
        return self._strategies.get((name or "").strip().lower())

    def available(self) -> List[str]:
        return sorted(self._strategies.keys())


def build_registry_from_config(cfg: Dict[str, Any]) -> MemoryRegistry:
    """
    Config shape:
    {
      "strategies": {
        "thread_window": { "max_context_tokens": 1024 }
      }
    }
    """
    reg = MemoryRegistry()
    strategies = (cfg or {}).get("strategies", {}) or {}

    # thread_window (optional)
    if "thread_window" in strategies:
        tw_cfg = strategies["thread_window"] or {}
        max_ctx = int(tw_cfg.get("max_context_tokens", 1024))
        reg.register(ThreadWindowMemory(ThreadWindowConfig(max_context_tokens=max_ctx)))

    # You can add more registrable strategies here later (vector stores, postgres, etc.)

    return reg
