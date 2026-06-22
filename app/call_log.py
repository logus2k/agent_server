# app/call_log.py
"""In-memory ring buffer of recent /v1/chat/completions calls.

Live-only: the buffer is process-local and is therefore lost on restart
(and so on every model switch, which restarts agent_server). It exists to
feed the admin dashboard's "recent calls" panel - it is NOT an audit log.

Recording is strictly best-effort: record() never raises, so a logging
problem can never affect model serving. The hot path calls record() in a
guarded side-effect only.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List

_MAX = 200
_buf: "Deque[Dict[str, Any]]" = deque(maxlen=_MAX)
_lock = threading.Lock()
_seq = 0


def record(entry: Dict[str, Any]) -> None:
    """Append a call record (newest last). Never raises."""
    global _seq
    try:
        with _lock:
            _seq += 1
            entry.setdefault("ts", time.time())
            entry["seq"] = _seq
            _buf.append(entry)
    except Exception:
        # A telemetry failure must never propagate into request handling.
        pass


def recent(n: int = 50) -> List[Dict[str, Any]]:
    """The most recent calls, newest first."""
    with _lock:
        items = list(_buf)
    if n and n > 0:
        items = items[-n:]
    return items[::-1]


def stats() -> Dict[str, Any]:
    with _lock:
        return {"shown": len(_buf), "capacity": _MAX, "total_seen": _seq}


def cache_stats(n: int = 100) -> Dict[str, Any]:
    """Rolling llama-server prompt-cache reuse over the last n calls that
    carry cache data. Each record stores cache_n (prefix tokens reused) and
    prompt_n (tokens reprocessed); total prompt = cache_n + prompt_n.
    hit_ratio = reused / total. None when no call carries cache data yet
    (e.g. right after a restart). Never raises."""
    try:
        with _lock:
            items = list(_buf)
        if n and n > 0:
            items = items[-n:]
        total = 0
        cached = 0
        calls = 0
        for e in items:
            c = e.get("cache_n")
            p = e.get("prompt_n")
            if c is None or p is None:
                continue
            t = c + p
            if t <= 0:
                continue
            total += t
            cached += c
            calls += 1
        return {
            "calls": calls,
            "prompt_tokens": total,
            "cached_tokens": cached,
            "hit_ratio": (cached / total) if total else None,
        }
    except Exception:
        return {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "hit_ratio": None}
