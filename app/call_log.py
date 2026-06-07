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
