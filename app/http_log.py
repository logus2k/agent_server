# app/http_log.py
"""In-memory aggregation of HTTP requests by client IP, for the admin Clients
view. Records EVERY HTTP request (admin pages, API, /v1, static) so a real
browser visitor — via the reverse proxy's X-Forwarded-For — shows up with its
geolocation, not just /v1 API callers.

Live-only and best-effort: process-local (cleared on restart / model switch),
and record() never raises so it can't affect serving.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

_MAX_IPS = 500
_lock = threading.Lock()
_by_ip: Dict[str, Dict[str, Any]] = {}   # ip -> {ip,count,first,last,last_path,last_method}


def record(ip: str, path: str, method: str = "GET") -> None:
    """Count one request from `ip`. Never raises."""
    if not ip:
        return
    try:
        now = time.time()
        with _lock:
            e = _by_ip.get(ip)
            if e is None:
                if len(_by_ip) >= _MAX_IPS:
                    # Evict the least-recently-seen IP to bound memory.
                    oldest = min(_by_ip, key=lambda k: _by_ip[k]["last"])
                    _by_ip.pop(oldest, None)
                e = _by_ip[ip] = {"ip": ip, "count": 0, "first": now,
                                  "last": now, "last_path": path, "last_method": method}
            e["count"] += 1
            e["last"] = now
            e["last_path"] = path
            e["last_method"] = method
    except Exception:
        pass


def all() -> List[Dict[str, Any]]:
    with _lock:
        return [dict(v) for v in _by_ip.values()]


def clear() -> None:
    """Drop all recorded HTTP-request clients so the list starts fresh."""
    with _lock:
        _by_ip.clear()


def remove(ip: str) -> None:
    """Drop a single recorded client by IP."""
    with _lock:
        _by_ip.pop(ip, None)
