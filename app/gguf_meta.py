# app/gguf_meta.py
"""Minimal, dependency-free GGUF metadata reader for the admin "register a
discovered model" flow. Reads ONLY the key/value header (not tensor data),
and skips large array values (e.g. tokenizer token lists) without decoding
them, so a scan of a multi-GB file is fast.

Exposes summarize(path) -> high-level fields + heuristics the register form
pre-fills. Best-effort: read() raises on a non-GGUF/corrupt file; callers
guard and fall back to filename-only suggestions.
"""
from __future__ import annotations

import struct
from typing import Any, Dict, Optional

# GGUF scalar value types -> fixed byte size (8 = string, 9 = array: special).
_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}


def _u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_str(f) -> str:
    n = _u64(f)
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t: int):
    if t == 8:                       # string
        return _read_str(f)
    if t == 9:                       # array — capture nothing, seek past it
        et = _u32(f)
        cnt = _u64(f)
        if et == 8:                  # array of strings: skip each len+bytes
            for _ in range(cnt):
                f.seek(_u64(f), 1)
        elif et == 9:
            raise ValueError("nested array unsupported")
        else:
            f.seek(_SCALAR_SIZE.get(et, 0) * cnt, 1)
        return None
    sz = _SCALAR_SIZE.get(t)
    if sz is None:
        raise ValueError(f"unknown gguf value type {t}")
    raw = f.read(sz)
    return {
        0: lambda: raw[0],
        1: lambda: struct.unpack("<b", raw)[0],
        2: lambda: struct.unpack("<H", raw)[0],
        3: lambda: struct.unpack("<h", raw)[0],
        4: lambda: struct.unpack("<I", raw)[0],
        5: lambda: struct.unpack("<i", raw)[0],
        6: lambda: struct.unpack("<f", raw)[0],
        7: lambda: raw[0] != 0,
        10: lambda: struct.unpack("<Q", raw)[0],
        11: lambda: struct.unpack("<q", raw)[0],
        12: lambda: struct.unpack("<d", raw)[0],
    }[t]()


def read(path: str) -> Dict[str, Any]:
    """Return the GGUF KV header as a dict (scalars only; arrays -> skipped)."""
    out: Dict[str, Any] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _u32(f)                      # version
        _u64(f)                      # tensor count
        kvc = _u64(f)                # kv count
        for _ in range(kvc):
            key = _read_str(f)
            v = _read_value(f, _u32(f))
            if v is not None:
                out[key] = v
    return out


def summarize(path: str) -> Dict[str, Any]:
    """High-level metadata + heuristics used to pre-fill the register form."""
    kv = read(path)
    arch = kv.get("general.architecture")
    name = kv.get("general.name")
    ctx: Optional[int] = None
    for k, v in kv.items():
        if k.endswith(".context_length"):
            ctx = int(v)
            break
    tmpl = kv.get("tokenizer.chat_template") or ""
    reasoning = ("<think>" in tmpl) or ("enable_thinking" in tmpl) or ("/think" in tmpl)
    is_vision = (str(arch).lower() in ("clip", "mmproj")
                 or "mmproj" in (name or "").lower())
    return {
        "architecture": arch,
        "name": name,
        "context_length": ctx,
        "has_chat_template": bool(tmpl),
        "reasoning_hint": reasoning,
        "is_vision_adapter": is_vision,
    }
