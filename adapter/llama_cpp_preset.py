#!/usr/bin/env python3
"""llama.cpp adapter: render a llama-server `--models-preset` INI from
agent_config.json.

This is the backend-specific translation layer. agent_config.json is the
neutral, backend-agnostic source of truth; this script turns the set of
*active* models (across the chat / embedding / reranking groups) into the
INI that upstream llama-server requires. It is invoked by the llama-vision
container's entrypoint (adapter/entrypoint.sh) on every boot — never by a
human — and the INI it produces lives only inside that container.

Swapping llama.cpp for another stack (e.g. vLLM) means writing a sibling
adapter that reads the same agent_config.json; nothing upstream changes.

Schema consumed:
  models: { chat: [...], embedding: [...], reranking: [...] }
Each entry is fully self-describing — there is no shared `[*]` defaults
block. Per-entry fields:
  - neutral: model_id, family (chat only; ignored here), context,
    reasoning (chat), vision (chat), sampling (ignored here — sampling is
    per-request, not preset-level), active, active_backend
  - backend specifics: backends.<active_backend> = { model_file,
    projector (if vision), options{...}, comments[] }

Rules: exactly one active model in `chat`; any number active in the other
groups. Server-process flags (host/port/models-max/cache-reuse) are NOT
here — they live in the compose command.

No GPU, container, or third-party deps required — pure stdlib text work.

Usage:
    python3 adapter/llama_cpp_preset.py            # write the preset
    python3 adapter/llama_cpp_preset.py --print    # to stdout
    python3 adapter/llama_cpp_preset.py --models-max   # print active count
    python3 adapter/llama_cpp_preset.py --check     # exit 1 if --out stale
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "data" / "config" / "agent_config.json"
DEFAULT_OUT = ROOT / "llama-router-models.ini"
BACKEND = "llama_cpp"
GROUP_ORDER = ("chat", "embedding", "reranking")


def _err(msg: str):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def collect_models(config: dict):
    """Return (entries, active_chat) where entries is a list of
    (model_dict, load_on_startup) tuples in preset order.

    EXACTLY ONE text/chat model is ever declared in the preset: the active one
    (load-on-startup=true, resident at boot). The other chat models are NOT
    emitted at all. This enforces a single resident text model at any moment.

    Why not declare the inactive chat models load-on-demand: doing so makes them
    autoloadable, so a stray request for an inactive chat model (e.g. another
    service still calling model=gemma-4) loads a SECOND text model as a 4th
    resident model, which evicts the active model and triggers VRAM eviction
    churn ("thrashing") that cold-reloads multi-GB weights mid-turn. By leaving
    them undeclared, such a request 404s harmlessly and the active text model
    stays put. (Trade-off: the per-request "pick any chat model" / notebook
    multi-model capability is intentionally given up for this stability.)

    Active models in the other groups (embedding / reranking) are emitted
    load-on-startup=true. Validates exactly one active chat model.
    """
    groups = config.get("models")
    if not isinstance(groups, dict):
        _err("agent_config.json: 'models' must be an object "
             "{chat:[...], embedding:[...], reranking:[...]}")

    chat = groups.get("chat") or []
    active_chat = [m for m in chat if isinstance(m, dict) and m.get("active") is True]
    if len(active_chat) != 1:
        _err(f'exactly one model in models.chat must have "active": true '
             f"(found {len(active_chat)})")

    entries: list[tuple[dict, bool]] = []
    # The active text/chat model is declared resident at boot. Inactive chat
    # models are deliberately omitted (so they cannot be autoloaded and evict
    # the active one) — EXCEPT those explicitly flagged "resident": true, which
    # are pinned load-on-startup so several chat models can be served at once
    # (callable by model_id via /v1). Keep --models-max high enough to fit them.
    entries.append((active_chat[0], True))
    for m in chat:
        if (isinstance(m, dict) and m.get("resident") is True
                and not m.get("active")):
            entries.append((m, True))
    # Other groups (embedding, reranking, ...): only active models, resident.
    other_groups = [g for g in (list(GROUP_ORDER) + [g for g in groups if g not in GROUP_ORDER]) if g != "chat"]
    for g in other_groups:
        for m in groups.get(g) or []:
            if isinstance(m, dict) and m.get("active") is True:
                entries.append((m, True))
    return entries, active_chat[0]


def render_value(v) -> str:
    # bool before int (bool is a subclass of int).
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def render_model_section(m: dict, load_on_startup: bool = True) -> str:
    model_id = (m.get("model_id") or "").strip()
    if not model_id:
        _err(f"model entry {m.get('name')!r} has no 'model_id'")
    ab = (m.get("active_backend") or BACKEND).strip()
    if ab != BACKEND:
        _err(f"model {model_id!r}: active_backend is {ab!r}, "
             f"but this is the {BACKEND!r} adapter")
    bk = (m.get("backends") or {}).get(BACKEND)
    if not isinstance(bk, dict):
        _err(f"model {model_id!r} has no backends.{BACKEND} block")
    model_file = (bk.get("model_file") or "").strip()
    if not model_file:
        _err(f"model {model_id!r}: backends.{BACKEND}.model_file is required")

    lines = [f"[{model_id}]"]
    for c in bk.get("comments") or []:
        lines.append(f"; {c}")
    lines.append(f"model = {model_file}")
    # vision (neutral bool) -> mmproj (llama.cpp path)
    if m.get("vision"):
        proj = (bk.get("projector") or "").strip()
        if not proj:
            _err(f"model {model_id!r}: vision is true but "
                 f"backends.{BACKEND}.projector is missing")
        lines.append(f"mmproj = {proj}")
    # context (neutral) -> c
    if "context" in m:
        lines.append(f"c = {render_value(m['context'])}")
    # reasoning (neutral bool) -> reasoning on/off  (chat models only)
    if "reasoning" in m:
        lines.append(f"reasoning = {'on' if m.get('reasoning') else 'off'}")
    # remaining backend-specific knobs, verbatim
    for k, v in (bk.get("options") or {}).items():
        lines.append(f"{k} = {render_value(v)}")
    lines.append(f"load-on-startup = {'true' if load_on_startup else 'false'}")
    return "\n".join(lines)


def render_preset(config: dict):
    entries, chat = collect_models(config)
    resident = [m.get("model_id") for m, los in entries if los]
    on_demand = [m.get("model_id") for m, los in entries if not los]
    header = (
        "; ============================================================\n"
        "; GENERATED by adapter/llama_cpp_preset.py from agent_config.json.\n"
        "; DO NOT EDIT - materialised inside the llama-vision container at\n"
        "; boot. Edit data/config/agent_config.json instead.\n"
        f"; active chat model: {chat.get('name')!r}  (model_id: {chat.get('model_id')!r})\n"
        f"; resident at boot (load-on-startup=true): {resident}\n"
        f"; declared, load-on-demand (0 VRAM until used): {on_demand}\n"
        "; (server flags --host/--port/--models-max/--cache-reuse live in the\n"
        ";  compose command, not here.)\n"
        "; ============================================================\n"
    )
    parts = [header, "version = 1"]
    for m, los in entries:
        parts.append("")
        parts.append(render_model_section(m, los))
    return "\n".join(parts) + "\n", chat, len(resident)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="print the preset to stdout")
    ap.add_argument("--models-max", dest="models_max", action="store_true",
                    help="print only the count of active (resident) models")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if --out differs from freshly generated output")
    args = ap.parse_args()

    if not args.config.exists():
        _err(f"config not found: {args.config}")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _err(f"{args.config}: invalid JSON: {e}")

    text, chat, n_active = render_preset(config)

    if args.models_max:
        print(n_active)
        return
    if args.to_stdout:
        sys.stdout.write(text)
        return
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"STALE: {args.out} differs from agent_config.json.", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {args.out} matches active chat model ({chat.get('model_id')}).")
        return

    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}  (active chat=[{chat.get('model_id')}], "
          f"{n_active} resident models)")


if __name__ == "__main__":
    main()
