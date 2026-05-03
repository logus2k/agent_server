"""Reproduce the 'tool-call-as-JSON-text leak' bug by replaying a multi-turn
conversation against /api/llm/chat. SAME client_id across turns so noted
builds the multi-turn memory the bug seems to need. Inspects each turn's
answer_text (post-</think>) for the JSON-as-text signature; stops on first
hit and dumps the failing transcript.

Reuses chat_mode_probe.py's SSE parsing approach (urllib + tokens). No
external Python deps required (no httpx).
"""
import json
import re
import sys
import time
import urllib.request
import uuid

URL = "http://localhost:8123/api/llm/chat"

SCRIPT = [
    "Explain me the difference between supervised and unsupervised learning",
    "What is agentic computing?",
    "List some design patterns for agents that can be helpful when designing agentic systems",
    "Tell me about Gradient Descent and SGD",
    "Now tell me about the ADAM optimizer",
    "compare those two",
]

LEAK_RE = re.compile(r'\{\s*"name"\s*:\s*"[a-z_]+"\s*,\s*"args"\s*:', re.IGNORECASE)


def stream_chat(client_id, message, timeout=300):
    body = json.dumps({
        "message": message,
        "client_id": client_id,
        "think_enabled": True,
        "vector_rag_enabled": True,
        "graph_rag_enabled": True,
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    tokens = []
    tool_badges = []
    errors = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "token" in ev:
                tokens.append(ev["token"])
            elif "tool_badge" in ev:
                tool_badges.append(ev["tool_badge"])
            elif "error" in ev:
                errors.append(ev["error"])
    full = "".join(tokens)
    # Split at </think> like chat_mode_probe.py does.
    answer = full
    for marker in ("</think>", "<|/think|>", "</thinking>"):
        if marker in answer:
            answer = answer.split(marker, 1)[1]
            break
    return {
        "elapsed_s": round(time.time() - t0, 2),
        "full_text": full,
        "answer_text": answer,
        "tool_badges": tool_badges,
        "errors": errors,
    }


def detect_leak(answer_text):
    m = LEAK_RE.search(answer_text or "")
    return m.group(0) if m else None


def run_session(idx, max_turns=None):
    client_id = f"leak-probe-{idx:03d}-{uuid.uuid4().hex[:8]}"
    print(f"\n=== session {idx} client_id={client_id} ===", flush=True)
    transcript = []
    turns = SCRIPT[:max_turns] if max_turns else SCRIPT
    for i, msg in enumerate(turns):
        print(f"  turn {i+1}/{len(turns)}: {msg!r}", flush=True)
        try:
            r = stream_chat(client_id, msg)
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {e}", flush=True)
            return None
        transcript.append({"user": msg, "result": r})
        leak = detect_leak(r["answer_text"])
        print(
            f"    elapsed={r['elapsed_s']}s tokens={len(r['full_text'])} "
            f"answer={len(r['answer_text'])} tool_badges={len(r['tool_badges'])} "
            f"leak={'YES' if leak else 'no'}",
            flush=True,
        )
        if leak:
            print(f"\n*** LEAK DETECTED ***", flush=True)
            print(f"    matched: {leak!r}", flush=True)
            print(f"    answer_text (first 400):\n{r['answer_text'][:400]}", flush=True)
            print(f"    answer_text (last 400):\n{r['answer_text'][-400:]}", flush=True)
            return {
                "session_idx": idx, "client_id": client_id,
                "leak_turn_index": i, "leak_user_message": msg,
                "leak_match": leak, "transcript": transcript,
            }
    return None


def main(max_sessions=4):
    for s in range(1, max_sessions + 1):
        hit = run_session(s)
        if hit:
            out_path = "/home/logus/env/assets/agent_server/data/leak_capture.json"
            with open(out_path, "w") as f:
                json.dump(hit, f, indent=2)
            print(f"\nDump written to {out_path}", flush=True)
            return
    print(f"\n{max_sessions} sessions completed, no leak captured.", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n)
