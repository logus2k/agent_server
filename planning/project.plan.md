# Agent Service — Architecture & Incremental Implementation Plan

This document captures the design decisions we’ve converged on and proposes a phased plan to implement a **local LLM agent service** using **Python + FastAPI + python-socketio** and **llama-cpp-python** (CUDA-accelerated). It’s meant to be self-contained so we can restart from scratch at any time.

---

## 1) Goals (near-term and eventual)

* **Single service** that streams LLM output to web clients with minimal latency.
* **Model:**

  * Use `llama-cpp-python` **chat-completions** API.
  * Run on GPU (CUDA wheel) when available.
* **Agents:**

  * **Router (one-shot, stateless)** — evaluates each user query with its own context & strict JSON output shape (via GBNF grammar, optional).
  * **Topic Agent (optionally stateful)** — may use conversation history per topic or session.
* **Streaming:** forward every chunk as the model produces it; no extra buffering.
* **Config-driven:** models, prompts, and (later) agents/memory declared in JSON.
* **Scalability (later):** worker pool per model, pluggable memory policies, and orchestration.

---

## 2) Current baseline (works today)

* **Server:** FastAPI (ASGI) + python-socketio.
* **Frontend:** a single `test.html` page served from `/static` to verify round-trip streaming.
* **LLM engine:** `Llama.create_chat_completion(..., stream=True)` with immediate chunk forwarding over Socket.IO.
* **Config:** `agent_config.json` with:

  * selected model (file path, active flag, mode=`chat`),
  * system prompt **file path**, optional grammar file path (currently optional),
  * generation params (temp, top\_k, top\_p, min\_p, max\_tokens, stop),
  * model params (`n_ctx`, `n_threads`, `n_gpu_layers`, **`n_batch`**, **`n_ubatch`**, `flash_attn`, `verbose`).

**Notes:**

* For Qwen3/Jan-nano GGUF, prefer stop token `"<|im_end|>"`.
* Grammar is supported (via `LlamaGrammar`) but can be disabled while stabilizing.
* Streaming is truly chunk-by-chunk; no accumulation beyond a 1–256 sized async queue for backpressure.

---

## 3) High-level Architecture

```
[ Browser test.html ]
       │   (Socket.IO)
       ▼
[ FastAPI + socket.io server ] ───────────▶  emits 'token'/'done'/'error'
       │
       │ calls
       ▼
[ LLM Engine: LlamaCppEngine ]
   • builds messages: [system, user]
   • optional grammar (GBNF) compiled once
   • chat_completion(stream=True)
   • yields deltas immediately
```

**Later:** introduce a **Worker Pool** per model and a **Memory Store** for Topic Agents; Router remains stateless.

---

## 4) Agents & Roles

### 4.1 Router Agent (stateless, one-shot)

* Purpose: classify user intent / routing decisions.
* Context: **system prompt + user input only** (no history).
* Output: strict JSON (e.g., `{"Operation":"LOCATE","Term":"Boston",...}`) enforced by **GBNF**.
* Failure behavior: if grammar fails to load, fallback to free-text (only while stabilizing); preferred behavior is to **fail fast** and report error.

### 4.2 Topic Agent (optional memory)

* Purpose: produce content on a topic; may use history.
* Memory options (to be added incrementally):

  * **none** — stateless (like Router).
  * **session window (k)** — last *k* turns in current session (ephemeral).
  * **topic window (k, topic\_id)** — last *k* turns persisted across sessions.
  * **topic summary** — rolling summary + last *k* most recent turns.
  * **RAG** (later) — retrieve top-k snippets from an index.

**Key principle:** *role* (Router/Topic) is a **request-time decision**. The engine stays the same; we change system prompt, grammar, and memory policy per request.

---

## 5) Configuration

### 5.1 Models block (current)

```json
{
  "models": [
    {
      "name": "jan-nano-128k-Q4_K_M",
      "path": "./models/jan-nano-128k-Q4_K_M.gguf",
      "active": true,
      "mode": "chat",
      "system_prompt": "./router_system_prompt.txt",
      "grammar_path": "",
      "params": {
        "n_ctx": 8192,
        "n_threads": 0,
        "n_gpu_layers": -1,
        "n_batch": 2048,
        "n_ubatch": 2048,
        "flash_attn": true,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.1,
        "max_tokens": 512,
        "stop": ["<|im_end|>"]
      }
    }
  ]
}
```

### 5.2 Future extensions (phased)

* `prompts`: `{ "router": "path", "topic/wiki": "path", ... }`
* `grammars`: `{ "router_json": "path", ... }`
* `agents`: presets mapping `{name, model, prompt_id, grammar_id, default_memory}`.
* `runtime.pools`: `{ "<model>": { "size": 2 } }`.

---

## 6) Socket Protocol (minimal but future-proof)

### Client → Server (`run`)

```json
{
  "agent": "router" | "topic",            // optional now; default uses active model
  "input": "user text...",
  "session_id": "optional-session-id",
  "topic_id": "optional-topic-id",
  "memory_mode": "default" | "none" | "session" | "topic",
  "prompt_id": "router|topic/wiki",        // future
  "grammar_id": "router_json|null",        // future
  "trace": false
}
```

### Server → Client (stream)

* `token`: `{ "text": "<delta>" }`    (multiple)
* `done`:  `{ "finish_reason": "stop|length|error", "usage": {...} }`
* `error`: `{ "message": "..." }`

**Today:** we always use the active model and the configured system prompt; Router/Topic flags are informational until we wire agent presets.

---

## 7) Streaming, Backpressure & Error Surfacing

* The engine consumes the llama stream and **immediately** forwards every `delta.content`.
* A tiny asyncio queue (size 1–256) provides backpressure; the producer thread blocks on `q.put(...)` so nothing accumulates.
* Any exception in llama stream is captured and sent upstream as a single `error` followed by end-of-stream.

---

## 8) GPU Acceleration (llama-cpp-python)

* Prefer **prebuilt CUDA wheel** matching your CUDA runtime (e.g., `cu125`).
* Verify with `verbose=True` at model init:

  * Expect `layer … assigned to device CUDA` and **cuBLAS** mentions.
  * At runtime, `nvidia-smi` should show utilization.
* Tunables that affect throughput:

  * `n_gpu_layers: -1` (full offload; reduce if VRAM tight),
  * `n_batch` / `n_ubatch`: 1536–4096 on a 4090 is typical for 4B models,
  * `flash_attn: true` if supported by your build.

---

## 9) Grammar (GBNF) Usage

* For **Router**, we’ll enforce strict JSON via a small, well-tested `.gbnf` file.
* Load grammar **once** into `LlamaGrammar` (compiled object) and reuse.
* If a grammar parse error occurs, **fail visibly** with the parse diagnostic.
* Keep grammar optional per model/agent while we validate stability.

---

## 10) Memory Strategy (planned)

Define a `MemoryStore` and `MemoryPolicy` abstraction (pluggable):

* `MemoryStore` (start with simple implementations):

  * **Session (in-proc)**: dict `{session_id: [turns]}` with TTL.
  * **Topic (SQLite/JSONL)**: `(topic_id, ts, role, content, tokens)`.
  * **Summary**: `(topic_id -> summary_text)`, updated when size > threshold.

* `MemoryPolicy`:

  * `NonePolicy`: ignore history.
  * `SessionWindow(k)`: fetch last k turns from session.
  * `TopicWindow(k, topic_id)`: last k turns from persistent store.
  * `TopicSummary(topic_id, k)`: summary + last k turns.

**Token budgeting:** fit `system + user + (summary?) + recent turns` into `n_ctx - max_new_tokens - safety`. Drop oldest first; keep summary if present.

---

## 11) Worker Pool (planned)

* Pool **workers bound to a model**; each holds one `Llama` instance (same ctor params).
* Workers are stateless aside from **caches** (prompts, grammars).
* Scheduler: a `PriorityQueue` so **Router** jobs have higher priority than Topic jobs.
* Start with pool size **2** for the active model; tune based on VRAM and latency.

**Why pool workers (not agents):** roles are just different prompts/grammars/memory—cheap to swap per request. Reloading models is expensive.

---

## 12) Phased, Incremental Plan

### Phase 0 — Baseline (✅ you have most of this)

* [x] FastAPI + python-socketio server.
* [x] Serve `/static/test.html`.
* [x] `LlamaCppEngine` with **chat-completion** streaming + immediate deltas.
* [x] `agent_config.json` for the active model; use **system prompt from file**.
* [x] GPU wheel install docs & verification.

**Validation:** open `/test.html`, send a prompt, confirm token streaming & CUDA logs.

---

### Phase 1 — Router Agent (stateless, JSON)

* [ ] Create `prompts/router.txt` (system prompt).
* [ ] Create `grammars/router.gbnf` (strict JSON).
* [ ] Add config entries for these (keep per-model for now).
* [ ] In request payload, allow `"agent":"router"`; load router prompt and grammar; ignore any history.
* [ ] On grammar parse error, return clear diagnostics (no silent fallback).

**Deliverable:** Live routing that emits valid JSON chunks/complete JSON.

---

### Phase 2 — Topic Agent (no memory)

* [ ] Create `prompts/topic.txt` (or several topic presets).
* [ ] Allow `"agent":"topic"` with `"memory_mode":"none"`.
* [ ] No grammar by default; free-text output.

**Deliverable:** Topic answers with the same streaming behavior.

---

### Phase 3 — Session Memory (ephemeral)

* [ ] `MemoryStore(session)` in-proc dict w/ TTL (configurable).
* [ ] `SessionWindow(k)` policy; allow `"memory_mode":"session"` + `"session_id"`.
* [ ] Token budgeting; include last *k* turns if they fit.

**Deliverable:** Topic Agent can maintain short context during a session.

---

### Phase 4 — Topic Memory (persistent)

* [ ] `MemoryStore(topic)` with SQLite or JSONL.
* [ ] `TopicWindow(k, topic_id)` policy; allow `"memory_mode":"topic"` + `"topic_id"`.
* [ ] Persist user/assistant turns + metadata.

**Deliverable:** Topic context persists across sessions.

---

### Phase 5 — Summarized Memory

* [ ] Background or on-demand summarization after threshold.
* [ ] `TopicSummary(topic_id, k)` combining summary + recent turns.
* [ ] Minimal summarizer prompt; keep tokens small.

**Deliverable:** Long-running topics stay within context limits.

---

### Phase 6 — Worker Pool & Scheduler

* [ ] Pool per active model: size N (start with 2).
* [ ] Priority scheduling: Router > Topic.
* [ ] Grammar/prompt caches per pool.

**Deliverable:** Parallelism with sane resource usage.

---

### Phase 7 — RAG (optional, later)

* [ ] Document embeddings & index (e.g., local FAISS).
* [ ] `RAG(k)` policy to prepend retrieved snippets.

---

## 13) Operational Notes

* **Start server from the correct venv** (the one with the CUDA wheel):

  ```
  /path/to/env/agent_server/bin/uvicorn app.main:asgi_app --host 0.0.0.0 --port 7701
  ```
* **Verify CUDA** at startup (`verbose=True` once), and runtime (`watch -n 0.5 nvidia-smi`).
* **Tune batch sizes** (`n_batch`, `n_ubatch`) for your GPU; reduce if OOM.
* **Stop tokens:** for Qwen3 chat template: `"<|im_end|>"`.
* **Error handling:** surface grammar/model errors verbosely to clients.

---

## 14) Testing Checklist

* **Smoke:** send “hello” and observe live streaming.
* **Throughput:** prompt >1k tokens; confirm GPU prompt speed (hundreds–thousands tok/s).
* **Router JSON:** malformed vs well-formed inputs; ensure strict JSON output or clear error.
* **Session memory:** send multi-turn and verify last k turns inclusion.
* **Topic memory:** restart server and confirm persisted context loads by `topic_id`.
* **Backpressure:** simulate slow client; ensure producer blocks without memory leaks.

---

## 15) Security & Safety (initial stance)

* **File access:** only allow **whitelisted** prompt/grammar IDs from config—not arbitrary paths from clients.
* **Resource caps:** global `max_tokens` and per-request timeouts/cancel.
* **Logging:** avoid logging raw PII; redact tokens if needed.
* **CORS:** restrict in production.

---

## 16) What to implement next (actionable)

1. Add **Router prompt** + **Router grammar** files and wire `"agent":"router"` path.
2. Keep Topic Agent stateless initially (`memory_mode:"none"`).
3. Add minimal **SessionWindow(k)** behind a feature flag and verify token budgeting.
4. Once stable, introduce **TopicWindow** (SQLite) and a worker pool of **2**.
