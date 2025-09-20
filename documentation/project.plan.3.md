Here’s a fresh, self-contained plan that reflects where we are **right now** and what’s left to finish — especially the change to **server-controlled memory**.

---

# Agent Service — Updated Architecture & Incremental Plan (v3)

This document captures the **current state** of our local LLM agent service and a crisp, incremental plan to complete the next features. It’s enough to reboot a fresh session and keep moving.

## 0) What changed since the last plan

* **No grammars** anywhere (feature removed). The server fails fast if grammar keys are present.&#x20;
* **Agent presets** live in `app/agents/*.agent.json` and supply:

  * `name`
  * `system_prompt` (path, resolved relative to the agent file)
  * `params_override`
  * `memory_policy` (intent; wiring still pending, see below)&#x20;
* **Strict preset parsing**: we accept only `system_prompt` (not `system_prompt_path`). Missing/unknown agent names error out (no defaults).&#x20;
* **Worker pool** runs and serves requests; prompts are swapped per call (no model reload).&#x20;
* **Static UI split**: `test.html`, `app.js`, `styles.css`. Clean token streaming, separate **Events** log, input clears on send.
* **Client memory UI removed**. Client always sends a `thread_id`, auto-generated on first load. (Memory mode **should be server-controlled** next.)

## 1) Current snapshot (behavior today)

* **Server**: FastAPI + python-socketio. Static is mounted at `/`; Socket.IO path is `/socket.io`.&#x20;
* **Model**: `llama-cpp-python` chat completions with **true streaming**.
* **Workers**: pool size from config; each worker holds one Llama instance.&#x20;
* **Agents**: loaded from `app/agents`. Example presets:

  * `router.agent.json` → `system_prompt: ../prompts/router_system_prompt.txt`, `memory_policy: none` (or “stateless” in older file — update to `none`).
  * `topic.agent.json` → `system_prompt: ../prompts/topic_system_prompt.txt`, `memory_policy: thread_window` (what we want). Some copies still say `"topic"` — that must be fixed to `"thread_window"`.
* **Important current limitation**: the server **still looks at the request payload** for `"memory"` and normalizes to `"none"` if absent; since the client no longer sends it, the effective mode is **stateless** every time. That’s why the Topic agent isn’t retaining history yet. (See `_parse_memory_request` in `main.py`.)&#x20;

## 2) What “done” looks like (near-term)

* **Server-controlled memory**:

  * The server selects the memory strategy **exclusively** from the agent preset’s `memory_policy`.
  * The request’s `"memory"` field is **ignored** (or rejected if present).
  * If a preset requires memory (e.g., `thread_window`), the server **requires a `thread_id`** (client already always sends one).
* **Thread-window memory works**:

  * On each request, server builds a **preamble** from stored turns (bounded by a token budget), then streams the answer, and **appends** both user and assistant turns back to storage.

## 3) Configuration

### 3.1 Model/runtime (`agent_config.json`)

* One active model, with params (e.g., `n_ctx`, `n_gpu_layers`, sampling defaults).
* Runtime options include `pool_size` and optional `per_request_timeout_s`. (Already in place.)

### 3.2 Agent presets (`app/agents/*.agent.json`)

**Supported keys:**

```json
{
  "name": "router",
  "system_prompt": "../prompts/router_system_prompt.txt",
  "params_override": { "temperature": 0.2, "max_tokens": 256 },
  "memory_policy": "none"          // allowed: "none", "thread_window"
}
```

```json
{
  "name": "topic",
  "system_prompt": "../prompts/topic_system_prompt.txt",
  "params_override": { "max_tokens": 2048, "temperature": 0.6, "top_k": 40, "top_p": 0.9, "min_p": 0.1 },
  "memory_policy": "thread_window"
}
```

* Paths are resolved **relative to the agent file**; since `prompts` is a **sibling** of `agents`, use `"../prompts/..."`.&#x20;
* Any unsupported keys or missing files raise clear errors.&#x20;

## 4) Socket protocol (as it stands)

**Client → `Chat`**

```json
{ "agent": "router" | "topic", "text": "..." , "thread_id": "<uuid or string>" }
```

* We no longer expose a client memory mode. (Today, the server still reads it if present; we’re about to remove that.)

**Server → stream**

* `RunStarted`, many `ChatChunk`, then `ChatDone` (or `Interrupted` / `Error`). Streaming is immediate; no per-word splitting.

## 5) Code status re: memory

* We already have a **strategy interface** and a thread-window implementation file. (You shared `memory.py` earlier; keep using it.)
* In `main.py`, the memory mode is still derived from the **request** (`_parse_memory_request`), not from the **agent preset**. That is the root cause for Topic’s lack of memory right now.&#x20;

## 6) Minimal diffs to finish server-controlled memory

1. **Lock down supported values**

   * Accept only `memory_policy: "none" | "thread_window"` in agent presets; error otherwise (no aliases). (The loader already enforces strictness for keys like `system_prompt`.)&#x20;

2. **Select strategy from preset (not request)**

   * In `Chat`: remove `_parse_memory_request(...)`.
   * Do:

     ```
     mem_mode = preset.memory_policy
     if mem_mode == "thread_window":
         require thread_id (from payload)
         mem_strategy = MEMORY.get("thread_window")
     else:
         mem_strategy = None
     ```
   * Build `preamble = await mem_strategy.preamble(thread_id)` (if any), **record user turn** before generation, **record assistant turn** after generation (we already do this pattern).&#x20;

3. **Tidy request contract**

   * If the payload includes `memory`, **ignore it** (or emit a one-time warning). Server is the source of truth now.

4. **Preset fixes**

   * Ensure `router.agent.json` uses `"memory_policy": "none"`. (Update older copies still saying `"stateless"`.)&#x20;
   * Ensure `topic.agent.json` uses `"thread_window"` (update any `"topic"` variants).

## 7) Roadmap (crisp & incremental)

### ✅ Phase A — Solid baseline (done)

* Socket.IO server + static UI (split JS/CSS), clean streaming.
* Worker pool.
* Agents as presets, strict parsing, no grammar.

### ▶ Phase B — **Server-controlled thread memory** (next)

* Wire **preset-driven** `memory_policy`.
* Enforce `thread_id` only when the preset requires memory.
* Use `memory.py`’s thread-window strategy for preamble + append.

### ▶ Phase C — Observability

* Log `{agent, memory_policy, thread_id}` per run.
* Light counters (tokens used / context budget) in the Events panel.

### ▶ Phase D — Persistence (optional, later)

* Add `postgres` memory strategy (async SQLAlchemy). Same interface, survives restarts by reusing the same `thread_id`.

## 8) Folder layout (reference)

```
project-root/
├─ app/
│  ├─ main.py               # server (Socket.IO, routing, pool, memory wiring)
│  ├─ llm_engine.py
│  ├─ worker_pool.py
│  ├─ memory.py             # strategies (thread_window now; postgres later)
│  ├─ agents/
│  │  ├─ router.agent.json
│  │  └─ topic.agent.json
│  └─ static/
│     ├─ test.html
│     ├─ app.js
│     ├─ styles.css
│     └─ socket.io.min.js
├─ prompts/
│  ├─ router_system_prompt.txt
│  └─ topic_system_prompt.txt
└─ agent_config.json
```

## 9) Quick checks

* Agents load and resolve prompts correctly (note the `../prompts/...` paths).&#x20;
* Unknown agent names produce an **Error**; there is **no fallback**.&#x20;
* Streaming is immediate; UI shows message area and a separate Events log.
* With server-controlled memory wired, the Topic agent produces consistent follow-ups when the same `thread_id` is reused.

---

If you want, I can deliver the exact `main.py` edits for Phase B straight away (swap to `preset.memory_policy`, drop `_parse_memory_request`, and keep the rest intact).
