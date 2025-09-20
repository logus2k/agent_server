# Agent Service — Architecture & Incremental Implementation Plan (Updated)

This document reflects our **current state** and a clear, incremental path forward for the local LLM agent service built with **FastAPI + python-socketio** and **llama-cpp-python (CUDA)**. It’s self-contained so we can reboot a fresh session and still have everything needed.

---

## 0) What changed since the last plan

* **Grammars removed entirely.** No GBNF, no grammar loading, no fallback. Simpler + fewer moving parts.
* **Agent presets are now JSON files** (one per agent) loaded from `app/agents/`. Each preset contains:

  * `name`
  * `system_prompt_path` (file path)
  * `params_override` (sampling overrides)
* **No “auto” routing mode** and **no default agent fallback**. If the request names an unknown agent, we **error fast**.
* **Worker pool is live.** Pool size comes from config. Each worker holds a model instance; requests acquire a worker and apply the selected agent’s policy per call.
* **System prompts**: the chosen agent preset supplies the **system prompt path**; we load its content on each request.
* **Static UI** split into `test.html`, `app.js`, and `styles.css`. The UI:

  * streams tokens cleanly (no per-word line breaks),
  * logs protocol events separately from chat messages,
  * **always sends a `thread_id`** (auto-generated client-side), and shows it in Events.
* **Health endpoint removed** to declutter.

---

## 1) Near-term goals

* **Minimal latency streaming** (send tokens immediately, no buffering).
* **Agent per request** (Router, Topic, etc.) via presets; **no grammar**.
* **Concurrency** via **worker pool** with isolated sessions per Socket.IO client.
* **Memory is next** (thread-scoped, then optional persistence).

---

## 2) Current snapshot (working today)

* **Server**: FastAPI (ASGI) + python-socketio, Socket.IO path `/socket.io`.
* **Static**: served at `/` (app mounts `app/static` at root); `test.html`, `app.js`, `styles.css`.
* **Model**: `llama-cpp-python` chat completion with **true token streaming** to the client.
* **GPU**: using a **prebuilt CUDA wheel** (e.g., `cu125`); full offload (`n_gpu_layers: -1`) when VRAM allows.
* **Worker pool**: initialized at startup; size from config.
* **Agents**: loaded from `app/agents/*.agent.json`. Request **must** specify `"agent"`.

---

## 3) High-level architecture

```
[ Browser (test.html + app.js) ]  --Socket.IO-->
    emit "Chat" { agent, text, memory {mode, thread_id}, ... }

[ FastAPI + Socket.IO ]
    - loads agent preset (system_prompt_path, params_override)
    - POOL.acquire() -> worker (LLama instance)
    - builds messages [system, user] (+ memory preamble later)
    - streams llama deltas -> emit "ChatChunk"
    - emits "ChatDone" / "Error" / "Interrupted"

[ WorkerPool ]
    - N model-bound workers (Llama instances)
    - semaphore-backed acquire/release
```

**Key idea:** workers are **model-agnostic agents**. On each request we “dress” the worker with the selected agent’s **system prompt** and **sampling overrides**. No model reload.

---

## 4) Configuration files

### 4.1 `agent_config.json` (model + runtime)

```json
{
	"runtime": {
		"pool_size": 1,
		"per_request_timeout_s": 0
	},
	"models": [
		{
			"name": "jan-nano-128k-Q4_K_M",
			"path": "./models/jan-nano-128k-Q4_K_M.gguf",
			"active": true,
			"mode": "chat",
			"system_prompt": "",            // optional at model level; agents override
			"params": {
				"n_ctx": 8192,
				"n_threads": 0,
				"n_gpu_layers": -1,
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

* Exactly **one** model must have `"active": true`.
* **No grammar keys** anywhere (feature dropped).
* `pool_size` and `per_request_timeout_s` (0 = disabled) control the worker pool and timeouts.

### 4.2 Agent presets (`app/agents/*.agent.json`)

Examples:

`router.agent.json`

```json
{
	"name": "router",
	"system_prompt_path": "../prompts/router_system_prompt.txt",
	"params_override": {
		"temperature": 0.2,
		"top_k": 20,
		"top_p": 0.9,
		"min_p": 0.1,
		"max_tokens": 256,
		"stop": ["<|im_end|>"]
	}
}
```

`topic.agent.json`

```json
{
	"name": "topic",
	"system_prompt_path": "../prompts/topic_system_prompt.txt",
	"params_override": {
		"temperature": 0.6,
		"top_k": 40,
		"top_p": 0.9,
		"min_p": 0.1,
		"max_tokens": 512,
		"stop": ["<|im_end|>"]
	}
}
```

> Paths are resolved **relative to the agent JSON file**. In our tree, `prompts` is a **sibling of** `agents`, so `../prompts/...` is correct.

---

## 5) Socket protocol (as implemented)

### Client → Server: `Chat`

```json
{
	"agent": "router" | "topic",
	"text": "user input",
	"memory": {
		"mode": "none" | "thread_window",   // "none" for now; thread_window is coming next
		"thread_id": "uuid-or-string"       // client auto-generates; always sent
	}
}
```

### Server → Client: events

* `RunStarted` — `{ runId }`
* `ChatChunk` — `{ runId, chunk: "<delta text>" }` (one per delta; **no buffering**)
* `ChatDone` — `{ runId }`
* `Interrupted` — `{ runId }`
* `Error` — `{ runId, message }`
* `Log` — `{ msg }` (lightweight diagnostics; shown in Events panel)

**Strictness:**

* Missing/unknown `agent` ⇒ `Error` and no fallbacks.
* Missing `thread_id` (in future memory modes) ⇒ server will generate, but logs a warning. (Today the client always sends one.)

---

## 6) Streaming behavior

* We forward **each llama delta immediately** over Socket.IO — no artificial chunking or word-splitting.
* Backpressure is handled via await on the emit path; we don’t accumulate on the server.

---

## 7) Worker pool

* Initialized at startup from `runtime.pool_size`.
* Each worker owns a **Llama instance**.
* Requests `acquire()` a worker, run the stream, and release it.
* This provides **concurrency** and **isolation** without model reloads.

---

## 8) Memory (design agreed; next feature)

**Goal:** memory is **agent-agnostic** and **strategy-based**.

* **Strategy interface** (planned):

  * `append(thread_id, role, content)`
  * `build_preamble(thread_id, max_context_tokens)` → string used before the user message
* **Initial strategy:** `thread_window` (in-memory)

  * Keeps a rolling window of turns per `thread_id`
  * Token budgeted to fit context (`max_context_tokens` per request or default)
* **Persistence (later):** `postgres` strategy (async SQLAlchemy)

  * Same interface; stores turns in Postgres
  * Survives restarts; client **reuses `thread_id`** after reconnect
* **Modes in config/request (planned):**

  * `"none"` — stateless (today’s behavior)
  * `"thread_window"` — rolling window by `thread_id`
  * `"postgres"` — persistent store (later)

**Session continuity:** Socket.IO `sid` is ephemeral; continuity is achieved by the client **always sending the same `thread_id`** when it wants continuity.

---

## 9) GPU notes (llama-cpp-python)

* Use a **prebuilt CUDA wheel** matching your CUDA runtime, e.g.:

  * `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125`
* Indicators you’re on GPU:

  * Startup logs show CUDA/cuBLAS and layers assigned off-CPU.
  * High prompt tok/s; `nvidia-smi` shows activity.
* Tunables that matter:

  * `n_gpu_layers: -1` (or limit to fit VRAM),
  * Prompt batch sizes (`n_batch`, `n_ubatch`) if exposed,
  * We **don’t** hardcode flash attention; let the build decide.

---

## 10) Testing & troubleshooting checklist

* **UI loads** at `/test.html` and includes `/app.js` and `/styles.css`.
* **Socket.I/O client** served from `./socket.io.min.js` in static root.
* **Events** are on a separate log panel; **chat messages** are clean (no mixed logs).
* Every run shows: `RunStarted → ChatChunk... → ChatDone` (or `Error/Interrupted`).
* **Unknown agent** produces a clear `Error` (no silent fallbacks).
* **CUDA in use** (check startup logs and `nvidia-smi` while generating).
* If you see per-word line breaks, ensure we’re not splitting on whitespace and that the client appends raw deltas.

---

## 11) Roadmap (incremental, actionable)

### ✅ Phase A — Solid baseline (done)

* Socket.IO server + static UI.
* True token streaming.
* Agent presets from `app/agents/*.agent.json`.
* Worker pool with `pool_size`.
* No grammar; strict agent resolution; health endpoint removed.

### ▶ Phase B — Memory foundation (next)

* Implement `MemoryStrategy` interface and **`thread_window`** strategy (in-proc).
* Server:

  * append user/assistant turns under `thread_id`,
  * build preamble within `max_context_tokens` (config default; request can override),
  * include light `Log` events (stored N tokens, trimmed M tokens).
* Client:

  * keep auto-generated `thread_id` and always send it.

### ▶ Phase C — Observability & polish

* Per-run log line with `{agent, memory.mode, thread_id}`.
* Simple counters in UI (e.g., “ctx \~2.1k / 4k tokens”).

### ▶ Phase D — Persistence (optional)

* Add `postgres` strategy (SQLAlchemy async).
* Config accepts `"memory": {"type":"postgres", "dsn":"...", "max_context_tokens":4096}`.
* With persistent store, client can **reconnect** (new Socket.IO `sid`), reuse the **same `thread_id`**, and the thread history is recovered.

### ▶ Phase E — Scaling tweaks

* Increase pool size (dependant on VRAM).
* Optional priority queue (e.g., “router” > “topic”).

---

## 12) Example request/response (today)

**Client → `Chat`:**

```json
{
	"agent": "topic",
	"text": "Give me a quick overview of New Zealand's economy.",
	"memory": { "mode": "none", "thread_id": "6f67d4e2-0a6e-4a2e-972b-29a4a6d5a9af" }
}
```

**Server → stream:**

* `RunStarted`
* many `ChatChunk` (plain text deltas)
* `ChatDone`

---

## 13) Known limitations

* No memory yet (stateless only) — coming next via `thread_window`.
* No persistence across restarts (until `postgres` strategy).
* One active model per service instance (by design, for simplicity).
* Prompts must exist on disk; **missing paths cause errors** (intentionally no fallsbacks).

---

## 14) Folder layout (reference)

```
project-root/
├─ app/
│  ├─ main.py
│  ├─ llm_engine.py
│  ├─ worker_pool.py
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

---

## 15) Quick start notes

* Start without reloads (in VSCode, **remove `--reload`** from the uvicorn launch config).
* Ensure `socket.io.min.js` is served from static root and referenced as `./socket.io.min.js`.
* Confirm agent presets resolve their prompt paths (`../prompts/...`).

---

If you want, I can draft the thin `MemoryStrategy` interface + a minimal `thread_window` implementation next, wired to the existing `Chat` path (kept behind the `"memory.mode":"thread_window"` flag).
