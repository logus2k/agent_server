# Agent Server

A local-first AI orchestration backend that coordinates LLM inference, voice services, and multi-agent routing — entirely on-device, with no cloud dependencies.

Built with **FastAPI** and **Socket.IO**, the Agent Server exposes two API surfaces — a real-time WebSocket interface for streaming chat and voice pipelines, and an OpenAI-compatible REST API for drop-in integration with existing tools — plus a **web admin dashboard** for live operations.

It runs as a **thin orchestrator**: LLM inference is forwarded over HTTP to a **llama.cpp `llama-server`** sidecar (`llama-vision`) that hosts the models. `data/agent_config.json` is the **single source of truth** for which models run; a small backend **adapter** translates it into the llama-server config at container boot, so swapping the inference backend (e.g. to vLLM) means writing one adapter, not touching agent_server.

> See [architecture.drawio](architecture.drawio) for a visual diagram, [documents/how_to.md](documents/how_to.md) for model-switching and agent-creation workflows, and [documents/active_model_switching_sdk.md](documents/active_model_switching_sdk.md) for the model-switching API contract.

---

## Architecture

```
Real-time Clients ─WebSocket─► ┌──────────────────────────────────────┐
(Browser, IoT)                 │        Agent Server (:7701)          │  thin orchestrator
                               │  Socket.IO   │   REST API (/v1)      │
OpenAI Clients ──HTTP/SSE────► │      ↓       │        ↓              │
(curl, SDKs, noted, CV)        │  Session  Router  Presets   Memory   │
                               │            Worker Pool               │
Admin Browser ──HTTP─────────► │   Admin API + Dashboard (/admin/)    │
                               │   STT Mgr ─► STT Server (:2700)      │
                               │   TTS Mgr ─► TTS Server (:7700)      │
                               └──────────────────┬───────────────────┘
                                                  │ HTTP forward (OpenAI proto)
                                                  ▼
                               ┌──────────────────────────────────────┐
                               │        llama-vision (:8500)          │  model host
                               │   llama.cpp llama-server (router)    │
                               │   active chat model + resident models │
                               │   + embedders + reranker (GGUF)      │
                               └──────────────────────────────────────┘
                                                  ▲
                     data/agent_config.json ──(llama.cpp adapter)──► models preset
                     (single source of truth)      (regenerated at boot)
```

### Modules

| Module | File | Purpose |
|:---|:---|:---|
| **Main** | `app/main.py` | FastAPI + Socket.IO orchestration, session management, event handlers, HTTP client-tracking middleware |
| **Forwarding Engine** | `app/llm_engine_server.py` | HTTP-forwards inference to the `llama-server` sidecar (the default engine) |
| **In-process Engine** | `app/llm_engine.py` | In-process `llama-cpp-python` engine (rollback path; built from `Dockerfile.fat`) |
| **Worker Pool** | `app/worker_pool.py` | Async queue of N engine instances for concurrent requests |
| **Memory** | `app/memory.py` | Pluggable memory strategies; ships with `ThreadWindowMemory` (rolling window per thread) |
| **Router Dispatch** | `app/router_dispatch.py` | Fire-and-forget intent classification using the `router` agent preset |
| **OpenAI Compat** | `app/openai_compat.py` | OpenAI-compatible REST layer (`/v1/chat/completions`, `/v1/models`) |
| **Admin API** | `app/admin_api.py` | Agent CRUD, config edit, model/context/vision-adapter switching, model registration, live status — `/admin/api/*`; web dashboard at `/admin/` |
| **Call Log** | `app/call_log.py` | In-memory ring buffer of recent LLM calls surfaced on the dashboard |
| **HTTP Log** | `app/http_log.py` | Per-IP record of every HTTP request, feeding the Clients view |
| **GeoIP** | `app/geoip.py` | MaxMind GeoLite2 lookup (country/city) for client IPs |
| **GGUF Meta** | `app/gguf_meta.py` | Dependency-free GGUF header reader (max context, file metadata) for the register-model flow |
| **STT Manager** | `app/stt_manager.py` | Multiplexed Socket.IO connections to external STT servers |
| **TTS Manager** | `app/tts_manager.py` | Streams text chunks to an external TTS server for voice synthesis |
| **llama.cpp Adapter** | `adapter/` | Generates the `llama-server` preset from `agent_config.json` at container boot |

---

## Features

- **Streaming chat** via Socket.IO with per-chunk delivery to the browser
- **OpenAI-compatible REST API** — works with `curl`, Python/JS OpenAI SDKs, LangChain, and any tool that speaks the OpenAI protocol
- **Multi-agent presets** — 40+ agents, each a JSON config with its own system prompt, sampling parameters, and memory policy (see `data/agents/`)
- **Conversation memory** — `ThreadWindowMemory` keeps a rolling context window per `thread_id`, injected as a preamble to the LLM
- **Router agent** — classifies user intent in parallel (fire-and-forget) and emits structured JSON to the client
- **Voice pipeline** — integrates with separate STT and TTS servers over Socket.IO for end-to-end voice interaction
- **Worker pool** — bounds concurrent LLM usage; additional requests queue until a worker is available
- **Cancellation** — clients can interrupt an active generation at any time
- **GPU acceleration** — NVIDIA GPU offloading via llama.cpp's `n-gpu-layers`
- **Speculative decoding** — backend options pass through verbatim, so draft-model / MTP speculative decoding (`spec-type`, `spec-draft-model`) can be enabled per model for higher throughput
- **Hot model switching** — switch the active chat model (or embedder/reranker, or vision adapter) from the dashboard or one API call; the orchestrator regenerates the preset and restarts the stack via the Docker socket — no manual edits
- **Multi-resident models** — mark models `resident` to keep more than one loaded at once (bounded by `--models-max`) and address any of them by `model_id` in the same API
- **Server-side context control** — change the active model's context window from the dashboard
- **Per-request reasoning + structured output** — toggle thinking per request (`chat_template_kwargs`) and request JSON-schema-constrained output (`response_format`)
- **Admin dashboard** — live GPU/VRAM, recent LLM calls, request logs, agent CRUD, model/vision management, and a GeoIP-aware Clients view at `/admin/`
- **Backend-agnostic config** — the neutral config core is translated to llama.cpp by an adapter; a different backend (e.g. vLLM) is one adapter away
- **JavaScript SDK** — `agentClient.js` ES module for browser integration

---

## Quick Start

### Prerequisites

- Docker with the NVIDIA Container Toolkit (the default deployment is two GPU-backed containers)
- GGUF model files in `data/models/`
- An external Docker network named `logus2k_network` (and `noted-network` if integrating with noted)

### Docker (default deployment)

The live deployment uses the **adapter** stack: the llama-server preset is generated from `data/agent_config.json` at boot, so there is no host `.ini` file to keep in sync.

```bash
docker compose -f docker-compose.adapter.yml --profile default up -d --build
```

This brings up two services:

- **`llama-vision`** (`:8500`) — the model host: llama.cpp's `llama-server` in router mode (`--models-max 4`), hosting the active chat model, any `resident` models, the embedder, and the reranker. Owns the GPU. Built from `Dockerfile.llama-adapter` (the pinned llama.cpp image + the adapter entrypoint).
- **`agent_server`** (`:7701`) — the thin orchestrator (REST + Socket.IO + presets + memory + STT/TTS + admin) that forwards LLM calls to `llama-vision`. The compose mounts the Docker socket so the admin API can restart the stack to apply a model switch.

`docker-compose.adapter.yml` mounts `data/` **read-write** (the admin API writes presets and config back to it) and configures NVIDIA GPU passthrough for `llama-vision`. To switch the resident model, use the admin dashboard / API (below) or edit `data/agent_config.json` and restart — see [documents/how_to.md](documents/how_to.md).

> **Static-ini alternative.** `docker-compose.yml` runs the stock llama.cpp image directly against a hand-maintained `llama-router-models.ini` instead of generating the preset from `agent_config.json`. It predates the adapter cutover; prefer the adapter compose unless you specifically need the static ini.

### In-process mode (rollback)

For a single-container deployment that loads the model in-process via `llama-cpp-python` (no `llama-server` sidecar), build from `Dockerfile.fat` and leave `LLAMA_SERVER_URL` unset. This is the legacy/rollback path; forwarding mode is the default.

### Published images (Docker Hub)

Prebuilt images are published under `logus2k` (weights are **not** baked in — you download GGUFs yourself):

| Image | Role |
|:---|:---|
| [`logus2k/agent-server`](https://hub.docker.com/r/logus2k/agent-server) | OpenAI-compatible API + admin dashboard (`:7701`) |
| [`logus2k/agent-server-llama-adapter`](https://hub.docker.com/r/logus2k/agent-server-llama-adapter) | `llama-vision` — the llama.cpp host that loads & serves the GGUFs (`:8500`) |

The `dockerhub/` folder is a self-contained install bundle (compose file, `download_models.sh` with GGUF URLs, a starter `agent_config.json`, and a README) for deploying from the published images.

---

## Configuration

### `agent_config.json`

The **single source of truth**, loaded at startup (override path via `AGENT_CONFIG`). `models` is grouped by task — `chat`, `embedding`, `reranking` — and every entry is **self-describing**: a neutral, backend-agnostic core plus a per-backend block. Exactly one `chat` entry is `active` (the default model agent_server forwards to); additional entries can be `resident` to stay loaded alongside it.

```jsonc
{
  "runtime": { "pool_size": 20, "per_request_timeout_s": 0 },
  "memory": { "strategies": { "thread_window": { "max_context_tokens": 65536 } } },
  "models": {
    "chat": [
      {
        "active": true,
        "name": "Gemma 4 E4B IT Q4 KXL GGUF",
        "model_id": "gemma-4",            // forward id + generated [section] header (can't drift)
        "family": "gemma",                // gates model-specific response handling
        "active_backend": "llama_cpp",
        "context": 65536,
        "reasoning": true,
        "vision": true,
        "download_url": "https://huggingface.co/...",   // shown in the admin UI; used by download_models.sh
        "sampling": { "temperature": 1, "top_k": 64, "top_p": 0.95, "min_p": 0, "max_tokens": 65536 },
        "backends": {
          "llama_cpp": {
            "model_file": "/agent_server_models/gemma-4-E4B-it-UD-Q4_K_XL.gguf",
            "projector": "/agent_server_models/mmproj-F16.gguf",
            "options": { "n-gpu-layers": -1, "flash-attn": "on", "jinja": true,
                         "chat-template-file": "/agent_server_models/chat_template_gemma-4.jinja" }
          }
        }
      },
      {
        "active": false,
        "resident": true,                 // stays loaded alongside the active model; callable by model_id
        "model_id": "ma2-360m-dpo-b01",
        "family": "llama",
        // ...
      }
      // ...other inactive chat models (qwen3.5, qwen3.5-9b, smollm3, granite-3.3, nemotron, ministral)
    ],
    "embedding": [ /* bge-m3   — backends.llama_cpp.options */ ],
    "reranking": [ /* bge-reranker */ ]
  }
}
```

- **Neutral core** (read by agent_server *and* any adapter): `model_id`, `family`, `context`, `reasoning`, `vision`, `sampling`, `download_url`.
- **`active` / `resident`**: exactly one chat model is `active` (the default route); any model may be `resident` to stay loaded (bounded by `--models-max` in the compose command).
- **`backends.<name>.options`** holds raw backend flags passed through **verbatim** to llama-server — so advanced flags (e.g. `flash-attn`, `spec-type`/`spec-draft-model` for speculative decoding, `ctx-checkpoints`) work without adapter changes.
- **Server-process flags** (`--host`, `--port`, `--models-max`, `--cache-reuse`) live in the compose command, not here.

**To switch the active model:** use the dashboard or `POST /admin/api/active-model` (below) — it flips the `active` flags, regenerates the preset, and restarts `llama-vision` + `agent_server` to apply (~40 s). Or edit `active` by hand and restart. Full workflow + per-model notes: [documents/how_to.md](documents/how_to.md), [documents/llama_server_model_notes.md](documents/llama_server_model_notes.md).

### Agent Presets

Agents are defined as JSON files in `data/agents/` (`*.agent.json`). Each file configures an agent's behavior:

```json
{
  "name": "general",
  "system_prompt": "/agent_server/app/data/prompts/general_assistance_prompt.txt",
  "params_override": {
    "temperature": 0.6,
    "max_tokens": 2048,
    "chat_template_kwargs": { "enable_thinking": false }
  },
  "memory_policy": "thread_window"
}
```

| Field | Description |
|:---|:---|
| `name` | Unique agent identifier (used in API calls) |
| `system_prompt` | Path to the system prompt text file |
| `params_override` | Sampling parameters that override model defaults; may include `chat_template_kwargs` (per-request reasoning toggle) and `response_format` (JSON-schema-constrained output) |
| `memory_policy` | `"none"` or `"thread_window"` |
| `tts_field` | (Optional) Extract a specific JSON field from the response for TTS |

**Included agents:** 40+ presets in `data/agents/` (`general`, `router`, `cv_assistant`, `cv_query_rewriter`, `diana`, `researcher`, `planner`, `aigeo`, the `job2cool_*`, `jobunter_*`, `noted_*`, and `atlm_*` families, …). All run on the **active** chat model unless their request targets a specific `resident` model. See [documents/how_to.md](documents/how_to.md) for creating your own (file-based or via the admin API).

---

## API Reference

### Socket.IO Events

Connect to `ws://localhost:7701/socket.io`.

**Client → Server:**

| Event | Payload | Description |
|:---|:---|:---|
| `Chat` | `{ agent, text, thread_id? }` | Start a chat run with the named agent |
| `Interrupt` | — | Cancel the active generation |
| `JoinSTT` | `{ clientId, agent, threadId? }` | Subscribe to STT transcripts |
| `LeaveSTT` | `{ clientId }` | Unsubscribe from STT |
| `JoinTTS` | `{ clientId, voice?, speed? }` | Subscribe to TTS output |
| `LeaveTTS` | `{ clientId }` | Unsubscribe from TTS |

**Server → Client:**

| Event | Payload | Description |
|:---|:---|:---|
| `RunStarted` | `{ runId }` | A new generation run has begun |
| `ChatChunk` | `{ runId, chunk }` | A streamed token/chunk of the response |
| `ChatDone` | `{ runId }` | Generation completed |
| `Interrupted` | `{ runId }` | Generation was cancelled |
| `Error` | `{ code, message }` | An error occurred |
| `RouterResult` | `{ Operation, ... }` | Router agent classification result |
| `UserTranscript` | `{ clientId, text, ... }` | STT transcription result |
| `STTSubscribed` / `STTUnsubscribed` | `{ clientId }` | STT subscription state |
| `TTSSubscribed` / `TTSUnsubscribed` | `{ clientId }` | TTS subscription state |

### OpenAI REST API

**POST /v1/chat/completions**

```bash
curl http://localhost:7701/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

The `model` field accepts either:
- an **agent preset name** (e.g. `"general"`, `"cv_query_rewriter"`) — runs that preset on the active chat model, **or**
- a **`model_id`** (e.g. `"gemma-4"`, or a `resident` model like `"ma2-360m-dpo-b01"`) — routes directly to that model.

**GET /v1/models** — lists the active model, all configured chat/embedding/reranking models, and every agent preset (as virtual models). Each entry carries `kind` (`chat` / `embedding` / `reranking` / `agent`), and chat models additionally expose `active`, `resident`, `context`, `max_context` (from the GGUF header), `size_bytes`, and `download_url`.

**GET /v1/agents/{name}** — returns a single agent preset's resolved configuration.

Optional auth: set the `OPENAI_API_KEY` environment variable and pass `Authorization: Bearer <key>`.

### Admin API

Operational endpoints under `/admin/api/*`, also driving the dashboard at `/admin/`:

| Endpoint | Purpose |
|:---|:---|
| `GET/POST/PUT/DELETE /agents[...]` | CRUD on agent presets |
| `GET/PUT /config` | View / edit `agent_config.json` |
| `POST /active-model` | Switch the active model in a category (`{ model_id, category }`); restarts the stack to apply |
| `POST /active-context` | Change the active model's context window |
| `GET /vision-adapters`, `POST /vision-adapter` | List / select the vision (mmproj) projector |
| `GET /discovered`, `POST /register` | Discover unregistered GGUFs and register one as a new (inactive) model |
| `GET /status` | Live GPU/VRAM + resident models + router state |
| `GET /calls`, `GET /logs` | Recent LLM calls and server logs |
| `GET /memory`, `GET /memory/{thread_id}` | Inspect conversation memory threads |
| `GET /clients`, `POST /clients/clear` | GeoIP-aware caller list (honors `X-Forwarded-For`); clear one or all |

The dashboard tabs are **Dashboard** (status + recent calls), **Agents** (preset editor + live test), **Clients** (GeoIP table with country flags), and **Configuration** (active model / embeddings / reranking / vision adapter + register model).

### JavaScript SDK

```html
<script src="/socket.io.min.js"></script>
<script type="module">
  import { AgentClient } from "/sdk/agentClient.js";

  const client = new AgentClient({ url: "http://localhost:7701" });
  await client.connect();

  const result = await client.runText("What is machine learning?", {
    agent: "general",
    threadId: "my-thread"
  }, {
    onChunk: (piece) => console.log(piece),
    onDone:  ()      => console.log("Done!")
  });
</script>
```

SDK docs and runnable samples live under `app/static/sdk/` (see `documentation/`).

---

## Project Structure

```
agent_server/
├── app/
│   ├── main.py                # FastAPI + Socket.IO orchestration + HTTP middleware
│   ├── llm_engine_server.py   # HTTP-forwarding engine (default)
│   ├── llm_engine.py          # in-process llama-cpp-python engine (rollback)
│   ├── worker_pool.py         # async engine pool
│   ├── memory.py              # memory strategies (ThreadWindowMemory)
│   ├── router_dispatch.py     # intent classification dispatcher
│   ├── openai_compat.py       # OpenAI-compatible REST endpoints
│   ├── admin_api.py           # admin API (/admin/api/*) — agents, config, model switching
│   ├── call_log.py            # recent-LLM-calls ring buffer
│   ├── http_log.py            # per-IP HTTP request log
│   ├── geoip.py               # MaxMind GeoLite2 client lookup
│   ├── gguf_meta.py           # GGUF header reader (register-model flow)
│   ├── stt_manager.py         # multiplexed STT connections
│   ├── tts_manager.py         # TTS streaming manager
│   └── static/
│       ├── admin/             # admin dashboard (served at /admin/) + country flags
│       ├── test.html, openai_test.html
│       └── sdk/agentClient.js # client SDK (ES module) + docs/samples
├── adapter/                   # llama.cpp adapter: preset generator + entrypoint
├── data/
│   ├── agent_config.json      # single source of truth (models + runtime + memory)
│   ├── agents/                # agent preset configs (*.agent.json)
│   ├── prompts/               # system prompt text files
│   ├── models/                # GGUF model files (not in git)
│   └── geoip/                 # GeoLite2 MMDB databases (not in git)
├── dockerhub/                 # install bundle for the published Docker Hub images
├── docker-compose.adapter.yml # default deployment (adapter generates the preset)
├── docker-compose.yml         # static-ini alternative (stock llama.cpp + .ini)
├── Dockerfile, Dockerfile.fat, Dockerfile.llama-adapter
├── documents/                 # how_to.md, llama_server_model_notes.md, active_model_switching_sdk.md, plans/
├── architecture.drawio        # architecture diagram (draw.io)
└── README.md
```

---

## Related Services

| Service | Port | Purpose | Repository |
|:---|:---|:---|:---|
| **STT Server** | 2700 | Speech-to-Text transcription | [logus2k/stt_server](https://github.com/logus2k/stt_server) |
| **TTS Server** | 7700 | Text-to-Speech synthesis | [logus2k/tts_server](https://github.com/logus2k/tts_server) |

Consumers in the same stack (e.g. **noted** and the **CV** assistant) call agent_server over the OpenAI-compatible API. All services communicate over a shared Docker network (`logus2k_network`) using Socket.IO / HTTP.

---

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
</content>
</invoke>
