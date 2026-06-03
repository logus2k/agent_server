# Agent Server

A local-first AI orchestration backend that coordinates LLM inference, voice services, and multi-agent routing — entirely on-device, with no cloud dependencies.

Built with **FastAPI** and **Socket.IO**, the Agent Server exposes two API surfaces — a real-time WebSocket interface for streaming chat and voice pipelines, and an OpenAI-compatible REST API for drop-in integration with existing tools.

It runs as a **thin orchestrator**: LLM inference is forwarded over HTTP to a **llama.cpp `llama-server`** sidecar (`llama-vision`) that hosts the models. `data/agent_config.json` is the **single source of truth** for which model runs; a small backend **adapter** translates it into the llama-server config, so swapping the inference backend (e.g. to vLLM) means writing one adapter, not touching agent_server.

> See [architecture.drawio](architecture.drawio) for a visual diagram, and [documents/how_to.md](documents/how_to.md) for the model-switching and agent-creation workflows.

---

## Architecture

```
Real-time Clients ─WebSocket─► ┌──────────────────────────────────────┐
(Browser, IoT)                 │        Agent Server (:7701)          │  thin orchestrator
                               │  Socket.IO   │   REST API (/v1)      │
OpenAI Clients ──HTTP/SSE────► │      ↓       │        ↓              │
(curl, SDKs, noted)            │  Session  Router  Presets   Memory   │
                               │            Worker Pool               │
                               │   STT Mgr ─► STT Server (:2700)      │
                               │   TTS Mgr ─► TTS Server (:7700)      │
                               └──────────────────┬───────────────────┘
                                                  │ HTTP forward (OpenAI proto)
                                                  ▼
                               ┌──────────────────────────────────────┐
                               │        llama-vision (:8500)          │  model host
                               │   llama.cpp llama-server (router)    │
                               │   chat model + embedders (GGUF)      │
                               └──────────────────────────────────────┘
                                                  ▲
                     data/agent_config.json ──(llama.cpp adapter)──► models preset
                     (single source of truth)
```

### Modules

| Module | File | Purpose |
|:---|:---|:---|
| **Main** | `app/main.py` | FastAPI + Socket.IO orchestration, session management, event handlers |
| **Forwarding Engine** | `app/llm_engine_server.py` | HTTP-forwards inference to the `llama-server` sidecar (the default engine) |
| **In-process Engine** | `app/llm_engine.py` | In-process `llama-cpp-python` engine (rollback path; built from `Dockerfile.fat`) |
| **Worker Pool** | `app/worker_pool.py` | Async queue of N engine instances for concurrent requests |
| **Memory** | `app/memory.py` | Pluggable memory strategies; ships with `ThreadWindowMemory` (rolling window per thread) |
| **Router Dispatch** | `app/router_dispatch.py` | Fire-and-forget intent classification using the `router` agent preset |
| **OpenAI Compat** | `app/openai_compat.py` | OpenAI-compatible REST layer (`/v1/chat/completions`, `/v1/models`) |
| **Admin API** | `app/admin_api.py` | CRUD for agent presets + config view/edit (`/admin/api/*`); web UI at `/admin/` |
| **llama.cpp Adapter** | `adapter/` | Generates the `llama-server` preset from `agent_config.json` at container boot |
| **STT Manager** | `app/stt_manager.py` | Multiplexed Socket.IO connections to external STT servers |
| **TTS Manager** | `app/tts_manager.py` | Streams text chunks to an external TTS server for voice synthesis |

---

## Features

- **Streaming chat** via Socket.IO with per-chunk delivery to the browser
- **OpenAI-compatible REST API** — works with `curl`, Python/JS OpenAI SDKs, LangChain, and any tool that speaks the OpenAI protocol
- **Multi-agent presets** — each agent is a JSON config with its own system prompt, sampling parameters, and memory policy
- **Conversation memory** — `ThreadWindowMemory` keeps a rolling context window per `thread_id`, injected as a preamble to the LLM
- **Router agent** — classifies user intent in parallel (fire-and-forget) and emits structured JSON to the client
- **Voice pipeline** — integrates with separate STT and TTS servers over Socket.IO for end-to-end voice interaction
- **Worker pool** — bounds concurrent LLM usage; additional requests queue until a worker is available
- **Cancellation** — clients can interrupt an active generation at any time
- **GPU acceleration** — supports NVIDIA GPU offloading via llama.cpp's `n-gpu-layers`
- **Switchable models** — one config file (`agent_config.json`) defines every model; flip `active` + restart to switch the resident chat model (one at a time)
- **Backend-agnostic config** — the neutral config core is translated to llama.cpp by an adapter; a different backend (e.g. vLLM) is one adapter away
- **Admin UI** — manage agent presets and view/edit the service config at `/admin/`
- **JavaScript SDK** — `agentClient.js` ES module for easy browser integration

---

## Quick Start

### Prerequisites

- Docker with the NVIDIA Container Toolkit (the default deployment is two GPU-backed containers)
- GGUF model files in `data/models/`

### Docker (default — forwarding mode)

```bash
docker compose --profile default up -d
```

This brings up two services:

- **`llama-vision`** (`:8500`) — the model host: llama.cpp's `llama-server` in router mode, hosting the active chat model + embedders. Owns the GPU.
- **`agent_server`** (`:7701`) — the thin orchestrator (REST + Socket.IO + presets + memory + STT/TTS) that forwards LLM calls to `llama-vision`.

`docker-compose.yml` mounts `data/` **read-write** (the admin API writes presets and config back to it) and configures NVIDIA GPU passthrough for `llama-vision`. To switch the resident model, edit `data/agent_config.json` and restart both services — see [documents/how_to.md](documents/how_to.md).

> An **adapter cutover** compose (`docker-compose.adapter.yml` + `Dockerfile.llama-adapter`) makes the llama-server preset fully generated from `agent_config.json` at container boot — no host `.ini` to manage.

### In-process mode (rollback)

For a single-container deployment that loads the model in-process via `llama-cpp-python` (no `llama-server` sidecar), build from `Dockerfile.fat` and leave `LLAMA_SERVER_URL` unset. This is the legacy/rollback path; forwarding mode is the default.

---

## Configuration

### `agent_config.json`

The **single source of truth**, loaded at startup (override path via `AGENT_CONFIG`). `models` is grouped by task — `chat`, `embedding`, `reranking` — and every entry is **self-describing**: a neutral, backend-agnostic core plus a per-backend block. Exactly one `chat` entry is `active` (the resident model agent_server forwards to; VRAM allows one at a time).

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
        "context": 131072,
        "reasoning": true,
        "vision": true,
        "sampling": { "temperature": 1, "top_k": 64, "top_p": 0.95, "min_p": 0, "max_tokens": 131072 },
        "backends": {
          "llama_cpp": {
            "model_file": "/agent_server_models/gemma-4-E4B-it-UD-Q4_K_XL.gguf",
            "projector": "/agent_server_models/mmproj-F16.gguf",
            "options": { "n-gpu-layers": -1, "flash-attn": "on", "jinja": true,
                         "chat-template-file": "/agent_server_models/chat_template_gemma-4.jinja",
                         "ctx-checkpoints": 0 }
          }
        }
      }
      // ...other inactive chat models (qwen3.5, phi-4-mini-reasoning, ...)
    ],
    "embedding": [ /* bge-m3   — backends.llama_cpp.options */ ],
    "reranking": [ /* bge-reranker */ ]
  }
}
```

- **Neutral core** (read by agent_server *and* any adapter): `model_id`, `family`, `context`, `reasoning`, `vision`, `sampling`.
- **`backends.<name>.options`** holds raw backend flags (no shared-defaults section — each model self-describes).
- **Server-process flags** (`--host`, `--port`, `--models-max`, `--cache-reuse`) live in the compose command, not here.

**To switch the model:** flip `active` under `models.chat`, then restart `llama-vision` + `agent_server`. The llama.cpp adapter regenerates the llama-server preset from this file at boot. Full workflow + per-model notes: [documents/how_to.md](documents/how_to.md), [documents/llama_server_model_notes.md](documents/llama_server_model_notes.md).

### Agent Presets

Agents are defined as JSON files in `data/agents/`. Each file configures an agent's behavior:

```json
{
  "name": "general",
  "system_prompt": "/agent_server/app/data/prompts/general_assistance_prompt.txt",
  "params_override": {
    "temperature": 0.6,
    "max_tokens": 2048
  },
  "memory_policy": "thread_window"
}
```

| Field | Description |
|:---|:---|
| `name` | Unique agent identifier (used in API calls) |
| `system_prompt` | Path to the system prompt text file |
| `params_override` | Sampling parameters that override model defaults |
| `memory_policy` | `"none"` or `"thread_window"` |
| `tts_field` | (Optional) Extract a specific JSON field from the response for TTS |

**Included agents:** 30+ presets in `data/agents/` (`general`, `router`, `cv_assistant`, `diana`, `researcher`, `planner`, the `jobunter_*` and `noted_*` families, …). See [documents/how_to.md](documents/how_to.md) for creating your own (file-based or via the admin API).

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

The `model` field accepts any agent preset name (e.g., `"general"`, `"ml"`) or the active model's `model_id` (e.g. `"gemma-4"`).

**GET /v1/models** — lists the active model and all agent presets as virtual models.

Optional auth: set the `OPENAI_API_KEY` environment variable and pass `Authorization: Bearer <key>`.

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

---

## Project Structure

```
agent_server/
├── app/
│   ├── main.py                # FastAPI + Socket.IO orchestration
│   ├── llm_engine_server.py   # HTTP-forwarding engine (default)
│   ├── llm_engine.py          # in-process llama-cpp-python engine (rollback)
│   ├── worker_pool.py         # async engine pool
│   ├── memory.py              # memory strategies (ThreadWindowMemory)
│   ├── router_dispatch.py     # intent classification dispatcher
│   ├── openai_compat.py       # OpenAI-compatible REST endpoints
│   ├── admin_api.py           # admin CRUD API (/admin/api/*)
│   ├── stt_manager.py         # multiplexed STT connections
│   ├── tts_manager.py         # TTS streaming manager
│   └── static/
│       ├── admin/             # admin web UI (served at /admin/)
│       ├── test.html, openai_test.html
│       └── sdk/agentClient.js # client SDK (ES module)
├── adapter/                   # llama.cpp adapter: preset generator + entrypoint
├── data/
│   ├── agent_config.json      # single source of truth (models + runtime + memory)
│   ├── agents/                # agent preset configs (*.agent.json)
│   ├── models/                # GGUF model files
│   └── prompts/               # system prompt text files
├── docker-compose.yml         # default (forwarding) deployment
├── docker-compose.adapter.yml # adapter-cutover deployment
├── Dockerfile, Dockerfile.fat, Dockerfile.llama-adapter
├── documents/                 # how_to.md, llama_server_model_notes.md, plans/
├── architecture.drawio        # architecture diagram (draw.io)
└── README.md
```

---

## Related Services

| Service | Port | Purpose | Repository |
|:---|:---|:---|:---|
| **STT Server** | 2700 | Speech-to-Text transcription | [logus2k/stt_server](https://github.com/logus2k/stt_server) |
| **TTS Server** | 7700 | Text-to-Speech synthesis | [logus2k/tts_server](https://github.com/logus2k/tts_server) |

All services communicate over a shared Docker network (`logus2k_network`) using Socket.IO.

---

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
