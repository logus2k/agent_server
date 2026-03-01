# Agent Server

A local-first AI orchestration backend that coordinates LLM inference, voice services, and multi-agent routing — entirely on-device, with no cloud dependencies.

Built with **FastAPI**, **Socket.IO**, and **llama.cpp**, the Agent Server exposes two API surfaces — a real-time WebSocket interface for streaming chat and voice pipelines, and an OpenAI-compatible REST API for drop-in integration with existing tools.

> See [architecture.drawio](architecture.drawio) for a visual diagram of the system.

---

## Architecture

```
Real-time Clients ──WebSocket──► ┌─────────────────────────────────────┐
(Browser, IoT)                   │         Agent Server (:7701)        │
                                 │                                     │
OpenAI Clients ────HTTP/SSE────► │  Socket.IO    │   REST API (/v1)   │
(curl, SDKs)                     │       ↓       │        ↓           │
                                 │  Session  Router  Agent Presets     │
                                 │       ↓       ↓        ↓           │
                                 │     Worker Pool    Memory Registry  │
                                 │           ↓              ↓          │
                                 │        LLM Engine (llama.cpp)       │
                                 │                                     │
                                 │   STT Manager ─► STT Server (:2700)│
                                 │   TTS Manager ─► TTS Server (:7700)│
                                 └─────────────────────────────────────┘
                                          ↓               ↓
                                    GGUF Models    Agent Configs
```

### Modules

| Module | File | Purpose |
|:---|:---|:---|
| **Main** | `app/main.py` | FastAPI + Socket.IO orchestration, session management, event handlers |
| **LLM Engine** | `app/llm_engine.py` | llama.cpp wrapper with streaming inference and chat template support |
| **Worker Pool** | `app/worker_pool.py` | Async queue of N engine instances for concurrent requests |
| **Memory** | `app/memory.py` | Pluggable memory strategies; ships with `ThreadWindowMemory` (rolling window per thread) |
| **Router Dispatch** | `app/router_dispatch.py` | Fire-and-forget intent classification using the `router` agent preset |
| **OpenAI Compat** | `app/openai_compat.py` | OpenAI-compatible REST layer (`/v1/chat/completions`, `/v1/models`) |
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
- **GPU acceleration** — supports NVIDIA GPU offloading via llama.cpp's `n_gpu_layers`
- **JavaScript SDK** — `agentClient.js` ES module for easy browser integration

---

## Quick Start

### Prerequisites

- Python 3.10+
- A GGUF model file (place in `data/models/`)
- (Optional) Docker with NVIDIA Container Toolkit for GPU support

### Local Setup

```bash
# Install dependencies
pip install fastapi uvicorn python-socketio llama-cpp-python pydantic

# Edit agent_config.json to point to your model
# Ensure exactly one model has "active": true

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 7701
```

### Docker

```bash
docker compose up -d
```

The `docker-compose.yml` exposes port **7701** and mounts `data/` as a read-only volume. GPU passthrough is configured for NVIDIA devices.

---

## Configuration

### `agent_config.json`

The main configuration file. Loaded at startup (override path via `AGENT_CONFIG` env var).

```jsonc
{
  "runtime": {
    "pool_size": 1,              // Number of parallel LLM engine instances
    "per_request_timeout_s": 0   // Generation timeout (0 = unlimited)
  },
  "memory": {
    "strategies": {
      "thread_window": {
        "max_context_tokens": 8192
      }
    }
  },
  "models": [
    {
      "active": true,
      "name": "Qwen 2.5 7B Instruct Q8",
      "path": "/agent_server/app/data/models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
      "system_prompt": "",
      "params": {
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.005,
        "max_tokens": 2048
      }
    }
  ]
}
```

Exactly one model must have `"active": true`. Multiple models can be listed for easy switching.

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

**Included agents:** `general`, `router`, `topic`, `ml`, `robot`, `docbro`, `floorplan`, `succint`

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

The `model` field accepts any agent preset name (e.g., `"general"`, `"ml"`) or the active model slug.

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
│   ├── main.py              # FastAPI + Socket.IO orchestration
│   ├── llm_engine.py        # llama.cpp streaming wrapper
│   ├── worker_pool.py       # Async engine pool
│   ├── memory.py            # Memory strategies (ThreadWindowMemory)
│   ├── router_dispatch.py   # Intent classification dispatcher
│   ├── openai_compat.py     # OpenAI-compatible REST endpoints
│   ├── stt_manager.py       # Multiplexed STT connections
│   ├── tts_manager.py       # TTS streaming manager
│   └── static/
│       ├── test.html         # Built-in chat test UI
│       ├── openai_test.html  # OpenAI API test page
│       └── sdk/
│           └── agentClient.js  # Client SDK (ES module)
├── data/
│   ├── agents/              # Agent preset configs (*.agent.json)
│   ├── models/              # GGUF model files
│   └── prompts/             # System prompt text files
├── agent_config.json        # Main configuration
├── docker-compose.yml       # Docker deployment
├── architecture.drawio      # Architecture diagram (draw.io)
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
