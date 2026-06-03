Here’s a compact, up-to-date reference for your **config files** as the code stands now.

# Agent preset files (`app/agents/*.agent.json`)

Each file declares **one agent type** the server can run on a per-request basis. Files are loaded at startup; missing/invalid fields raise errors (no silent fallbacks). Paths are resolved **relative to the agent JSON file** (so with `app/agents` and `prompts/` as siblings, use `../prompts/...`).  &#x20;

**Schema (strict):**

```json
{
  "name": "router | topic | <custom>",
  "system_prompt": "../prompts/<file>.txt",
  "params_override": { /* optional generation overrides */ },
  "memory_policy": "none | thread_window"
}
```

* **name** (string, required): unique agent id, lower-cased by loader. Errors if missing or duplicated.&#x20;
* **system\_prompt** (string, required): file path to the system prompt. Only this key is accepted; `system_prompt_path` is rejected to avoid ambiguity. The path is resolved relative to the preset file.&#x20;
* **params\_override** (object, optional): per-agent **generation** overrides merged at request time. Typical keys: `max_tokens`, `temperature`, `top_k`, `top_p`, `min_p`, `stop` (array). These are passed to the engine for sampling. See examples below.&#x20;
* **memory\_policy** (string, required): server-controlled memory mode. Supported:

  * `"none"` — stateless.
  * `"thread_window"` — rolling window per `thread_id` (in-proc). If set, the **client must send `thread_id`**; the server enforces this and errors when missing.&#x20;

**Examples (yours):**

* `topic.agent.json` (uses memory)&#x20;
* `router.agent.json` (stateless; similar shape, with its own prompt)

**Gotchas & validation:**

* Any `grammar_path` key is **rejected** (grammar feature removed).&#x20;
* If the prompt path doesn’t exist, load fails at startup (by design).&#x20;

---

# Service config (`agent_config.json`)

Loaded at startup; exactly **one** model must be `"active": true`.&#x20;

```json
{
  "runtime": {
    "pool_size": 2,
    "per_request_timeout_s": 0
  },
  "memory": {
    "strategies": {
      "thread_window": {
        "max_context_tokens": 1024
      }
    }
  },
  "models": {
    "chat": [
      {
        "active": true,
        "name": "Gemma 4 E4B IT Q4 KXL GGUF",
        "model_id": "gemma-4",
        "family": "gemma",
        "active_backend": "llama_cpp",
        "context": 131072,
        "reasoning": true,
        "vision": true,
        "system_prompt": "",
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
    ],
    "embedding": [ { "active": true, "model_id": "bge-m3", "active_backend": "llama_cpp", "context": 8192,
                     "backends": { "llama_cpp": { "model_file": "/noted_models/.../bge-m3-Q8_0.gguf",
                       "options": { "n-gpu-layers": -1, "embedding": true, "pooling": "cls",
                                    "batch-size": 8192, "ubatch-size": 8192 } } } } ],
    "reranking": [ { "active": true, "model_id": "bge-reranker", "active_backend": "llama_cpp", "context": 8192,
                     "backends": { "llama_cpp": { "model_file": "/noted_models/.../bge-reranker-v2-m3-Q8_0.gguf",
                       "options": { "n-gpu-layers": -1, "embedding": true, "pooling": "rank",
                                    "batch-size": 8192, "ubatch-size": 8192 } } } } ]
  }
}
```



## `runtime`

* **pool\_size** (int): number of **model workers** created at startup.&#x20;
* **per\_request\_timeout\_s** (int, 0 disables): hard timeout for a `Chat` run; server cancels and emits an error on expiry.&#x20;

## `memory`

Defines available **memory strategies** the server can use when an agent’s `memory_policy` requests one.

* **strategies.thread\_window\.max\_context\_tokens** (int): token budget used to build the preamble from prior turns (rough \~4 chars/token heuristic). If omitted, defaults to 1024. &#x20;

> The registry is constructed from this section at startup; currently only `thread_window` is implemented.&#x20;

## `models{}`

`models` is an **object grouped by task** — `chat`, `embedding`,
`reranking`. Exactly one entry in `chat` must be `"active": true` (the
chat model agent_server forwards to; VRAM allows only one resident).
Embedding/reranking entries are hosted alongside for noted-rag.

Each entry is **self-describing** (no shared defaults section):

* **Neutral core** (backend-agnostic — read by agent_server *and* any backend adapter):
  * **model\_id** (string): the forward/routing id; also the generated llama-server `[section]` header (so they can't drift).
  * **name** (string): human display label (used for `display_name` in `/v1/models`).
  * **family** (string): `gemma` \| `qwen` \| `phi` — gates model-specific response handling.
  * **active\_backend** (string): which `backends.<name>` block is live (today only `llama_cpp`).
  * **context** (int), **reasoning** (bool), **vision** (bool).
  * **sampling** (object): `temperature`, `top_k`, `top_p`, `min_p`, `max_tokens` (and optionally `stop`). Agent `params_override` overrides these per request.
  * **system\_prompt** (string): optional model-level default; agents override per request.
* **backends.<name>** (object): backend specifics — `model_file`, `projector` (vision), and an `options` block of raw backend flags (`n-gpu-layers`, `flash-attn`, `jinja`, `chat-template-file`, `ctx-checkpoints`, `chat-template-kwargs`, ...).

Server-process flags (`--host`, `--port`, `--models-max`, `--cache-reuse`)
live in the **compose command**, not here. The llama.cpp adapter generates
the llama-server preset from this file; see
`documents/llama_server_model_notes.md`.

**Removed feature (hard fail):**

* Any `grammar_path` in a backend `options` block triggers an error at startup. Grammar support is gone.&#x20;

---

## How the pieces fit (server behavior)

1. **Startup**

   * Load `agent_config.json` → pick the single active model; init **worker pool** with `pool_size`.&#x20;
   * Load all `app/agents/*.agent.json` into an **agent registry** (strict validation & path resolution).&#x20;
   * Build the **memory registry** from `memory.strategies` (e.g., `thread_window`).&#x20;

2. **Chat request**

   * Client sends `{ agent, text, thread_id? }`.
   * Server looks up the agent preset → applies `system_prompt` and `params_override`.
   * If the preset’s `memory_policy` ≠ `none`, the server resolves the strategy from the registry and **requires** `thread_id`; it builds a preamble (rolling transcript trimmed to `max_context_tokens`) and appends new turns on the fly. Errors if the strategy isn’t configured or `thread_id` is missing. &#x20;
