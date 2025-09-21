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
  "models": [
    {
      "name": "jan-nano-128k-Q4_K_M",
      "path": "./models/jan-nano-128k-Q4_K_M.gguf",
      "active": true,
      "mode": "chat",
      "system_prompt": "",
      "params": {
        "n_ctx": 8192,
        "n_threads": 0,
        "n_gpu_layers": -1,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.1,
        "max_tokens": 512
      }
    }
  ]
}
```



## `runtime`

* **pool\_size** (int): number of **model workers** created at startup.&#x20;
* **per\_request\_timeout\_s** (int, 0 disables): hard timeout for a `Chat` run; server cancels and emits an error on expiry.&#x20;

## `memory`

Defines available **memory strategies** the server can use when an agent’s `memory_policy` requests one.

* **strategies.thread\_window\.max\_context\_tokens** (int): token budget used to build the preamble from prior turns (rough \~4 chars/token heuristic). If omitted, defaults to 1024. &#x20;

> The registry is constructed from this section at startup; currently only `thread_window` is implemented.&#x20;

## `models[]`

Exactly one active model is required. These values are read once and used by the engine factory & worker pool. &#x20;

* **name** (string): identifier (for logs).
* **path** (string): `.gguf` model path.
* **active** (bool): must be `true` for one entry.
* **mode** (string): currently `"chat"`.
* **system\_prompt** (string): optional **model-level default**; agent presets **override per-request**. Empty string is allowed.&#x20;
* **params** (object): model/runtime & default generation params passed to `llama-cpp-python`. Typical keys you already use:

  * **Model init / runtime:** `n_ctx`, `n_threads`, `n_gpu_layers` (and optionally `n_batch`, `n_ubatch` in other examples).&#x20;
  * **Generation defaults:** `temperature`, `top_k`, `top_p`, `min_p`, `max_tokens` (and optionally `stop`). Agent `params_override` can override these per request. &#x20;

**Removed feature (hard fail):**

* Any `grammar_path` in the model block triggers an error at startup. Grammar support is gone.&#x20;

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
