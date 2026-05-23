# How to Create and Use a New Agent in agent_server

This guide describes, step by step, how to add a new **agent role** (its
prompt and configuration) to `agent_server`, and how to call it.

## What an "agent" is here

In `agent_server`, an **agent** (also called an *agent preset*) is a named
bundle of two things:

- a **system prompt** - a plain `.txt` file, and
- a small **JSON configuration** - sampling parameters plus a memory policy.

Every agent runs on the **same active model** (the one marked
`"active": true` in `agent_config.json`). An agent does **not** select a
model - it only changes the system prompt and the sampling behaviour.
Creating an agent is therefore just adding two files and restarting the
service.

Existing agents live in `data/agents/` - for example `planner`,
`tool_author`, `researcher`, `idealab_relation_extractor`, `router`.

## Where the files live

| On the host | Inside the container |
|---|---|
| `~/env/assets/agent_server/data/agents/<name>.agent.json` | `/agent_server/app/data/agents/<name>.agent.json` |
| `~/env/assets/agent_server/data/prompts/<name>_system_prompt.txt` | `/agent_server/app/data/prompts/<name>_system_prompt.txt` |

`agent_server/data/` is **bind-mounted** into the container
(`data:/agent_server/app/data:ro`). That means:

- Files you add on the host appear inside the container immediately -
  **no image rebuild is needed.**
- But the agent registry is built **once at startup**, so a new agent is
  picked up only after an **`agent_server` restart**.

## How agents are loaded

At startup, `app/main.py` -> `load_agent_presets()` scans
`data/agents/*.agent.json`, validates each file **strictly** (no silent
fallbacks), and registers it under its `name`. The registry is then
reachable three ways: the Socket.IO `Chat` event, the
`GET /v1/agents/{name}` REST endpoint, and the agent name used as the
`model` field of `POST /v1/chat/completions`.

A malformed file (bad JSON, missing `name`, a forbidden key) raises an
error and **`agent_server` will not start** - so always check the logs
after a restart.

---

## Part 1 - Create the agent

The steps below create an example agent called **`keyword_extractor`**
that extracts keywords from text as JSON.

### Step 1 - Write the system prompt

Create `~/env/assets/agent_server/data/prompts/keyword_extractor_system_prompt.txt`:

```text
You are a keyword extraction engine. Given a block of text, identify the
5 to 10 most important keywords or key phrases.

Respond ONLY with a JSON object of this exact shape:
{"keywords": ["keyword one", "keyword two"]}

Do not add commentary, explanations, or markdown fences.
```

The filename is a convention (`<name>_system_prompt.txt`); the agent JSON
below is what actually points to it.

### Step 2 - Write the agent definition

Create `~/env/assets/agent_server/data/agents/keyword_extractor.agent.json`:

```json
{
  "name": "keyword_extractor",
  "system_prompt": "/agent_server/app/data/prompts/keyword_extractor_system_prompt.txt",
  "params_override": {
    "max_tokens": 512,
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05
  },
  "memory_policy": "none"
}
```

The `system_prompt` value is the **container** path
(`/agent_server/app/data/prompts/...`), not the host path. See the
configuration reference below for every field.

### Step 3 - Restart agent_server

```bash
docker restart agent_server
```

No rebuild: `data/` is bind-mounted, so the restart simply makes the
loader re-scan `data/agents/`.

Startup is **not instant**. agent_server rebuilds its model engines
before the agent registry loads and the HTTP port begins accepting
requests - allow roughly **30-60 seconds**. A request sent too early
fails with a connection error (curl reports `HTTP 000`); just retry.

### Step 4 - Verify it loaded

Check the startup logs:

```bash
docker logs --tail 40 agent_server | grep -iE "agents|keyword_extractor"
```

You should see a line `[agents] loaded 'keyword_extractor' from
keyword_extractor.agent.json` and a final `Agents: [...]` list that
includes it.

Or query the agent directly:

```bash
curl -s --retry 30 --retry-delay 1 --retry-connrefused \
  http://localhost:7701/v1/agents/keyword_extractor | python3 -m json.tool
```

This returns the resolved preset (full system prompt text + params). A
`404` means it did not load - re-check the JSON and the logs.

---

## Part 2 - Use the agent

There are two ways to invoke an agent.

### Option A - Programmatically (REST), for services and scripts

This is how `noted` and `idealab` use agent presets. There are two REST
styles - prefer **A1** unless a service needs the preset itself.

#### A1 - One call, by agent name (recommended)

POST to agent_server's chat-completions endpoint with the **agent name**
in the `model` field. agent_server finds that name in its registry and
applies the agent's system prompt and sampling for you - you send only
the user input.

```python
import httpx

AGENT_SERVER = "http://agent_server:7701"   # or http://localhost:7701 from the host

resp = httpx.post(f"{AGENT_SERVER}/v1/chat/completions", timeout=120, json={
    "model": "keyword_extractor",        # the AGENT name, not a model id
    "messages": [
        {"role": "user", "content": "Your input text goes here..."},
    ],
}).json()

print(resp["choices"][0]["message"]["content"])
```

No system message and no `params_override` are needed - the agent preset
supplies both.

#### A2 - Two calls, fetch the preset then run the model

Use this only when a service needs the resolved preset itself - to log
the exact prompt, or to call a model host directly. Fetch the preset,
then run the model:

```python
import httpx

AGENT_SERVER = "http://agent_server:7701"
LLM          = "http://llama-vision:8500"   # the model host

# 1. fetch the preset: {"name","system_prompt","params_override","memory_policy"}
preset = httpx.get(f"{AGENT_SERVER}/v1/agents/keyword_extractor", timeout=10).json()

# 2. call the model with the preset's prompt + sampling
resp = httpx.post(f"{LLM}/v1/chat/completions", timeout=120, json={
    "model": "gemma-4-e4b-it-q4-kxl-gguf",   # real model id - see below
    "messages": [
        {"role": "system", "content": preset["system_prompt"]},
        {"role": "user",   "content": "Your input text goes here..."},
    ],
    **preset["params_override"],
}).json()

print(resp["choices"][0]["message"]["content"])
```

Real reference in the codebase:
`noted/backend/app/workflow/llm_dispatcher.py` (`_fetch_preset_config()`
and `dispatch_claude()`).

#### Finding the model id and agent names

`GET http://localhost:7701/v1/models` lists every value accepted in the
`model` field: the **active model** - the entry that carries a
`display_name` (currently `gemma-4-e4b-it-q4-kxl-gguf`) - plus **every
agent name**. The model id is **not** simply `gemma-4`; a wrong value
fails with `404 model_not_found`. There is no list endpoint for agents
(`GET /v1/agents` with no name returns 404) - enumerate them via
`/v1/models`.

### Option B - Interactively (Socket.IO), for the browser / chat UI

Connect to agent_server's Socket.IO endpoint (port `7701`) and emit a
`Chat` event:

```js
socket.emit("Chat", {
  agent: "keyword_extractor",
  text:  "Your input text goes here...",
  // thread_id: "abc",   // required only if memory_policy != "none"
  // memory: "none",     // optional, overrides the preset's policy
});
```

The server streams the response back as events on the same connection:

- `RunStarted` `{ runId }`
- `ChatChunk` `{ runId, chunk }` - emitted repeatedly with output deltas
- `ChatDone` `{ runId }` - completion
- `Error` `{ runId, message }` or `Interrupted` `{ runId }`

Emit an `Interrupt` event to cancel an in-flight run.

---

## Configuration reference (`<name>.agent.json`)

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique agent id. Lower-cased by the loader. This is the name you call the agent by. |
| `system_prompt` | yes | Path to the system-prompt `.txt`. An absolute container path is recommended (`/agent_server/app/data/prompts/...`); a relative path is resolved against the `.agent.json`'s own folder. The key **must** be `system_prompt` - `system_prompt_path` is rejected. |
| `params_override` | no | Sampling overrides merged at request time: `max_tokens`, `temperature`, `top_k`, `top_p`, `min_p`, `stop` (array), `chat_template_kwargs` (for example `{"enable_thinking": false}` to disable Gemma's thinking channel). |
| `memory_policy` | no (default `none`) | `"none"` = stateless. `"thread_window"` = rolling per-conversation memory; callers must then pass a `thread_id`. |
| `tts_field` | no | If the agent emits JSON and only one field should be spoken aloud, name it here; the server extracts that field for TTS instead of streaming everything. |
| `grammar_path` | forbidden | Grammar support was removed; including this key fails startup. |

All other generation defaults come from the active model in
`agent_config.json`. `params_override` only overrides sampling - never
the model itself.

## Gotchas

- **Restart, do not rebuild.** New agent files need
  `docker restart agent_server`. An image rebuild is only for `app/`
  code changes.
- **A restart is not instant.** The HTTP port needs ~30-60s after a
  restart before it answers; an early call fails with `HTTP 000`. Retry,
  e.g. `curl --retry 30 --retry-delay 1 --retry-connrefused ...`.
- **The `model` field is not `gemma-4`.** Pass an *agent name* there to
  run that agent (A1), or the real model id to run the raw model. The
  real id (`gemma-4-e4b-it-q4-kxl-gguf`) comes from `GET /v1/models`; a
  wrong value fails with `404 model_not_found`.
- **A bad agent file breaks startup.** The loader is strict: invalid
  JSON, a missing `name`, `system_prompt_path`, or `grammar_path` raises
  an error and agent_server will not come up. Validate JSON before
  restarting: `python3 -m json.tool < <file>.agent.json`.
- **`name` is the call key**, not the filename. Keep them matching to
  avoid confusion, but the registry keys on the JSON `name` field.
- **Duplicate `name`s silently collide** - the last file (alphabetical
  order) wins. Keep names unique.
- **`thread_window` requires `thread_id`.** If you set that policy, every
  caller must pass a `thread_id` or the run errors.

## Optional - router awareness

agent_server runs a `router` agent on every interactive message
(`RouterDispatcher`) and emits a `RouterResult` classification to the
client. A new agent is fully usable **without** touching the router -
you call it by name. Only if you want the router's classification to be
*aware* of the new agent would you edit
`data/prompts/router_system_prompt.txt`. For programmatic agents (the
common case), the router is not involved.

---

## Quick checklist

1. Add `data/prompts/<name>_system_prompt.txt`.
2. Add `data/agents/<name>.agent.json` (valid JSON; `name`,
   `system_prompt`, `params_override`, `memory_policy`).
3. `docker restart agent_server`.
4. Confirm with `docker logs agent_server | grep agents` or
   `curl http://localhost:7701/v1/agents/<name>`.
5. Call it - Option A1 (one REST call, `model` = the agent name) for
   services, Option B (Socket.IO `Chat`) for the chat UI.
