# Active-Model Switching — Client SDK / Integration Guide

**Audience:** developers of client applications (e.g. `noted`, future tools) that talk to
`agent_server` and want to (a) discover the available local chat models, (b) request a change
of the **active** local model, and (c) be **notified** when the active model changes — from any
source — so their UI stays in sync.

**Status of this document:** the contract below is **implemented and verified** (2026-06-05) —
see the status table in §7. Notification is handled implicitly via restart + Socket.IO reconnect
(§4); there is no custom change event.

---

## 1. Background — why there is exactly one active model

`agent_server` fronts a single `llama-server` router (`llama-vision`) that holds a small number
of models resident in VRAM (`--models-max`, currently 3: the active chat model + 2 embedders).

The router **auto-loads any model that is *declared* in its preset the moment a request names
it**, and under the resident cap that evicts another model — causing multi-GB reload "thrash"
mid-request (empirically verified, 2026-06). To prevent this, `agent_server` declares **only the
active chat model** in the router preset. Consequence:

> A client **cannot** simply send `model: "qwen3.5"` to use a non-active model — only the
> active model (and the agent presets that resolve to it) are loadable. To use a different
> local model you must **switch the active model** via the API in §3. The switch regenerates
> the preset with the new model as the sole declared chat model and hot-reloads the router
> (no container restart).

This is the single-resident-model invariant. The switch API is the *only* supported way to
change which local model serves.

---

## 2. Discover available models — `GET /v1/models`

Returns the local chat models and the agent presets. Each **chat model** entry carries an
`active` flag and metadata; **agent presets** (e.g. `cv_assistant`) are listed as virtual models
that always resolve to whatever chat model is active.

**Request**
```
GET /v1/models
Host: agent_server:7701          # internal; via proxy: https://logus2k.com/llm/v1/models
```

**Response (lists ALL chat models; each `kind:"chat"` carries `active`/`family`)**
```jsonc
{
  "object": "list",
  "data": [
    {
      "id": "gemma-4",
      "object": "model",
      "owned_by": "local",
      "display_name": "Gemma 4 E4B IT",
      "family": "gemma",
      "active": true,            // <-- the currently serving local model
      "kind": "chat"
    },
    { "id": "qwen3.5",   "object": "model", "owned_by": "local", "display_name": "Qwen3.5 4B",     "family": "qwen",    "active": false, "kind": "chat" },
    { "id": "smollm3",   "object": "model", "owned_by": "local", "display_name": "SmolLM3 3B",     "family": "smollm",  "active": false, "kind": "chat" },
    { "id": "granite-3.3","object": "model","owned_by": "local", "display_name": "Granite 3.3 2B", "family": "granite", "active": false, "kind": "chat" },
    { "id": "nemotron",  "object": "model", "owned_by": "local", "display_name": "NVIDIA Nemotron Nano 4B", "family": "nemotron", "active": false, "kind": "chat" },

    { "id": "cv_assistant", "object": "model", "owned_by": "local", "kind": "agent" }
    // ... other agent presets ...
  ]
}
```

> Clients should treat `kind: "chat"` entries as selectable models (the one with `active: true`
> is currently serving) and ignore `kind: "agent"` entries for a model picker.

Clients that only need "what's active right now" can read the entry where `active === true`.

### Registered model IDs

These are the canonical `model_id`s. Use them verbatim in the `model_id` body of the switch call
(§3). The single `active` one is also the only id valid in the `model` field of
`/v1/chat/completions` (agent ids resolve to it server-side). **Always prefer the live
`GET /v1/models` list over hard-coding** — the set changes as models are added/removed.

**Local chat models (`owned_by:"local"`, `kind:"chat"`) — switchable via §3:**

| `model_id` | family | notes |
|------------|--------|-------|
| `gemma-4`     | gemma   | E4B vision-capable; reliable citations (5/5); **recommended CV default** |
| `qwen3.5`     | qwen    | 4B reasoning; reliable citations (5/5); good CV alt |
| `smollm3`     | smollm  | 3B; **over-cites heavily** (`[R:]` spam, 30-90s) — avoid for CV |
| `granite-3.3` | granite | 2B; fast but citations unreliable (~1/5) |
| `nemotron`    | nemotron| NVIDIA Nemotron Nano 4B (`nemotron_h`); reasoning + `enable_thinking`; loads OK, not CV-stress-tested |
| `ministral`   | mistral | Ministral 3B Reasoning 2512 (`mistral3`); thinks via `[THINK]`→`<think>`; **over-cites + slow (≤30s) + empty-answer turns + VRAM-heavy (Q8)** — not a good CV default |

> CV-suitability summary: **gemma-4 / qwen3.5** cite cleanly every turn; **smollm3 / ministral**
> over-cite (spam) and are slow; **granite-3.3** under-cites; **nemotron** loads but wasn't
> stress-tested in the CV. All six are switchable; only the citation/latency behaviour differs.

**Agent presets (`kind:"agent"`)** — virtual ids that always resolve to the *active* local model
server-side. You send these as the `model` field for chat; you do **not** switch them. The CV app
uses `cv_assistant`; there are ~30 others (`router`, `general`, `noted`, …). Get the full set from
`GET /v1/models` (filter `kind:"agent"`).

**Cloud models** — `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` are **not**
agent_server models; they live in `noted` (Anthropic). `noted`'s `/api/llm/health` merges them
with the local list for its dropdown, but agent_server's `/v1/models` never returns them.

---

## 3. Request a switch — `POST /admin/api/active-model`

Changes the active local chat model. Server-to-server call (your **backend** calls it, not the
browser — the admin surface lives on the trusted internal network, not the public proxy).

**Request**
```
POST /admin/api/active-model
Content-Type: application/json

{ "model_id": "qwen3.5" }
```

**Behaviour (restart-based — chosen approach)**
1. Validates `model_id` is a configured chat model (`404`/`422` otherwise — see errors).
2. If already active → no-op success (`noop: true`), no restart.
3. Flips the `active` flag in `agent_config.json` (atomic write); the choice now persists.
4. Restarts **`llama-vision`** and **`agent_server`**. On boot:
   - the llama-vision adapter **regenerates the preset** from the (mounted) config with the new
     model as the sole declared chat model (the single-resident-model invariant holds — no
     thrash window), and
   - `agent_server` re-reads `RAW_CONFIG` and re-derives everything (active id / family /
     system-prompt variant / engine upstream id) cleanly — no runtime hot-reload code.
5. Clients re-sync automatically (see §4): the restart drops Socket.IO connections; on reconnect
   they re-run their health check and pick up the new active model. **No custom event required.**

> **Why restart, not hot-reload:** `agent_server` bakes the active model into module globals and
> into the engine pool's upstream `default_model` at startup, and is "restart-to-apply" by
> design. Re-deriving all of that live (and rebuilding the worker pool mid-flight) is fragile for
> no real benefit — a model switch is deliberate and infrequent, so a brief restart is the robust
> choice. The adapter already regenerates the preset on llama-vision boot, so "restart
> llama-vision" *is* the preset switch.

**Response** — returned *before* the restart fires (the client should expect the socket to drop):
```jsonc
{
  "status": "switching",
  "active_model": "qwen3.5",
  "display_name": "Qwen3.5 4B",
  "family": "qwen",
  "note": "agent_server is restarting (~10-20s); reconnect and re-fetch /v1/models to confirm."
}
```

**Latency (measured):** ~**30–45 s** end-to-end. The endpoint restarts **llama-vision first**
(blocking — it must stop, regenerate the preset, and **reload the new multi-GB GGUF**, ~30 s),
**then** agent_server (~8 s). agent_server stays up during the llama-vision phase, then is briefly
unreachable (~8–10 s) during its own restart. In-flight chats during the window are dropped; the
first request after settle is warm. (Earlier estimates of ~10–20 s undercounted the model reload.)

**Restart trigger (deployment decision):** `agent_server` cannot cleanly restart its own process
mid-request. Two options:
- **Docker socket** mounted into `agent_server` → the endpoint shells `docker restart
  llama-vision agent_server` (spawned detached so the response is flushed first). Simplest;
  couples `agent_server` to the Docker daemon (trusted-box trade-off).
- **Sidecar/watcher** with Docker access that `agent_server` signals (flag file / ping) → it
  performs the restart. Keeps the Docker socket out of `agent_server`.

**Errors**
| HTTP | When | Body |
|------|------|------|
| `404` | `model_id` is not a configured chat model | `{ "detail": "unknown model_id 'x'; available: [...]" }` |
| `200` | already active (no-op) | `{ "status": "ok", "active_model": "x", "noop": true }` |
| `422` | malformed body | `{ "detail": "model_id is required" }` |
| `200` | Docker unreachable | `{ "status":"ok", "restart_pending":true, "note":"...auto-restart unavailable; restart manually" }` (config written; applies on next manual restart) |

> **Auth note:** local-model switches are free (on-prem) and need no secret. If your deployment
> wants to gate switching, put it behind the same trusted-network rule as the rest of
> `/admin/api`, or add a shared-secret header. (Contrast: cloud models like `claude-*` in `noted`
> are gated by `noted`'s own access key — that gating lives in `noted`, not here.)

---

## 4. Receive change notifications — via reconnect + re-fetch (no custom event)

Because the switch **restarts `agent_server`** (§3), notification is implicit and requires no new
event:

1. The restart drops every client's Socket.IO connection to `agent_server` (`/socket.io`, public
   path `/llm/socket.io`).
2. Clients reconnect automatically (e.g. `socket.io-client` with `reconnection: true`).
3. **On reconnect, re-fetch `GET /v1/models`** and update the model picker from the entry where
   `active === true`. This catches *any* switch source — your own client, another client, or an
   operator — because everyone reconnects to the freshly-restarted server.

**Client pattern (browser, socket.io-client):**
```js
socket.on('connect', async () => {
  // runs on first connect AND every reconnect (e.g. after an active-model switch restart)
  const { data } = await fetch('/llm/v1/models').then(r => r.json());
  const active = data.find(m => m.kind === 'chat' && m.active);
  if (active) modelPicker.setActive(active.id, active.display_name);
});
```

> **Why no broadcast event:** with the restart-based switch the connection drop *is* the signal,
> and a health/`models` re-fetch on (re)connect is something well-behaved clients already do. A
> dedicated `ActiveModelChanged` event would only matter for a *no-restart* switch (not the chosen
> design). If a client cannot hold a socket, polling `GET /v1/models` on an interval is the
> fallback.

---

## 5. Worked example — `noted`'s model dropdown

`noted` already has the dropdown machinery; the change is to feed it all four local models and to
relay the switch + event. Flow:

```
Browser (ChatPanel dropdown)
  │  user picks a LOCAL model (e.g. "Qwen3.5 4B")
  ▼
noted frontend  ──POST /api/llm/model {model_id}──▶  noted backend (llm_router)
                                                        │  model is local (not claude-*)
                                                        ▼
                                          POST agent_server /admin/api/active-model {model_id}
                                                        │  flips config + restarts
                                                        ▼
                                     agent_server + llama-vision RESTART (~10-20s)
                                                        │
   sockets drop ─▶ noted reconnects ─▶ re-fetch /v1/models ─▶ setModels(models, active)
                                                        (dropdown reflects new active everywhere)
```

- **Listing:** `noted` backend `llm_router.health()` already merges local + Anthropic models for
  the dropdown. Change it to pass through **all** `kind:"chat"` models from
  `agent_server GET /v1/models` (not just `models[0]`), tagging them `backend:"local"`. Claude
  entries are unchanged.
- **Selecting a local model:** route to `POST /admin/api/active-model` on `agent_server` (the
  switch), instead of only setting `noted`'s in-memory `_active_model`. Selecting a `claude-*`
  model keeps `noted`'s existing Anthropic path (no `agent_server` call). Show a brief
  "switching…" state while the restart completes.
- **Dynamic reflection (no new event):** `noted` already re-runs its health check that calls
  `setModels(models, activeModel)`; ensure it fires **on socket reconnect** (after the restart)
  as well as on load. That alone keeps every open tab's dropdown in sync — no `ActiveModelChanged`
  plumbing needed.
- **Claude vs local:** unchanged — the dropdown shows the 4 local models *and* the Claude models;
  local picks switch the `agent_server` active model, Claude picks route to Anthropic.

---

## 6. Notes for `cv` (and other model-agnostic clients)

The `cv` conversational-CV app needs **no changes**. It always calls the `cv_assistant` agent,
which resolves to whatever chat model is active server-side. After a switch, the next CV turn
transparently uses the new model. A client only needs §2–§4 if it wants to *show* or *control*
which model is active.

---

## 7. Code examples

Base URL is agent_server: internal `http://agent_server:7701`, or via proxy `https://logus2k.com/llm`.

### 7.1 Discover models + the active one

```bash
# All chat models + which is active (curl + jq)
curl -s http://agent_server:7701/v1/models \
  | jq '.data[] | select(.kind=="chat") | {id, display_name, active}'
```

```python
# Python (requests)
import requests
data = requests.get("http://agent_server:7701/v1/models").json()["data"]
chat   = [m for m in data if m.get("kind") == "chat"]
active = next((m["id"] for m in chat if m.get("active")), None)
print("available:", [m["id"] for m in chat], "| active:", active)
```

### 7.2 Switch the active model

```bash
curl -s -X POST http://agent_server:7701/admin/api/active-model \
  -H 'Content-Type: application/json' -d '{"model_id":"qwen3.5"}'
# -> {"status":"switching","active_model":"qwen3.5",...}
# NOTE: agent_server + llama-vision restart (~30-45s). Poll /v1/models until it
# answers again, then confirm the active flag flipped.
```

```python
import requests, time
def switch(model_id, base="http://agent_server:7701"):
    r = requests.post(f"{base}/admin/api/active-model", json={"model_id": model_id})
    r.raise_for_status()
    res = r.json()
    if res.get("noop"):
        return res                       # already active, no restart
    # wait for the restart to settle, then confirm
    for _ in range(30):
        time.sleep(2)
        try:
            data = requests.get(f"{base}/v1/models", timeout=3).json()["data"]
            if any(m["id"] == model_id and m.get("active") for m in data if m.get("kind")=="chat"):
                return {"status": "active", "active_model": model_id}
        except requests.RequestException:
            pass                         # agent_server still restarting
    raise TimeoutError("switch did not settle in time")
```

### 7.3 Consume chat (uses whatever is active)

```bash
# Send the ACTIVE model id, or an agent name (e.g. cv_assistant) which resolves to it.
curl -s -X POST http://agent_server:7701/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cv_assistant","stream":false,
       "messages":[{"role":"user","content":"Hello"}]}'
```

```python
# Requesting an INACTIVE local model (e.g. "smollm3" while "qwen3.5" is active)
# fails at the router (it is not declared) - switch first (7.2), then chat.
import requests
requests.post("http://agent_server:7701/v1/chat/completions", json={
    "model": "cv_assistant",            # or the active model_id
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": False,
}).json()
```

### 7.4 Stay in sync after a switch (browser, socket.io-client)

```js
import { io } from 'socket.io-client';
const socket = io('https://logus2k.com', { path: '/llm/socket.io', transports: ['websocket'] });

async function refreshActive() {
  const { data } = await fetch('https://logus2k.com/llm/v1/models').then(r => r.json());
  const active = data.find(m => m.kind === 'chat' && m.active);
  if (active) modelPicker.setActive(active.id, active.display_name);
}
// Fires on first connect AND every reconnect (e.g. after a switch restart):
socket.on('connect', refreshActive);
```

### 7.5 Via `noted` (already wired)

```bash
# Against the noted app origin (relative path the frontend uses; noted backend
# on host = :8123). A LOCAL pick is proxied to agent_server's switch; a claude-*
# pick stays in noted (Anthropic) and needs the access key (`secret`).
curl -s -X POST http://localhost:8123/api/llm/model \
  -H 'Content-Type: application/json' -d '{"model_id":"qwen3.5"}'
# Dropdown source: GET /api/llm/health -> {models:[...local+claude], active_model}
```

---

## 7b. Toggling thinking (reasoning) on/off — per request

All five local models are reasoning models. **Thinking is controlled per request** via
`chat_template_kwargs` in the `/v1/chat/completions` body — agent_server forwards it to
llama-vision's chat template, and folds any reasoning into a `<think>…</think>` block in the
response `content`. The model's config sets a **default**; the per-request value **overrides** it
for that call (no switch/restart needed — this is independent of which model is active).

**The kwarg name differs by family:**

| model(s) | family | kwarg | values |
|----------|--------|-------|--------|
| `gemma-4`, `qwen3.5`, `smollm3`, `nemotron` | gemma/qwen/smollm/nemotron | `enable_thinking` | `true` / `false` |
| `granite-3.3` | granite | `thinking` | `true` / `false` |
| `ministral` | mistral | *(none)* | controlled by the **system prompt** — see note |

> **Ministral (`mistral` family) is the exception:** its template has no thinking kwarg. Thinking
> lives in the template's *default system message* (`[THINK]…[/THINK]` format), which a custom
> system prompt **replaces**. So a per-request `chat_template_kwargs` toggle does **nothing** for it
> — to control thinking you include/omit the `[THINK]` directive in the system prompt. (llama.cpp
> still folds its `[THINK]` into `reasoning_content` → `<think>`, so the rendered output matches the
> others.)

```bash
# Thinking OFF (verified on nemotron: no <think>, answer "4")
curl -s -X POST http://agent_server:7701/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "cv_assistant", "stream": false,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [{"role":"user","content":"What is 2+2?"}]}'

# Thinking ON (default for these models) -> reply contains <think>...</think>
curl -s -X POST http://agent_server:7701/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "cv_assistant", "stream": false,
    "chat_template_kwargs": {"enable_thinking": true},
    "messages": [{"role":"user","content":"What is 2+2?"}]}'

# Granite uses a DIFFERENT kwarg name:
#   "chat_template_kwargs": {"thinking": false}
```

```python
import requests
def chat(text, think=True, model="cv_assistant", base="http://agent_server:7701"):
    # active model is granite-3.3? use {"thinking": think}; otherwise {"enable_thinking": think}
    kw = {"thinking": think} if model == "granite-3.3" else {"enable_thinking": think}
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": model, "stream": False,
        "chat_template_kwargs": kw,
        "messages": [{"role": "user", "content": text}],
    })
    return r.json()["choices"][0]["message"]["content"]
```

Notes:
- Works whether you send a chat `model_id` (the active one) or an **agent** name (`cv_assistant`)
  — the kwarg rides through to the underlying active model.
- With thinking **on**, the reasoning is wrapped as `<think>…</think>` at the start of `content`
  (agent_server's splice); with it **off**, no such block is emitted.
- The CV pipeline keeps thinking **on** by default (the per-turn directive re-asserts it). To make
  a model answer without reasoning, pass `enable_thinking:false` (or `thinking:false` for granite)
  on that request.

---

## 8. Implementation status

| Piece | Status |
|-------|--------|
| `GET /v1/models` extended to all chat models + `active`/`family`/`kind` | **DONE + verified** (`app/openai_compat.py`) |
| `POST /admin/api/active-model` (validate + flip `active` flags + restart both via docker SDK) | **DONE + verified** (`app/admin_api.py`; switch/no-op/404 tested) |
| Restart trigger = docker socket in agent_server (`docker` SDK) | **DONE** (Dockerfile `pip install docker`; compose mounts `/var/run/docker.sock`) |
| Preset regeneration on llama-vision boot from config | **exists** (adapter entrypoint) |
| Per-request thinking toggle (`chat_template_kwargs`; §7b) | **verified** (enable_thinking on/off on nemotron; granite uses `thinking`) |
| `noted` backend: list all local models (`llm_manager`/`llm_router` health) + route local pick → switch API (`llm.py`) | **DONE + verified** (health lists 4 local + claude; pick triggers switch) |
| `noted` frontend: `_checkHealth()` on socket reconnect | **DONE** (`ChatService.js`) — wired + data source verified; browser dropdown not driven (pre-existing `setModels` render) |
| `cv`: no change | n/a |
| `ActiveModelChanged` Socket.IO broadcast | **dropped** — implicit via restart+reconnect (§4) |

**Key constraint to preserve:** never declare more than the active chat model in the router
preset (the single-resident-model invariant, §1). The restart-based switch upholds this for free
— the adapter regenerates the preset with only the new active model on llama-vision boot.
