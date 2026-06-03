# Agent Server Admin UX - Spec & Implementation Plan

A simple admin UI and REST API in `agent_server` to manage **agents**
(full CRUD on presets) and to **view/edit the service config**, so
administration no longer means hand-editing JSON files and guessing when
a restart is needed.

Status: **SHIPPED 2026-05-20** - all phases built, verified end-to-end,
and user-confirmed live. Agent schema reference: see
[`../how_to.md`](../how_to.md).

## Status - SHIPPED (2026-05-20)

All four phases are implemented, deployed, and confirmed working:

- **Phase 0** - `data/` mount `:ro` -> `:rw`; `agent_config.json` moved
  into `data/`; `AGENT_CONFIG` env added; Dockerfile `COPY` removed;
  `.dockerignore` added (build context = `app/` only).
- **Phase 1** - `app/admin_api.py` (presets CRUD, hot-reload); shared
  `validate_agent_dict`; wired into `main.py`.
- **Phase 2** - admin UI at `/admin/` (`app/static/admin/`).
- **Phase 3** - `GET/PUT /admin/api/config` + restart-pending drift banner.
- **Addendum** - in-UI multi-turn agent tester (Section 7).

Verified: 26 agents load; presets hot-reload (a new agent is usable via
`/v1/chat/completions` with no restart); config drift banner works;
create / update / delete and the protected-`router` guard all pass.

Post-ship fix: the tab switch left the Agents view visible on the Config
tab - `#view-agents`'s `display: grid` (ID specificity) overrode the
`hidden` attribute; fixed with `#view-agents[hidden] { display: none; }`.

The sections below are the original spec/plan, kept as the design record.

---

## 1. Locked decisions

| Decision | Choice |
|---|---|
| Scope | Agent presets (full CRUD) **+** `agent_config.json` (view/edit) |
| Presets apply mode | **Hot-reload** - changes are live immediately, no restart |
| `agent_config.json` apply mode | Edit + validate via the API/UI, but **applying needs a restart** |
| Restart signalling | UI shows a **"restart pending"** banner whenever the on-disk config differs from the running (startup-loaded) config |
| Access | Single admin, trusted network - **no login/auth**. Admin routes namespaced under `/admin/` so they can be excluded from any public proxy |
| UI | Plain HTML + vanilla JS, **no framework** |

If full live reload of `agent_config.json` is ever wanted, it is a
separate, larger effort (see Section 6) - not in this plan.

---

## 2. Background - why the design is shaped this way

Facts about `agent_server` today (verified in the code):

- **Agents** = a pair of files: `data/agents/<name>.agent.json` plus a
  prompt `data/prompts/<name>_system_prompt.txt`. Loaded **once at
  startup** by `load_agent_presets()` into the module-global `AGENTS`
  dict (`app/main.py`).
- `AGENTS` is a plain dict, looked up per request, and shared **by
  reference** with the `RouterDispatcher`. Mutating it in place is
  enough to make a preset change live - this is why **presets can
  hot-reload**.
- **`agent_config.json`** lives at the repo root
  (`agent_server/agent_config.json`), `COPY`'d into the image by the
  Dockerfile (`WORKDIR /agent_server`). Its values (`pool_size`, the
  active `model`, `memory` strategies) are wired into the WorkerPool,
  the engine, and the MemoryRegistry **at startup**. Those objects are
  not designed to be swapped on a live process - this is why
  **config changes need a restart**.
- `data/` is bind-mounted **read-only**:
  `~/env/assets/agent_server/data:/agent_server/app/data:ro`. A write
  API needs it `:rw`.
- The app already serves static files: `app.mount("/", StaticFiles(...))`
  from `app/static/`. The admin UI is just another static folder.
- A malformed `.agent.json` makes `agent_server` **fail to start**. The
  write API must therefore validate before persisting, every time.

---

## 3. Specification

### 3.1 REST API

All endpoints under `/admin/api/`. No auth (trusted-network deployment).
The router must be registered in `main.py` **before** the
`app.mount("/", StaticFiles...)` line so explicit routes win.

**Agent presets**

| Method + path | Purpose |
|---|---|
| `GET /admin/api/agents` | List all presets: `[{name, memory_policy, params_override, tts_field}]` |
| `GET /admin/api/agents/{name}` | Full preset: `{name, system_prompt (full text), params_override, memory_policy, tts_field}` |
| `POST /admin/api/agents` | Create. Body below. `409` if `name` exists |
| `PUT /admin/api/agents/{name}` | Update. `404` if missing. `name` is immutable |
| `DELETE /admin/api/agents/{name}` | Delete the preset (and its prompt file) |

Create/update request body (the API edits **prompt text**, not file paths):

```json
{
  "name": "keyword_extractor",
  "system_prompt": "<full prompt text>",
  "params_override": { "max_tokens": 512, "temperature": 0.1 },
  "memory_policy": "none",
  "tts_field": null
}
```

The API owns the two-file layout: on write it creates/updates
`data/prompts/<name>_system_prompt.txt` with the text, and
`data/agents/<name>.agent.json` with `system_prompt` pointing at that
file. The caller never deals with paths.

**Service config**

| Method + path | Purpose |
|---|---|
| `GET /admin/api/config` | `{ live, on_disk, restart_pending }` |
| `PUT /admin/api/config` | Validate + atomically write `agent_config.json`. Returns `{ restart_pending: true }` |

### 3.2 Behaviours

- **Validation is shared with the startup loader.** Refactor the
  per-file checks inside `load_agent_presets()` into a single
  `validate_agent(data: dict) -> list[str]`. Both startup and the API
  call it, so they can never disagree. The API must **never** persist a
  definition that would fail at startup. Config gets an analogous
  `validate_config()` (exactly one `active` model, no `grammar_path`,
  well-formed `runtime`/`memory`).
- **Atomic writes.** Write to `<file>.tmp`, then `os.replace()`. Never
  leave a half-written `.json` on disk.
- **Hot-reload (presets only).** On a successful create/update, build
  the `AgentPreset` and assign `AGENTS[name] = preset`; on delete,
  `AGENTS.pop(name)`. The router and request handlers see it instantly
  (shared dict). No restart.
- **Config is never hot-applied.** `PUT /admin/api/config` writes the
  file only. It must **not** touch the live snapshot.

### 3.3 Drift detection ("restart pending")

- `RAW_CONFIG` (parsed at module import) is the **immutable live
  snapshot** of what the running process actually loaded.
- `GET /admin/api/config` returns:
  - `live` = `RAW_CONFIG`
  - `on_disk` = a fresh read of `agent_config.json`
  - `restart_pending` = `on_disk != live`
- Because nothing post-startup mutates `RAW_CONFIG`, this catches **any**
  divergence - edits via this UI, a hand-edit, or another tool.
- After a restart the snapshot is re-read, `on_disk == live` again, and
  the banner clears.
- **Presets never raise this banner** - they hot-reload, so their live
  state always matches disk.

### 3.4 UI

Static, under `app/static/admin/` → served at `/admin/`. Plain HTML +
vanilla JS + CSS, no framework.

- **Agents view:** a table of presets; a Create button; an edit form
  (name [immutable on edit], system-prompt textarea, sampling params -
  `max_tokens`, `temperature`, `top_k`, `top_p`, `min_p` - a
  `memory_policy` dropdown, optional `tts_field`, and an "advanced" raw
  JSON box for `stop` / `chat_template_kwargs`); Delete with a confirm.
  Save is immediate (hot-reload) with a success toast - no banner.
- **Config view:** a form for the common fields plus a raw JSON editor
  for `agent_config.json`.
- **Restart-pending banner:** on load, and after every config save, the
  UI calls `GET /admin/api/config`; if `restart_pending` is true it
  shows a persistent banner:
  *"Restart pending - the on-disk config differs from the running
  config. Run `docker restart agent_server` to apply."*

---

## 4. Implementation plan

### Phase 0 - Make `data/` writable + relocate `agent_config.json`

1. Move `agent_server/agent_config.json` → `agent_server/data/agent_config.json`.
2. `docker-compose.yml`: change the data mount (line ~126) from
   `:ro` to `:rw`.
3. `docker-compose.yml`: add to the `agent_server` service
   `environment:` -> `AGENT_CONFIG: "/agent_server/app/data/agent_config.json"`.
4. `Dockerfile`: remove the now-dead `COPY ./agent_config.json
   ./agent_config.json` line (the file no longer exists at the repo
   root; leaving the line would break the next image build). This edit
   does **not** require a rebuild now - it just prevents a future
   build failure.
5. Restart `agent_server`; confirm it starts and loads config from the
   new path.

*No image rebuild needed for Phase 0 - `data/` is a bind mount; only a
restart is required.*

**Acceptance:** `agent_server` runs; `data/` is writable from inside the
container; config is read from `data/agent_config.json`.

### Phase 1 - Presets CRUD API

1. Refactor: extract `load_agent_presets()`'s per-file validation into
   `validate_agent(data) -> list[str]`; have the loader call it.
2. New module `app/admin_api.py` - an `APIRouter` with the
   `/admin/api/agents` endpoints. Include it in `main.py` **before** the
   static mount.
3. Implement list / get / create / update / delete: validate → atomic
   write of the `.txt` + `.agent.json` pair → mutate the live `AGENTS`
   dict. Clear error messages; nothing written on a validation failure.

**Acceptance:** an agent can be created/edited/deleted via `curl`; it is
usable immediately with no restart (via the `Chat` event and
`GET /v1/agents/{name}`); an invalid definition is rejected and leaves
no files behind.

### Phase 2 - Presets admin UI

1. `app/static/admin/` - `index.html`, `app.js`, `style.css`.
2. Agent list + create/edit/delete forms wired to the Phase 1 API.
3. Keep it plain: vanilla `fetch`, no build step.

**Acceptance:** the full preset lifecycle works from the browser at
`/admin/`.

### Phase 3 - Service-config view/edit

1. `GET /admin/api/config` and `PUT /admin/api/config` with
   `validate_config()` and the drift computation from Section 3.3.
2. UI config panel + the restart-pending banner.

**Acceptance:** editing the config writes the file and raises the
banner; a restart clears it; an invalid config is rejected.

### Effort estimate

| Phase | Effort |
|---|---|
| 0 - writable data | ~30 min |
| 1 - presets API | ~0.5-1 day |
| 2 - presets UI | ~1 day |
| 3 - config edit | ~0.5 day |

---

## 5. Risks & gotchas

- **A bad write would brick the next startup.** Mitigated by the shared
  validator (Section 3.2), validate-before-write, and atomic writes.
- **`:ro` → `:rw` removes a safety property** - the running service can
  now mutate its own config. Accepted trade-off for a write API.
- **Do not single-file bind-mount `agent_config.json`.** WSL2 pins
  single-file bind mounts to the original inode; an editor's
  atomic-rename then leaves the container reading stale content. This is
  exactly why Phase 0 moves the file into the `data/` **directory**
  mount.
- **Admin surface must not be publicly proxied.** Routes are namespaced
  under `/admin/` precisely so a `proxy_server` rule can exclude them if
  `agent_server` is ever exposed.
- **Concurrent edits:** last-write-wins. Acceptable for single-admin.

---

## 6. Out of scope (future)

- **Authentication / RBAC** - only needed if this goes multi-user or
  becomes externally reachable. The `/admin/` namespace means an auth
  gate or proxy rule can be added later without restructuring.
- **Live hot-reload of `agent_config.json`** (pool resize, active-model
  swap, memory-registry rebuild on a running process) - a much larger,
  riskier effort. Restart-to-apply is the deliberate v1 behaviour.
- **Edit history / versioning** of presets and config.

---

## 7. Addendum - in-UI agent tester (added 2026-05-20)

A "Test" panel was added to the Agents view so a freshly published agent
can be exercised without leaving the UI.

- It appears below the edit form whenever a saved agent is selected, and
  automatically right after a create/update - so you can test on publish.
- It is a multi-turn chat: each send POSTs the whole conversation to
  `POST /v1/chat/completions` with `model` set to the agent name.
  agent_server resolves that name to the preset and applies its system
  prompt and sampling. **No backend change was needed** - that endpoint
  already accepts an agent name as `model`.
- It tests the agent **as published** (the hot-reloaded registry entry),
  not unsaved form edits - so the flow is: edit -> Save -> test.
- Stateless server-side: the panel keeps the message array client-side
  and resends it each turn; "New conversation" clears it.

Also added: a repo-root `.dockerignore` (`*` then `!app`) so
`docker build` ships only `app/` as context. Without it the build would
send the whole repo, including the ~74 GB `data/` directory.

---

## 8. Addendum - config schema refactor + model-switcher follow-up (2026-06-03)

`agent_config.json` was restructured so it is the single source of truth
for which model the stack runs (Gemma 4 / Qwen3.5 / Phi-4-mini-reasoning,
one resident at a time, switch-by-restart). `models` changed from a flat
array to an object **grouped by task** - `chat`, `embedding`,
`reranking` - and each entry is self-describing (neutral core:
`model_id`, `family`, `context`, `reasoning`, `vision`, `sampling`; plus
`active_backend` + `backends.<name>.options`). The llama.cpp **adapter**
(`adapter/llama_cpp_preset.py` + `adapter/entrypoint.sh`,
`Dockerfile.llama-adapter`, `docker-compose.adapter.yml`) generates the
llama-server preset from this file inside the container, so there is no
`llama-router-models.ini` to manage. See
`documents/llama_server_model_notes.md`.

**Admin-area impact (verified, no breakage):** the config tab is a
free-form JSON editor (`GET`/`PUT /admin/api/config`), so it renders and
edits the new grouped shape unchanged; `validate_config` was updated for
the dict-shaped `models` and accepts it; the agents tab is unaffected.

**FOLLOW-UP (not yet built): structured model-switcher in the admin UI.**
Today switching the active chat model means hand-editing JSON in the
config tab - no UI guardrail; a bad edit is only caught by
`validate_config` on Save. Now that `models.chat` is a typed list, the UI
could offer a structured switcher:

- list `models.chat[]` with each entry's `model_id` + a radio/toggle for
  which is active (enforce exactly one), plus a read-only summary of
  `family` / `context` / `reasoning` / `vision`;
- on activate, `PUT /admin/api/config` with the single `active` flag
  flipped, then surface the existing restart-pending banner (the switch
  still requires restarting llama-vision + agent_server - VRAM holds one
  chat model; that constraint is unchanged).

This is straightforward on the grouped schema but deliberately deferred -
the raw-JSON editor already works, and a structured switcher is a UI-only
enhancement with no backend change.
