# GPU Service Control — per-container start/stop from the admin area

**Status:** PLAN (not built). Supersedes the earlier "GPU profiles" idea — the user
chose individual container control instead of named bundles.

**Goal:** From the agent_server admin UI, start and stop each GPU-consuming
container individually, with live status and a visible VRAM-headroom readout, so the
operator can free/allocate the RTX 4090's 24 GB on demand. Individual control + the
existing active-model switch already reproduces every scenario the operator wanted:

| Desired state | How it's reached |
|---|---|
| GPU clean of assistant models | Stop `llama-vision`, `stt_server`, `tts_server`, `avatar_server` |
| All up, LLM = Gemma 4 E4B | Start all + switch active model to `gemma-4` |
| Gemma 4 12B, no avatar/STT/TTS | Stop those three; active model = `gemma-4-12b` |
| Clean + Muse-Glimmer-30B | Stop the others; active model = `muse_glimmer_30b` (needs registration+test first — separate task) |

## Verified groundwork (this session, 2026-08-11)

- `/var/run/docker.sock` **is** mounted on the live `agent_server` container; docker-py
  `docker.from_env().ping()` → `True`. The existing active-model switch already uses this
  exact mechanism (`admin_api.py` `set_active_model` → `client.containers.get(name).restart()`).
- All sibling GPU services **eager-load** their model at container start and expose **no
  unload API** — so **stopping the container is the only way to free their VRAM** (confirmed
  by reading each service's startup path).
- Sibling containers currently exist as `Exited` (not removed), so docker-py `.start()` /
  `.stop()` by container name works without any compose invocation.

## Controlled-container registry (the "listed services" scope)

Per the operator's scope choice ("only my listed services"), the panel controls these and
**not** unrelated GPU apps (tutor, scipredictor, games, noted-serving, airflow-worker):

| Container | Label | GPU footprint | Notes / guardrails |
|---|---|---|---|
| `llama-vision` | LLM + embeddings (bge-m3) + reranker host | dominant (active model + residents) | Stopping it takes the LLM/embeddings offline for `agent_server`, `noted`, `cv`. Starting it **reloads the active model** from `agent_config.json` (~30–45 s, heavy). |
| `stt_server` | Speech-to-text (Whisper) | ~1.6 GB | A `stt_server_v2` (Parakeet, profile `v2`) is a mutually-exclusive alternative on the same host port 2700 — v1 is the default; surface v2 only as a note for now. |
| `tts_server` | Text-to-speech (Kokoro) | ~0.33 GB | — |
| `avatar_server` | Talking-head (LivePortrait/TRT) | ~2 GB/session + shared engines | Heaviest sibling; currently Exited (137). Uses `runtime: nvidia`. |
| `embeddings-server` | Reranker (bge-reranker-v2-m3) for noted | ~1.1 GB | **On `noted-network`, serves noted's RAG reranking.** Stopping it degrades noted retrieval. Show it with a dependency warning; default to *not* stopping it in casual use. |

**Never controllable:** `agent_server` itself (it is the control plane — stopping it kills
the panel). Enforce with an explicit deny + omission from the registry.

The registry lives in config (`data/config/gpu_services.json`) or a module constant so the
list is easy to extend: each entry `{name, label, description, warn?, kind: "gpu-service"}`.

## Backend (app/admin_api.py, mirrors the active-model pattern)

1. **`GET /admin/api/services`** → for each registry entry:
   `{name, label, status, uptime, health, warn}`. Status from docker-py
   `containers.get(name).status` (`running` | `exited` | map 404 → `absent`). Include the
   aggregate GPU line (total/used/free) already produced by `GET /admin/api/status` so the UI
   can show headroom in the same view. **Per-container VRAM attribution is not possible on
   WSL2 nvidia-smi** (no per-process memory) — show aggregate only, and say so.
2. **`POST /admin/api/services/{name}/start`** → allowlist-checked; docker-py `.start()`.
   If `.get()` 404s → return actionable error ("container absent — run `docker compose up`
   for that stack once; start/stop only manages existing containers").
3. **`POST /admin/api/services/{name}/stop`** → allowlist-checked; docker-py `.stop(timeout=30)`.
   Deny `agent_server`.
4. Reuse `_atomic_write`/validation only if the registry is file-backed. No `agent_config.json`
   mutation is needed for pure start/stop (that's the model-switch feature's job).
5. Widen the `/admin/api/logs` allowlist (currently `{agent_server, llama-vision}`) to the
   registry names if per-service logs are wanted in the panel (nice-to-have, Phase 3).

**Ordering / VRAM safety:** the panel does *not* auto-orchestrate; the operator drives it.
But because starting a heavy model while others hold VRAM can hit the exact
draft-memory / OOM failure diagnosed earlier (the `vector::_M_range_check` = out-of-VRAM
crash), the **VRAM-headroom readout is a first-class element** so the operator stops-before-
starting. For `llama-vision` specifically, its start reloads the active model and can be slow
or fail under low headroom — surface that.

## Frontend (app/static/admin/, follows the existing tab + overlay pattern)

- New top-level tab **"Services"** (`<button class="tab">` + `<section id="view-services">`),
  wired into `showTab()` exactly like the current tabs.
- One row/card per container: **status badge** (green running / grey stopped / red absent),
  uptime, a **Start** or **Stop** button (enabled by current state), and a ⚠ dependency note
  for `llama-vision` and `embeddings-server`.
- **GPU headroom bar** at the top (total/used/free from `/status`), so the effect of each
  stop/start is visible.
- **Confirm dialog** before stopping `llama-vision` or `embeddings-server` (they have external
  dependents).
- Reuse the existing **`runSwitch()` milestone overlay** for `llama-vision` start (it's slow —
  poll until the model actually serves, not just container `running`), matching the
  active-model switch UX.

## Live status — event-driven, no polling

The operator explicitly does not want polling. Extend the **existing admin Socket.IO room**
(already built for the Clients tab: `admin:subscribe` → room "admin", `_admin_clients_pusher`)
to also emit `admin:services` with each container's status, plus an immediate re-emit right
after any start/stop action so the UI reflects the change without a timer. Fall back to a
one-shot `GET /services` refresh if Socket.IO is unavailable.

## Phasing

- **Phase 1 — Backend:** registry + `GET /services` + `POST start|stop` (docker-py, allowlist,
  `agent_server`-protected, absent handling). Testable via curl against `:7701`.
- **Phase 2 — Frontend:** Services tab, status badges, start/stop buttons, GPU headroom bar,
  confirm dialogs, `runSwitch` overlay for `llama-vision` start. Rebuild the `agent_server:1.0`
  image (static is baked) and verify in a real browser.
- **Phase 3 — Live push + logs (optional):** `admin:services` Socket.IO event + post-action
  re-emit; widen logs allowlist for per-service log tails.
- **Separate/parallel task (not this feature):** register `muse_glimmer_30b` in
  `agent_config.json` (inactive) and run an **isolated arch load-test**. See the dedicated
  section below — Muse differs materially from the gemma-4 models and is **not** the same
  MTP risk class.

## Muse-Glimmer-30B — integration notes (differs from gemma-4)

Source: unsloth docs (fetched 2026-08-12) + local gguf metadata (verified this session).
Meta Superintelligence Labs, 30B **dense** vision model, Apache 2.0. This is its **own arch
`muse-glimmer`** (52 blocks, 131K native ctx up to 262K) — not a gemma-4 sibling.

Differences that change the config vs. the gemma-4 entries:

- **No MTP / no speculative decoding.** There is no draft file for Muse and the docs show no
  `--spec-*` flags. The config entry has **none** of the `spec-type` / `spec-draft-model`
  wiring the gemma-4-12b entry carries. Simpler, but also no MTP speedup.
- **Reasoning is effort *levels*, not a boolean.** Docs expose `low / medium / high / xhigh`
  reasoning efforts. The gguf chat template (7167 chars) has **`has<think>: False`,
  `enable_thinking: False`** — so the gemma-style `reasoning on/off` + `chat_template_kwargs
  {"enable_thinking":bool}` mechanism does **not** apply. OPEN ITEM: read the template and
  determine how effort is passed (a system directive vs. a `chat_template_kwargs` field such
  as `{"reasoning_effort":"high"}`); confirm live. This also affects how the admin
  UI / SDK would offer a thinking toggle for this model.
- **Recommended mmproj = `mmproj-Muse-Glimmer-30B-BF16.gguf` (3.8 GB).** We also have a
  `Q8_0` (2.0 GB) VRAM-saving fallback to A/B. mmproj arch `clip`, 50 vision blocks.
- **Sampling:** temp 1.0, top_p 0.95, top_k 64 (Meta defaults; no min_p specified).
- **Plain llama-server invocation** (unsloth guide): `--model <UD-Q4_K_XL> --mmproj <BF16>
  --temp 1.0 --top-p 0.95 --top-k 64 --alias ... --port ...` — **no `--jinja`, no reasoning
  flag, no `--spec-*`, no flash-attn flag**. The chat template is baked into the gguf (7167 ch)
  and used automatically. Our adapter preset must therefore emit a *minimal* options block for
  this model (unlike the gemma-4 entries): model + mmproj + sampling + `c`, nothing else.
- **Context:** the guide says "no need to set context length — llama.cpp uses the exact amount
  required." That holds for single-model CLI; in **our router** we still set `c` because
  `--parallel 2` sizes per-slot KV from it. So we set `c` explicitly (capped for VRAM), and it
  is the one place our preset diverges from the guide's flagless command.
- **VRAM reality on the 24 GB 4090:** 15.9 GB weights + 3.8 GB BF16 mmproj = ~19.7 GB before
  KV — so Muse must run essentially **alone** (this is why the operator's scenario 4 stops the
  other GPU containers first). Expect to **cap context** (e.g. 32–64K, not 131K) and possibly
  use **quantized KV** and/or the **Q8 mmproj** to fit. Exact fitting ctx is TBD by test.
- **Arch support is UNVERIFIED.** `muse-glimmer` has never been loaded in our llama.cpp
  (b10335). It could fail to load outright (unknown arch) the way an unsupported arch would.
  This is the gating risk and must be an **isolated load-test first**, run in a window where
  the GPU is freed (currently only ~3.5 GB free with the 12B resident — cannot test now).

Registration/test steps when a VRAM window is available:
1. Isolated `docker run` of the pinned llama.cpp image with a single-model preset for Muse
   (`model` + `mmproj-BF16`, temp/top_p/top_k, `c` capped, no spec flags) on a spare port —
   confirm it **loads** (arch recognized) and returns a coherent completion + a vision answer.
2. Determine the reasoning-effort mechanism from the template and verify a level takes effect.
3. Add the inactive `muse_glimmer_30b` entry to `agent_config.json` mirroring the fitted
   settings; only then is it selectable via the existing active-model switch.

## Known limitations to state plainly

- Start/stop manages **existing** containers only; a container removed via `compose down` needs
  a one-time `docker compose up` (agent_server has no compose files mounted and cannot run
  compose across the sibling stacks' directories).
- No per-container VRAM number on WSL2 — aggregate GPU used/free only.
- `stt_server` v1 ↔ v2 are mutually exclusive on port 2700; the panel manages v1 by default.
- Stopping `llama-vision` or `embeddings-server` has cross-service impact (noted/cv) — guarded
  by confirm dialogs, not prevented.
