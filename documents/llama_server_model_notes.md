# llama-server model notes (per-model hosting parameters)

> **START HERE for any LLM change/update** — model activation, MTP, swa-full,
> slots, llama.cpp updates, and how to test.

## CURRENT STATE — 2026-06-24 (read first)

- **ADAPTER mode is LIVE.** `llama-vision` runs from **`docker-compose.adapter.yml`**
  (built from `Dockerfile.llama-adapter`). On boot, `adapter/entrypoint.sh` runs
  `adapter/llama_cpp_preset.py` to **generate the llama-server preset from
  `data/config/agent_config.json`** inside the container — so `agent_config.json` is the ONE
  file to edit. No host `llama-router-models.ini` is used anymore (the old one is dead).
  Non-adapter `docker-compose.yml` is the **rollback** path only.
- **Active chat model:** `gemma-4-e2b` (Gemma 4 **E2B**, vision via E2B mmproj).
- **Resident (`--models-max 4`):** gemma-4-e2b, bge-m3, bge-reranker; ma2-360m-dpo-b01 (job2cool DPO) is
  `"resident": "on-demand"` since 2026-10-18 (1 request in 7 days): loaded on its first request, then kept. VRAM ~22.1 GB used / ~2.0 GB free on the 24 GB GPU.
- **MTP SHIPPED:** `spec-type=draft-mtp` + E2B draft head
  `gemma4_e2b/mtp-gemma-4-E2B-it.gguf`; requires `flash-attn=off`.
- **`swa-full=true`** (full SWA KV, PR #13194) — fixes Gemma wedging under
  context-reuse + speculative decoding. Needs **`c=65536`** (not 131072) to fit.
- **`--parallel 2`** (2 slots, not the default 4): set in the adapter `command:`.
  With swa-full each slot's KV is large, so 2 slots saved ~0.94 GB and is faster
  single-user (~230 t/s). 2 is plenty (low concurrency; voice-injection off).
  The chat KV (= slots × c) is the dominant VRAM cost; weights are only ~4.6 GB.
- **`--cache-ram 0`** (prompt cache off).
- **llama.cpp b11371** (commit 99b95488c) `sha256:f90b9de8ba…` since 2026-10-03 — pinned in BOTH
  `docker-compose.yml` AND `Dockerfile.llama-adapter` (keep equal). Rollback: b10868 `sha256:93e3b8da7c…`
  (that base is still pulled locally; no rollback-tagged adapter image exists - rebuild the adapter with
  `--build-arg LLAMA_CPP_BASE=ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:93e3b8da…`).
- **`FORCE_VOICE_INJECTION=false`** (agent_server): off by default. It only adds a
  SECOND llama-server call to synthesize a `<voice>` (TTS) block when the model omits
  one — extra GPU load, only for voice/avatar apps; not STT/multimodal.

### Runbook
- **Switch chat model:** set `"active": true` on the entry in `agent_config.json`
  (`models.chat`, exactly one), restart **llama-vision + agent_server** (or
  `POST /admin/api/active-model`, which does it via the mounted docker.sock). The
  adapter regenerates the preset on boot — never hand-edit an `.ini`.
- **Update llama.cpp:** `docker pull …:server-cuda`, take the digest, set it in BOTH
  `docker-compose.yml` and `Dockerfile.llama-adapter`, then
  `docker compose -f docker-compose.adapter.yml --profile default up -d --build`.
- **Apply a preset/config edit:** edit `agent_config.json`, then
  `docker compose -f docker-compose.adapter.yml --profile default up -d --force-recreate llama-vision`.
  Preview what will generate: `python3 adapter/llama_cpp_preset.py --print`.
- **Test:** ONE sequential request (`model: gemma-4-e2b`); expect 200 + good t/s; logs
  should show `using full-size SWA cache` + `speculative decoding context initialized`
  + (under use) `draft acceptance`. **NEVER run a concurrency/soak** — it queues a
  backlog that keeps generating server-side and pegs the GPU. Check no other caller is
  loading the LLM first: `docker logs agent_server | grep MODEL_REQ` (jobunter/cv/noted…).
- **Rollback to non-adapter:** `docker compose -f docker-compose.adapter.yml --profile
  default down && docker compose -f docker-compose.yml --profile default up -d`.

---

`agent_server` runs in **forwarding mode**: it does not host the model
itself. The model lives in the `llama-vision` container, which runs
llama.cpp's `llama-server` in router mode. The preset is generated from
`agent_config.json` by the adapter (see CURRENT STATE above).

## Source of truth + adapter-generated preset

**`data/config/agent_config.json` is the single source of truth — the only file
anyone edits.** `models` is grouped by task (`chat`, `embedding`,
`reranking`); each entry is self-describing (neutral core +
`backends.<name>.options`). There is no `llama-router-models.ini` to
manage: the **llama.cpp adapter** generates the llama-server preset from
`agent_config.json` *inside the llama-vision container* on every boot.

- Adapter generator: `adapter/llama_cpp_preset.py` (pure stdlib; the
  backend-specific translation layer).
- Container entrypoint: `adapter/entrypoint.sh` (regenerates the preset to
  `/tmp`, then exec's the real `llama-server`).
- Image: `Dockerfile.llama-adapter` (the pinned llama.cpp digest + python3
  + the two files above). Wired by `docker-compose.adapter.yml`.

Anti-drift guarantee: the generated `[section]` header *is* the entry's
`model_id`, which is the exact name `agent_server` forwards to. The two
can no longer disagree (the old bug: config said "Gemma 4 E2B" while the
host served E4B).

Server-process flags (`--host`, `--port`, `--models-max`,
`--cache-reuse`) live in the compose command, not in `agent_config.json` —
they're singular for the whole process, not per model.

### Switching the chat model

One chat model is resident in VRAM at a time. To switch — **one file, two
restarts:**

1. In `data/config/agent_config.json` under `models.chat`, set `"active": true`
   on the desired entry (exactly one active chat model).
2. Restart **llama-vision** (the adapter regenerates the preset and the
   GPU owner reloads the new model), then **agent_server** (re-reads its
   active entry): `docker restart llama-vision && docker restart agent_server`.

(For local sanity checks without a container, `python3
adapter/llama_cpp_preset.py --print` shows the preset that would be
generated, and `--check` flags drift against a written copy.)

## Per-model parameters

All three production models are **reasoning models** — each needs the
reasoning/thinking channel enabled at the llama-server layer, and
`agent_server`'s `_ThinkingSplice` folds the resulting
`reasoning_content` back into `<think>...</think>` for the noted UI.

| Model | family | model_id | vision | reasoning flag |
|---|---|---|---|---|
| Gemma 4 E4B IT Q4_K_XL | `gemma` | `gemma-4` | yes (mmproj) | `reasoning = on` + official Jinja template |
| Qwen3.5 4B Q5_K_XL | `qwen` | `qwen3.5` | no | `reasoning = on` + `enable_thinking` |
| Phi-4-mini-reasoning Q6_K_XL | `phi` | `phi-4-mini-reasoning` | no | `reasoning = on` |

### Reasoning / thinking enablement

- **`reasoning = on`** is the supported, non-deprecated llama-server flag.
  It injects the template's thinking tokens AND tells the reasoning
  extractor to populate `delta.reasoning_content`. `reasoning = auto`
  (the default) only does so if the template self-declares thinking.
- **Gemma 4**'s template does not self-declare thinking, so `on` is
  required. The official template (vendored at
  `data/models/chat_template_gemma-4.jinja`) also correctly replays
  `reasoning_content` into multi-turn tool-call history.
- **Qwen3.5** gates thinking behind the template's `enable_thinking`
  switch. `reasoning = on` activates it; we also set
  `chat-template-kwargs = {"enable_thinking": true}` explicitly so the
  thinking flag is unambiguous in config. If thinking ever fails to
  appear, that explicit kwarg is the lever to check first.
- **Phi-4-mini-reasoning** is reasoning-tuned; `reasoning = on` surfaces
  its thinking channel as `reasoning_content`.

### family-conditional post-processing in agent_server

`family` (set per entry in `agent_config.json`) gates the
Gemma-specific handling in `app/llm_engine_server.py` /
`app/openai_compat.py`:

- **Applied to ALL families** (reasoning-model behaviour):
  - `_ThinkingSplice` — `reasoning_content` → `<think>...</think>`.
  - `_strip_history_thinking` — drop prior-turn `<think>` blocks before
    re-feeding history (correct for every reasoning model).
- **`gemma` only**:
  - `_expand_tool_call_arguments` — converts `tool_calls[].arguments`
    from JSON string to dict for Gemma's pipe-marker chat template.
    Qwen/Phi use ChatML/standard tool rendering and expect the OpenAI
    string form, so this is skipped for them.
  - `<eos>` literal-text stop injection (`openai_compat._merge_request_params`)
    — Gemma emits a literal `<eos>` after tool calls; Qwen/Phi stop on
    their own EOS tokens.

---

## Preserved Gemma lore (was inline in llama-router-models.ini)

These hard-won operational notes used to live as comments in
`llama-router-models.ini`. They are preserved here now that the file is
generated. They are specific to the `gemma-4` entry unless noted.

### Multi-Token Prediction (MTP) — tested, not shipped

Multi-Token Prediction via the AtomicBot-ai turboquant fork was tested
live 2026-05-10. Measured 170 tok/s with ~50% draft accept rate on Gemma
4 E4B Q4_K_XL — within the fork README's +30-50% band. **NOT enabled in
production** because the fork auto-disables MTP when `mmproj` is loaded
(silent log line "speculative decoding is not supported by multimodal")
and we need vision. To re-enable on the fork image: comment `mmproj`,
override `flash-attn = off` (the fork's MTP CUDA decode crashes in
fattn.cu:109 when fa is on), add the four MTP lines below, and switch the
image in `docker-compose.yml`. The Q4_K_M assistant GGUF lives at
`data/models/gemma-4-E4B-it-assistant.Q4_K_M.gguf`. See
<https://ai.google.dev/gemma/docs/mtp/mtp> and
AtomicChat/gemma-4-E4B-it-assistant-GGUF.

```
spec-type = mtp
mtp-head = /agent_server_models/gemma-4-E4B-it-assistant.Q4_K_M.gguf
draft-block-size = 3
n-gpu-layers-draft = -1
```

### n-gram speculative decoding — tested, no benefit (2026-05-20)

Tried `spec-type = ngram-cache` (stock-b9246 prompt-lookup spec decoding:
drafts from context n-grams, no draft model, no GGUF). Measured
before/after on llama-vision, temp-0, 3 probes x3 runs: baseline 165.2
tok/s → ngram-cache 164.6 tok/s (flat / -0.4%). Why no gain: the drafter
fired on only ~35 of ~3900 decode steps (~1%) and accepted just 17% of
the tokens it proposed — only ~0.9% of output came free, swamped by
verify-batch overhead. Gemma's reasoning block is novel text with ~0
n-gram hits. Do not re-enable for this workload. (n_draft defaulted to 8
for ngram-cache.)

### context checkpoints DISABLED (2026-05-20) — `ctx-checkpoints = 0`

llama.cpp b9246's context-checkpoint feature crashes the gemma-4 worker:
restoring a saved checkpoint for a long SWA prompt aborts with
ggml_abort at common.cpp:2093 ("checkpoint size mismatch: expected <N>,
got 0") → SIGABRT → worker becomes a zombie → every chat returns HTTP
500. See `project_llama_vision_worker_instability.md`. Setting max
context checkpoints per slot to 0 means none are created, so the crashing
restore path is never entered. Cost: long SWA prompts re-prefill fully
(already happening). Re-enable only when the upstream restore bug is
fixed.

### KV cache: F16 at c=131072 — do NOT quantize (tombstone)

`cache-type-k = q8_0` was tried twice and is a trap:

- 2026-05-15 first: BOTH K and V to q8_0. Worker died mid-run (not OOM —
  VRAM had headroom). flash-attn + V-cache quantization is known-flaky in
  llama.cpp, especially with mmproj loaded.
- 2026-05-15 second: K-only quantization. ALSO breaks the chat path under
  realistic load (16k+ char system prompt + 62 tool definitions +
  streaming): prefill takes long enough that the upstream SSE read times
  out, agent_server cancels, llama-vision returns 500 "Connection
  handling canceled", the browser sees an empty assistant bubble.
  Validated via tests/chat_smoke_probe.py: with `cache-type-k = q8_0` the
  assistant message never appears; without, it appears at t+4.3s.

With the proper router compose, F16 KV at c=131072 fits in ~21 GB with
~3 GB margin. **No quantization needed.** If a real OOM ever returns,
drop `c` to 65536 (still long-context) before touching `cache-type`.

### slots / parallelism

No `parallel` setting → llama-server defaults to auto (4 slots), which
also default-enables `--kv-unified`. Experiments 2026-05-04 showed
`parallel = 2 + kv-unified = on` saved only ~200 MB vs auto, not worth
giving up 2 concurrent slots (needed for primary stream + voice-injection
secondary call + future multi-agent / multi-user). See
`project_llama_vision_vram_tuning.md`.

### build pinning

The `llama.cpp` server image is digest-pinned in `docker-compose.yml`
(and, for the adapter cutover, in `Dockerfile.llama-adapter`'s
`LLAMA_CPP_BASE` ARG default) with soak-test history (current **b9487**
`sha256:8c44d3ca…`, bumped 2026-06-03; rollback candidates b9309
`0814ef45…` and b9246 `4e13f877…`). Note `b9488` had no `server-cuda`
image as of the bump. Read those comments before bumping the image.

### flash attention ON for gemma-4 (E4B) with MTP and the vision projector (2026-11-12)

`flash-attn` was `off` for the two E models since the MTP decode crashed in `fattn.cu` (above; upstream llama.cpp
#24376: the E4B drafter's attention layer had no flash-attention kernel). Upstream closed it on 2026-06-30 as fixed by
PR #25148; this server's build (b11870, since 2026-10-25) has the fix. On 2026-11-12, with the owner's go-ahead,
`agent_config.json` was changed for **gemma-4 (E4B) only** - `"flash-attn": "on"` - and llama-vision + agent_server
restarted. The E2B entry was NOT changed (not tested). Backup of the config before the change:
`data/config/agent_config.json.before-flash-attn-20261112` (in the container's data folder). To go back: set it to
`off` and `docker restart llama-vision agent_server`.

Measured before (off) and after (on), same requests, model free, everything on the GPU, c=65536, 2 slots, MTP draft
(`spec-draft-n-max 4`), mmproj loaded:

| | off | on |
|---|---|---|
| Text, 182 tokens out | 441 tok/s | 458 tok/s |
| Picture + text, ~512 tokens out | 339 tok/s | 360 tok/s |
| Prompt reading (warm) | about 2,100-2,500 tok/s | about 2,500-3,200 tok/s |
| MTP draft accepted (text / picture) | 138 of 172 / 352 of 636 | 138 of 172 / 355 of 632 |
| Meeting Assistant tool decision (Cortex `eval/notes/follow_decision.py`, 12 runs a language) | 12/12 EN, 12/12 PT | 12/12 EN, 12/12 PT |
| A checker workload (66 statements, 5 requests with reasoning) | 20 of 21 false caught, 23.4 s | 20 of 21, 23.5 s |
| VRAM after load | 19.6 GB | 19.5 GB |

After the change, also: 12 concurrent streaming requests through agent_server in 3 rounds (two prompts of about
15,000 tokens and two pictures at a time) all completed; Cortex's Knowledge Base Assistant answered in the page
(tests/ui/assistant_thinking.mjs, EN and PT, 0 failed); bge-m3 still loads; no `fattn` / fatal / abort line in the
log. The gain is small (about 4-6% on writing): the reason to keep it on is that it is no longer a forced exception.
NOT covered: a long soak, the E2B model, voice. KV-cache quantisation stays ruled out (tombstone above) - flash
attention being on does not change that record.

### Candidate upgrade: llama.cpp v0.9.0 (assessed 2026-11-12, NOT tested, NOT installed)

Running: 0.7.0-dev b11870 (commit 74bca3950, 2026-10-25). Latest tagged release: v0.9.0 (2026-11-04), 240 commits
later; master is 208 more (daily prerelease builds, b12318 on 2026-11-12). Read from the commit titles and two PRs:

- #31356 "gemma4: graph optimizations (massive gains on pp for E2B and E4B)", in v0.9.0: prompt reading 1.6x (E4B) to
  2.9x (E2B) in the author's own figures (Apple, CPU, Snapdragon - no CUDA figure), writing speed unchanged. As merged
  it applies ONLY with one sequence (`n_seq_max == 1`) and a unified KV: this server runs `--parallel 2`, so the gain
  may not apply here without changing that - to be measured, not assumed.
- #30141 "spec : save and restore MTP hidden state", #31350 "restore the spec checkpoint for drafts longer than
  n_rs_seq", in v0.9.0: MTP with context reuse. May bear on `ctx-checkpoints = 0` (kept at 0 for an older crash).
- #30922 "mtmd : resize Gemma 4 images without letterbox padding", in v0.9.0: the picture the model sees changes -
  vision answers will differ; every vision prompt needs a look after an upgrade.
- #31635 (flash attention followed by the Gemma 4 assistant): only matters with a quantised V cache or a separate
  draft device - not this configuration.
- After v0.9.0, on master only: tool-call argument fixes (#31937 control characters, #31861 invalid UTF-8, #31956
  grammar with response_format and tools), #31927 MTP hidden states read from a device buffer, #31965 Gemma 4 audio.

Not urgent: nothing here fixes a fault this server has today. Worth a planned A/B (same requests before and after:
speed, tool decisions, vision, a Cortex chat), on the tagged release, with the present digest as the way back.
Whether a `server-cuda` image exists for v0.9.0 was not checked.

### Upgraded to llama.cpp v0.9.0 = b12110 (2026-11-12, the owner's go-ahead) - supersedes "Candidate upgrade" above

- Image: registry tag `server-cuda-b12110`, digest `sha256:a2963b4817f03eea7e0d6065ea576252322792441ef97aaae178b924520547d6`
  (`llama-server --version`: 0.9.0-dev, build 12110, commit cd959bae9). Pinned in `Dockerfile.llama-adapter` and
  `docker-compose.yml`. There is no `server-cuda-v0.9.0` tag: v0.9.0 is 240 commits after b11870, i.e. b12110.
- Way back: the previous adapter image is kept as `agent_server-llama-adapter:b11870-rollback` (digest
  `sha256:334ddc35...` in both files' comments): put that digest back, `docker compose build llama-vision`,
  `docker compose up -d llama-vision`, `docker restart agent_server`.
- Same configuration as before (gemma-4 E4B, c=65536, 2 slots, flash-attn on, MTP draft, mmproj).

Before (b11870) and after (b12110), same requests, model free (Cortex `documentation/code-analysis/model_server_bench.py`,
results in its `out/flash/`):

| | b11870 | b12110 |
|---|---|---|
| A cold 13,400-token prompt, read | 11,700-13,500 tok/s | 15,200-15,300 tok/s |
| Writing, text | 452 tok/s | 449 tok/s |
| Writing, picture request | 357 tok/s (514 tokens) | 337 tok/s (610 tokens: a longer answer) |
| Two cold long prompts at once | 2.9-3.0 s each | 2.7-3.1 s each (one first pair took 6.1 s, not reproduced after two restarts) |
| Tool call (`get_weather`), 10 runs | 10/10 | 10/10 |
| Answer forced to a JSON schema | right | right |
| bge-m3 embedding of one sentence | - | cosine 1.000000 with the old build's |
| Meeting Assistant tool decision (Cortex eval, 12 a language) | 12/12, 12/12 | 12/12, 12/12 |
| A checker workload (66 statements with reasoning) | 20 of 21 false caught, 23.5 s | 20 of 21, 22.7 s |
| 12 concurrent streaming requests (long text + pictures) | all completed | all completed |
| Cortex Knowledge Base Assistant in the page (EN, PT) | 0 failed | 0 failed |

Pictures (the release resizes Gemma 4 pictures without letterbox padding, #30922): four pictures with known content
asked to both builds (the old one as a small side container). Tab titles, a chat answer and a photograph's words read
right by both; the old build read the logo "bancoctt" as "bancocit", the new one right. Asked for "anything else" in a
plain picture, each invents a detail (old: "thin black lines", new: "a small glyph at the bottom right"). Cortex's own
vision uses (decks critic, movie review) were NOT re-measured: worth a look the next time they run.
The side container took the card to 23.4 GB for two minutes (the paging range): do not repeat it with a job running.

### MTP draft length: 4 stays (measured 2026-11-12 on b12110)

`spec-draft-n-max` 2, 3, 4, 6, 8, a restart each; median tokens a second of 3 runs (reasoning on):

| n-max | list + sentence | prose (PT) | code | question about a file |
|---|---|---|---|---|
| 2 | 369 | 284 | 332 | 298 |
| 3 | 410 | 283 | 363 | 296 |
| **4** | **452** | **276** | **372** | **297** |
| 6 | 464 | 251 | 370 | 282 |
| 8 | 414 | 231 | 341 | 241 |

Drafted tokens accepted at 4: 80% on a list, 62% on code, 47% on a file question, 36% on prose. Longer drafts waste
more on prose; shorter ones lose on lists and code. Left at 4.

### CUDA 13 tried (first note: CUDA 12 kept - superseded the same day by the section after it)

The host driver (621.14) supports CUDA 13.5; the pinned image is built on CUDA 12.8.1. The same build exists on
CUDA 13.4.1 (`server-cuda13-b12110`, digest `sha256:27907c3a...`, kept locally as
`agent_server-llama-adapter:b12110-cuda13`). Measured on it, same requests: cold 13,400-token prompt 15,200-15,300
tok/s (12.8: the same), text 446-460 tok/s (449), prose 274 (276), code 376 (372), file question 295 (297), tool call
10/10. No gain: left on CUDA 12.8 (`agent_server-llama-adapter:b12110-cuda12` = `:1.0`, the pinned digest).
The intermittent slow pair of long prompts (open-issues-agent_server.md) happens on both, and on b11870.

### CUDA 13.4.1 adopted (2026-11-12, the owner: "why not upgrade? ... we should")

No measured reason against it (equal speed), so the pinned base is now `server-cuda13-b12110`
(`sha256:27907c3a40a7ba377cc4374c00c626bd65aea0c60e2afc4cdaae75ca11b58106`) in `Dockerfile.llama-adapter` and
`docker-compose.yml`; the CUDA 12.8.1 image of the same build stays locally as
`agent_server-llama-adapter:b12110-cuda12` (digest in both files' comments). The release's CUDA 13 builds went to 13.5
only after v0.9.0 (#31788), so this tagged build is on 13.4.1.
Verified on it: cold 13,400-token prompt 15,200-15,300 tok/s, text 458 tok/s, picture 338, tool call 10/10, forced JSON
right, the Meeting Assistant's tool decision 12/12 in both languages, the checker 20 of 21, the four pictures read as
on CUDA 12 (logo "bancoctt" right), 12 concurrent streaming requests completed, Cortex's Knowledge Base Assistant in
the page 0 failed, no crash line in the log.

### Read every figure of 2026-11-12 above with this (found the same evening)

The WSL clock runs at 90% of real time (open-issues-agent_server.md, HOST entry). The model server times itself with
that clock, so the tokens-a-second figures above are about 11% higher than real (452 tok/s is about 407 in real
time); the comparisons between builds, CUDA versions, flash attention and draft lengths were made with the same
clock and stand. The "one slow pair of long prompts" (6.1-6.6 s) was a wall-clock correction of 3.6 s falling inside
the measurement, not a stall.
