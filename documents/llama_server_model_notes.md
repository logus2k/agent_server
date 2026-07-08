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
- **Resident (4, `--models-max 4`):** gemma-4-e2b, ma2-360m-dpo-b01 (job2cool DPO),
  bge-m3, bge-reranker. VRAM ~22.1 GB used / ~2.0 GB free on the 24 GB GPU.
- **MTP SHIPPED:** `spec-type=draft-mtp` + E2B draft head
  `gemma4_e2b/mtp-gemma-4-E2B-it.gguf`; requires `flash-attn=off`.
- **`swa-full=true`** (full SWA KV, PR #13194) — fixes Gemma wedging under
  context-reuse + speculative decoding. Needs **`c=65536`** (not 131072) to fit.
- **`--parallel 2`** (2 slots, not the default 4): set in the adapter `command:`.
  With swa-full each slot's KV is large, so 2 slots saved ~0.94 GB and is faster
  single-user (~230 t/s). 2 is plenty (low concurrency; voice-injection off).
  The chat KV (= slots × c) is the dominant VRAM cost; weights are only ~4.6 GB.
- **`--cache-ram 0`** (prompt cache off).
- **llama.cpp b9776** `sha256:0a8757369e…` — pinned in BOTH `docker-compose.yml`
  AND `Dockerfile.llama-adapter` (keep equal). Rollback: b9717 `sha256:7f3949110c…`.
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
