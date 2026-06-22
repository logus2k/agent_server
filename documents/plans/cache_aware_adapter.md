# Cache-Aware Routing Adapter — Architecture & Implementation Plan

> **Status:** PROPOSED (not built). Plan only.
> **Branch grounding:** `v2` (working tree, this repo). All "current state"
> claims below are read from v2 source and cite the file/line.
> **Author intent:** Let clients like **NousResearch Hermes** (which author
> prompts with Anthropic-style `cache_control` markers) *and* existing
> agent_server clients (noted, CV, idealab, …) call agent_server
> **transparently**, while each request extracts the maximum benefit from
> **both** caching mechanisms it may touch: **Anthropic prompt caching**
> (when routed to the Claude backend) and **llama.cpp KV / prefix caching**
> (when routed to the local `llama-vision` backend).

---

## 0. TL;DR

agent_server becomes a **cache-aware normalization layer**. A client authors
one request — optionally carrying Anthropic `cache_control` breakpoints — and
agent_server translates that single intent into whatever the *selected*
backend actually needs:

- **→ llama.cpp (local):** strip `cache_control` (per content-block, never
  flattening away vision/tool blocks), and **guarantee a byte-stable prompt
  prefix** so llama-server's `--cache-reuse` actually hits. This requires
  fixing a current prefix-buster (a minute-resolution timestamp injected into
  every system prompt) and adding optional slot affinity.
- **→ Anthropic (Claude):** convert OpenAI-shaped messages into the Anthropic
  *Messages* format and **place/preserve `cache_control` breakpoints** at the
  stable, high-value boundaries (system, tools, long context), honoring the
  4-breakpoint and 5-minute-TTL constraints.

The work is split so the **llama.cpp caching revision ships first and on its
own merits** (faster local turns, measurable cache-hit rate), and the
**Anthropic egress + unified `cache_control`** lands second as a new backend in
the router. Every phase is feature-flagged and zero-regression by default.

---

## 1. Goals / Non-Goals

### Goals
1. **Transparency.** Hermes sends Anthropic-style requests (with `cache_control`)
   to the *same* `/v1/chat/completions` endpoint existing clients use; existing
   clients keep sending exactly what they send today and see no behavior change.
2. **Dual cache exploitation.** A request routed local maximizes llama.cpp
   prefix-cache reuse; a request routed to Claude maximizes Anthropic prompt
   caching. One authored intent, two backends.
3. **Full revision of the current llama.cpp caching integration**, with
   before/after measurement (cache-hit rate, prefill tokens, TTFT, tok/s).
4. **Single source of caching policy** lives in agent_server, not scattered
   across clients — consistent with the existing model-agnostic principle.

### Non-Goals
- Replacing llama.cpp's internal KV cache implementation (we tune/feed it, not
  reimplement it).
- Persisting Anthropic cache across the 5-minute TTL (out of scope; we optimize
  *within* provider semantics).
- Training/altering chat templates (we change message *assembly*, not the Jinja
  templates).
- A general multi-provider gateway (OpenAI, Gemini, …). Scope is **local
  llama.cpp + Anthropic** only. The router is built so a third backend is
  additive, but that is future work.

---

## 2. Current State (v2) — grounded audit

### 2.1 Ingress: two paths, one assembly contract
agent_server exposes two client surfaces, both of which assemble a `messages`
array and forward to `llama-vision`:

- **OpenAI REST** — `POST /v1/chat/completions`
  ([app/openai_compat.py](../../app/openai_compat.py)). `_resolve_model()`
  maps the `model` field to either an **agent preset** or a **model_id**;
  `_build_messages()` assembles the prompt; `_merge_request_params()` resolves
  sampling. This is how noted, CV, idealab call in.
- **Socket.IO** — `Chat` event → `LlamaServerEngine.generate_stream()`
  ([app/llm_engine_server.py:781](../../app/llm_engine_server.py#L781)).

Both ultimately POST to llama-server's own `/v1/chat/completions`.

### 2.2 Egress: only llama.cpp today
The forwarding engine is `_LlamaServerProxy`
([app/llm_engine_server.py:268](../../app/llm_engine_server.py#L268)). Notably:

```python
def set_cache(self, *_args, **_kwargs) -> None:
    # No-op: llama-server manages its own prompt cache and slot reuse.
    return None
```

agent_server delegates **all** caching to llama-server. **There is no Anthropic
egress anywhere in agent_server** — the only mention is a comment
([app/main.py:235](../../app/main.py#L235)) noting that `/v1/agents/{name}`
exists so an external caller can fetch a preset for "the Claude cross-backend
path." That path lives in **noted** today (`noted/backend/app/workflow/
llm_dispatcher.py::dispatch_claude()`, and `llm_router.health()` which already
merges local + Anthropic models for noted's dropdown — see
[active_model_switching_sdk.md](../active_model_switching_sdk.md)). So
"add Anthropic to agent_server" is **net-new capability**, not a tweak — and it
largely **consolidates logic noted already has**.

### 2.3 llama.cpp cache configuration (live)
From [docker-compose.adapter.yml](../../docker-compose.adapter.yml) (the
deployed stack — `llama-vision` runs `agent_server-llama-adapter:1.0`):

```
--models-max 4  --host 0.0.0.0  --port 8500  --cache-reuse 256  --cache-ram 2048
```

- `--cache-reuse 256` — llama-server may reuse a cached prefix and re-prefill
  only the diverging tail when the divergence is within 256 tokens. This is the
  primary lever that makes prefix caching tolerant of small mid-prompt edits.
- `--cache-ram 2048` — host-RAM cache budget (MB) for slots' KV offload.
- **No `--parallel`** → llama-server auto-selects (4 slots) and auto-enables
  `--kv-unified` (per [llama_server_model_notes.md](../llama_server_model_notes.md)
  §slots). 4 slots are intentionally kept for primary stream + voice-injection
  secondary call + concurrency.
- `ctx-checkpoints = 0` (per-model option, [agent_config.json](../../data/agent_config.json))
  — context checkpoints are **disabled** because the restore path crashes the
  gemma-4 worker (model_notes §"context checkpoints DISABLED"). Consequence:
  **a prompt that diverges beyond `--cache-reuse` re-prefills fully.**
- **KV cache is F16, do not quantize** (model_notes "tombstone": q8_0 broke the
  chat path twice). Active model `gemma-4` runs `c = 65536`.

**Net:** local prefix-cache benefit today rests entirely on `--cache-reuse 256`
+ llama-server's automatic longest-prefix slot matching. There is **no slot
affinity** from agent_server — a conversation is not pinned to a slot, so under
concurrency two callers can evict each other's prefix.

### 2.4 Prefix-mutation hazards (why local cache hits are lower than they look)
llama-server's prefix cache only hits on a **byte-identical token prefix**.
Several existing transforms perturb the prompt **before** it reaches the cache:

1. **🔴 Minute-resolution timestamp injected into EVERY system prompt** —
   [app/openai_compat.py:261-266](../../app/openai_compat.py#L261-L266):
   ```python
   now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
   preamble = f"Today's UTC date and time: {now_utc}."
   sys_text = f"{preamble}\n\n{sys_text}"
   ```
   The system prompt is the **longest, most stable, highest-value** cache
   prefix — and this prepends content that **changes every minute**, at
   position 0. Result: the entire system prefix cache-misses on the first
   request of every new minute (and never matches across conversations).
   This is the single biggest local-cache regression in the codebase, and it
   is *also* exactly what would poison an Anthropic `cache_control` breakpoint
   placed after the system block. **The REST path does this; the Socket.IO
   path does not** (`generate_stream` reads the prompt file verbatim,
   [llm_engine_server.py:793-799](../../app/llm_engine_server.py#L793)).
2. **History thinking strip** — `_strip_history_thinking()`
   ([llm_engine_server.py:154](../../app/llm_engine_server.py#L154)) rewrites
   prior assistant turns (removes `<think>…</think>`). Deterministic, and it
   affects the *suffix* (mid/late prompt), not the system prefix — lower
   impact, but it does mean turn-N content isn't byte-identical to what was
   generated.
3. **Gemma tool-arg expansion** — `_expand_tool_call_arguments()`
   ([llm_engine_server.py:103](../../app/llm_engine_server.py#L103)) rewrites
   `tool_calls[].function.arguments` from string→dict for the Gemma template.
   Deterministic; only affects tool-calling turns.
4. **Gemma `<eos>` stop injection** — `_merge_request_params()`
   ([openai_compat.py:170](../../app/openai_compat.py#L170)). Sampling-only,
   does not change the prompt prefix.

Items 2–4 are *correct and necessary*; the plan keeps them but makes their
cache impact explicit and measured. Item 1 is the one to fix.

### 2.5 Multi-resident & routing
The adapter declares the active chat model + any `resident:true` models +
embedders/reranker ([adapter/llama_cpp_preset.py](../../adapter/llama_cpp_preset.py)),
all sharing the 4 slots. `_LlamaServerProxy` sets the `model` field so
router-mode llama-server selects the right resident model
([llm_engine_server.py:328](../../app/llm_engine_server.py#L328)). Relevant
because **slot affinity and cache reuse interact with model residency**: a
request for a different resident model lands on a different model's slots.

---

## 3. Design Principles

1. **Zero-regression by default.** Every new behavior is behind a flag (env or
   per-request) and OFF until proven, matching the repo's existing pattern
   (`FORCE_VOICE_INJECTION`, `LLAMA_SERVER_URL`).
2. **One canonical internal message model.** Ingress (OpenAI-shaped or
   Anthropic-shaped) is parsed once into a neutral structure that *carries
   cache intent as data*; each backend renderer consumes it. Mirrors the
   existing "neutral core + per-backend block" philosophy of
   `agent_config.json`.
3. **Never lose content.** Normalization for llama.cpp strips only the
   `cache_control` key, **per block**, preserving text/image/tool blocks. (The
   naïve "flatten text blocks with `" ".join(...)`" approach discussed earlier
   would silently drop vision and tool content — explicitly rejected.)
4. **Cache intent is advisory, not load-bearing.** A request with no
   `cache_control` must work and cache *as well as today or better*; markers
   only *improve* placement.
5. **Measure both sides.** Cache-hit rate is a first-class, logged metric for
   local (llama-server timings/slots) and Anthropic (`usage.cache_*`).

---

## 4. Target Architecture

### 4.1 Component view

```
                       ┌──────────────────────────────────────────────────────┐
 Hermes (Anthropic-    │                 agent_server (:7701)                  │
 style + cache_control)│                                                       │
        │              │   ┌───────────────┐    ┌──────────────────────────┐  │
        ├──HTTP /v1───►│──►│ Ingress Parser │──►│  Canonical Request Model  │  │
 noted / CV / idealab  │   │ (OpenAI |      │    │  • messages (blocks)      │  │
 (OpenAI-style)        │   │  Anthropic     │    │  • cache spans (intent)   │  │
        │              │   │  shape detect) │    │  • system / tools         │  │
 Socket.IO Chat ──────►│   └───────────────┘    └────────────┬─────────────┘  │
                       │                                      │                 │
                       │                          ┌───────────▼───────────┐     │
                       │                          │   Backend Router      │     │
                       │                          │  (model_id / agent →  │     │
                       │                          │   local | anthropic)  │     │
                       │                          └─────┬───────────┬─────┘     │
                       │                                │           │           │
                       │        ┌───────────────────────▼──┐   ┌────▼────────────────────┐
                       │        │  Local Renderer (llama)   │   │ Anthropic Renderer      │
                       │        │  • strip cache_control    │   │ • OpenAI→Messages map   │
                       │        │    (per block)            │   │ • place/keep cache_     │
                       │        │  • stable-prefix assembly │   │   control breakpoints   │
                       │        │  • optional slot affinity │   │ • SSE event translate   │
                       │        └───────────┬───────────────┘   └────────────┬────────────┘
                       │                    │ HTTP (existing proxy)           │ HTTPS
                       └────────────────────┼─────────────────────────────────┼──────────┘
                                            ▼                                  ▼
                                  llama-vision (:8500)                 api.anthropic.com
                                  KV / prefix cache                    prompt caching
                                  (--cache-reuse 256)                  (cache_control TTL)
```

### 4.2 Canonical Request Model (new, internal)
A small dataclass set (proposed `app/cache_routing.py`) that both renderers
consume:

```python
@dataclass
class CacheSpan:
    # Marks a boundary AFTER which the running prefix is a cache candidate.
    # Sourced from an Anthropic cache_control marker, OR synthesized by us
    # (e.g. after the system block, after tools). 'kind' lets the Anthropic
    # renderer respect the 4-breakpoint budget by priority.
    index: int            # message/block index the marker sits on
    kind: str             # "system" | "tools" | "context" | "conversation"
    ttl: str = "5m"       # Anthropic ephemeral TTL hint

@dataclass
class CanonicalRequest:
    system: list[ContentBlock]      # system blocks (text/structured)
    tools: list[dict] | None
    messages: list[CanonicalMessage]  # role + content blocks (text/image/tool)
    cache_spans: list[CacheSpan]    # ordered, de-duplicated, priority-ranked
    sampling: dict                  # merged gen params (existing _merge logic)
    backend_hint: str | None        # explicit override if a client picks one
```

Ingress detection rule (cheap, unambiguous):
- If any message `content` is a list whose blocks carry `cache_control`, **or**
  the request has a top-level `system` array with `cache_control` → treat as
  **Anthropic-shaped** (Hermes). Capture markers into `cache_spans`.
- Otherwise → **OpenAI-shaped** (existing clients). `cache_spans` starts empty;
  the local renderer may *synthesize* a span after the system block.

This keeps existing clients on a byte-identical path when the feature flag is
off, and even when on, an OpenAI request with no markers produces the same
prompt (plus the prefix-stability fix).

### 4.3 Backend Router
Selection precedence (first match wins):
1. **Explicit per-request** `backend` field or `X-Agent-Backend` header
   (`"local"` | `"anthropic"`), if present.
2. **model_id namespace** — `claude-*` (or `anthropic/*`) → Anthropic; any
   local chat `model_id` / agent name → local. (Mirrors noted's existing
   "claude-* picks route to Anthropic" rule.)
3. **Default** — local (preserves today's behavior).

`GET /v1/models` extends to list Anthropic chat models with
`kind:"chat", backend:"anthropic"` so pickers (noted dropdown) can show both —
again consolidating what noted already merges client-side.

---

## 5. The llama.cpp Caching Revision (Phase 1 — ships first)

This phase is valuable **independently** of Anthropic work: it raises local
cache-hit rate and lowers TTFT for *every* existing client.

### 5.1 Fix the volatile system-prefix (highest impact)
**Problem:** §2.4(1) — a per-minute timestamp at prompt position 0.

**Options (pick per measurement, default = B):**

- **A. Drop the auto-preamble; keep only the explicit `{{today_utc}}`
  placeholder.** Prompts that need the date opt in; the placeholder can be
  positioned by the author *after* the stable instructions. Lowest complexity;
  changes behavior for prompts that relied on the implicit date.
- **B. Move volatile content to the END of the system block (or into a
  trailing system message).** Keep auto-date, but assemble as
  `"{stable_prompt}\n\n{volatile_preamble}"` instead of
  `"{volatile_preamble}\n\n{stable_prompt}"`. The long stable prefix now caches;
  only the short tail re-prefills (and `--cache-reuse 256` absorbs it). Keeps
  the date for every agent. **Recommended.**
- **C. Coarsen the timestamp** to date-only (`%Y-%m-%d`) or hour resolution.
  Reduces miss frequency from per-minute to per-day/hour. Can combine with B.

> Whichever option, the Socket.IO path
> ([llm_engine_server.py:793](../../app/llm_engine_server.py#L793)) must use the
> **same** assembly so both ingress paths share one cache prefix per agent.

### 5.2 cache_control as ordering hints (even for local)
When an Anthropic-shaped request arrives but routes **local**, use its
`cache_spans` to *order* assembly so the marked-stable content forms a
contiguous front prefix (system → tools → long context → conversation), then
strip the markers. We never reorder *conversation* messages (semantically
illegal) — only confirm that system/tools/static-context sit ahead of the
volatile turn tail, which is the natural order anyway. This makes Hermes'
authored intent *help* the local cache instead of being discarded.

### 5.3 Per-block `cache_control` stripping (never flatten)
Replace any flattening approach with a structure-preserving strip in
`_build_messages` / a new normalizer:

```python
def strip_cache_control(blocks):
    return [{k: v for k, v in b.items() if k != "cache_control"} for b in blocks]
```

Applied to system blocks and to any list-valued message `content`. Text remains
text, `image_url` blocks and tool blocks pass through untouched (preserving the
multimodal/tool support documented at
[openai_compat.py:58-83](../../app/openai_compat.py#L58)).

### 5.4 Optional slot affinity
Today nothing pins a conversation to a llama-server slot. Add an **opt-in**
affinity: derive a stable key (`thread_id`, else a hash of the system+tools
prefix) and pass llama-server's slot-selection hint (`id_slot` /
`--slot-save`-style routing where supported by the pinned build) so repeat
turns of the same conversation prefer the slot that already holds their prefix.
Guard behind a flag; verify against the pinned llama.cpp build's actual slot
API before relying on it (do not assume — the build is digest-pinned, see
model_notes §"build pinning"). If the build doesn't expose usable slot routing,
fall back to relying on `--cache-reuse` + longest-prefix auto-match (today's
behavior) and document that.

### 5.5 Server-flag review
- Re-evaluate `--cache-reuse 256` vs a larger window now that the system prefix
  is stabilized (a bigger reuse window helps multi-turn tool flows where the
  mid-prompt diverges). Measure, don't guess.
- Confirm `--cache-ram 2048` is sized for the slot count × context.
- Leave `ctx-checkpoints = 0` **as-is** (crash tombstone) until the upstream
  restore bug is confirmed fixed; note it as the reason full re-prefill happens
  past the reuse window.
- Keep KV **F16** (quantization tombstone).

### 5.6 Instrumentation (prerequisite for §8 testing)
Add cache metrics to `call_log` ([app/call_log.py](../../app/call_log.py)) and
the admin status:
- **From llama-server response `timings`**: `prompt_n` (tokens prefilled),
  `prompt_ms`, `predicted_n`, `predicted_ms`. Cache reuse = `n_prompt_total -
  prompt_n` (the tokens it did *not* reprocess). Expose a derived
  `cache_hit_tokens` / `cache_hit_ratio` per call.
- **From `/slots`** (and optional `--metrics` Prometheus endpoint): per-slot
  `n_past`, occupancy, evictions.
- Surface a dashboard tile: rolling local cache-hit ratio + mean TTFT.

---

## 6. Anthropic Caching + Egress (Phase 2)

### 6.1 New backend in the router
Add `app/anthropic_engine.py` — an egress client mirroring the
`_LlamaServerProxy` contract (`create_chat_completion(messages, stream, **kw)`
sync + an async streaming variant), so `openai_compat.py` and the Socket.IO
path stay unchanged. Responsibilities:
- Auth via `ANTHROPIC_API_KEY` (env; never logged).
- **OpenAI → Anthropic Messages mapping**: hoist `system` to the top-level
  `system` param (as blocks, so breakpoints attach), map roles, require
  `max_tokens`, translate tool schemas (`tools` / `tool_choice`), map
  `stop` → `stop_sequences`.
- **Streaming translation**: Anthropic SSE events
  (`content_block_delta`, `message_delta`, …) → the OpenAI
  `chat.completion.chunk` shape the existing clients (and the `_ThinkingSplice`
  consumers) expect. Anthropic "thinking" maps to the same
  `reasoning_content`→`<think>` splice already used for local
  ([llm_engine_server.py:191](../../app/llm_engine_server.py#L191)) so noted's
  legacy parser keeps working unchanged.

### 6.2 Breakpoint placement policy
From `cache_spans`, emit up to **4** `cache_control: {type:"ephemeral"}`
markers (Anthropic's hard limit), chosen by priority:
1. **tools** block (largest, most static — noted ships 60+ tool defs).
2. end of **system** block.
3. end of static **context** (RAG/document preamble) if present.
4. end of the **conversation prefix** (all but the latest user turn) for
   multi-turn reuse.

If the client (Hermes) already supplied markers, **honor them** and only
synthesize to fill remaining budget. Respect the 1024-token (Sonnet/Opus) /
2048-token minimum cacheable prefix — don't mark spans too small to cache.

### 6.3 Unified `cache_control` semantics (the "author once" payoff)
| Author intent (`cache_control` on a block) | Local (llama.cpp) | Anthropic |
|---|---|---|
| Mark system / tools / context as cacheable | Ensure it sits in the stable front prefix; strip marker; rely on `--cache-reuse` + prefix match | Emit `cache_control` breakpoint at that boundary |
| No markers at all | Stable-prefix assembly still applies (§5.1) | Auto-place breakpoints by §6.2 policy |
| Marker on a volatile/late block | Ignored for prefix (can't help) | Skipped (would never read-hit; wastes a breakpoint) |

So Hermes authors **one** Anthropic-style request; local routing treats markers
as prefix-ordering hints (then discards them), Anthropic routing treats them as
real cache breakpoints. Identical authored artifact, backend-appropriate
behavior.

### 6.4 Consolidation note
Because noted already implements a Claude path + model merge, Phase 2 should
**lift that logic into agent_server** and let noted call through (selecting a
`claude-*` model_id just routes), rather than maintaining two Anthropic clients.
This is the genuine architectural win and should be coordinated with the noted
owner; see [active_model_switching_sdk.md](../active_model_switching_sdk.md)
§dropdown.

---

## 7. API & Data Changes (summary)
- **Request (additive, all optional):**
  - `backend`: `"local"｜"anthropic"` (or `X-Agent-Backend` header).
  - Accept Anthropic-shaped `cache_control` inside content blocks and a
    top-level `system` array (parsed, not rejected).
- **`GET /v1/models`:** add `backend` field; list Anthropic chat models when
  `ANTHROPIC_API_KEY` is configured.
- **No breaking change** to existing fields. An OpenAI request with no new
  fields is parsed, routed local, and (with the flag on) benefits only from the
  prefix-stability fix.
- **Flags:**
  - `CACHE_ROUTING_ENABLED` (master, default off → today's exact code path).
  - `STABLE_PREFIX_ASSEMBLY` (Phase 1 sub-flag for §5.1, so it can ship/measure
    alone).
  - `ANTHROPIC_BACKEND_ENABLED` (Phase 2).

---

## 8. Testing Strategy — BEFORE and AFTER

The plan **requires** a measured baseline before any change and the same suite
after each phase. Reuse the repo's existing probe style (`probe_vision.sh`,
`_vision_test/`, the `tests/chat_smoke_probe.py` pattern referenced in
model_notes) rather than inventing a framework.

### 8.1 Test assets to add (`tests/cache/`)
- `bench_local_cache.py` — drives `/v1/chat/completions` against `llama-vision`
  and records, per call, from the response `timings`: total prompt tokens,
  `prompt_n` (reprocessed), derived `cache_hit_tokens`, TTFT, tok/s. Also polls
  `/slots`.
- `bench_anthropic_cache.py` — same shape against the Anthropic backend,
  recording `usage.cache_creation_input_tokens` /
  `usage.cache_read_input_tokens`, TTFT, cost estimate.
- `scenarios.py` — fixed, reproducible conversations:
  1. **Cold single-turn** (long system prompt, no history).
  2. **Warm repeat** (identical request twice — isolates pure prefix reuse).
  3. **Multi-turn** (5 turns, growing history — tests suffix divergence vs
     `--cache-reuse`).
  4. **Tool-calling round-trip** (system + 60 tool defs + tool result —
     exercises §2.4 transforms and the largest cacheable block).
  5. **Multimodal** (text + `image_url`) — guards §5.3 (markers stripped,
     image preserved).
  6. **Minute-boundary** (two identical requests ~61s apart) — directly
     measures the §2.4(1) timestamp regression and its fix.
- `regression_suite.py` — asserts **functional** parity: same final answer
  content (modulo sampling seed), tool calls fire, vision answers, voice
  injection still works, Socket.IO stream intact.

### 8.2 BEFORE (baseline capture — run on v2 as-is)
1. Pin sampling (`seed`, `temperature` low) for determinism.
2. Run `scenarios 1–6` × N reps against the **current** stack; archive raw
   timings to `tests/cache/baseline/`. Key numbers:
   - Local cache-hit ratio per scenario (expect **near-zero on the
     minute-boundary** scenario, confirming §2.4(1)).
   - TTFT and tok/s cold vs warm.
3. Snapshot `/slots` behavior under 2 concurrent conversations (eviction).
4. Record the Socket.IO path separately (it lacks the timestamp preamble — its
   warm-repeat hit ratio should already be higher; this asymmetry is the proof
   point for the fix).

### 8.3 AFTER — per phase, gated on no regression
- **Phase 1 (local revision):**
  - Re-run all scenarios with `STABLE_PREFIX_ASSEMBLY` on.
  - **Acceptance:** warm-repeat and minute-boundary cache-hit ratio rises
    materially (target: minute-boundary ratio ≈ warm-repeat ratio, i.e. the
    timestamp no longer busts the prefix); cold-path answers byte-equivalent to
    baseline; `regression_suite` green; multimodal/tool scenarios unchanged.
  - Measure TTFT delta on warm turns (expected reduction = saved prefill time).
- **Phase 2 (Anthropic):**
  - `bench_anthropic_cache` on scenarios 1–4: second call shows
    `cache_read_input_tokens > 0` at the placed breakpoints; the 4-breakpoint
    budget respected; Hermes-authored markers honored (assert our emitted
    breakpoints ⊇ client markers, ≤ 4 total).
  - Streaming-shape conformance: Anthropic→OpenAI chunk translation passes the
    same `regression_suite` consumers (noted parser, `_ThinkingSplice`).
  - Cost check: cache-read path cheaper than cache-miss on repeat (sanity on
    `usage`).
- **Cross-backend transparency test:** one identical Hermes request sent with
  `backend=local` and `backend=anthropic`; assert both return a well-formed
  answer and the *same client code* parses both.

### 8.4 Continuous guardrails
- The dashboard cache-hit tile (§5.6) becomes a live regression canary in
  production.
- Add a CI-style smoke (local only, no GPU needed for the parsing/normalization
  unit tests): `strip_cache_control` preserves blocks; ingress detector
  classifies OpenAI vs Anthropic shapes; breakpoint planner never exceeds 4.

---

## 9. Phased Rollout

| Phase | Scope | Flag | Exit criteria |
|---|---|---|---|
| **0. Baseline** | Add instrumentation (§5.6) + `tests/cache/` + capture BEFORE (§8.2). No behavior change. | — | Baseline archived; cache-hit metric visible on dashboard. |
| **1. Local revision** | Stable-prefix assembly (§5.1), per-block strip (§5.3), unified assembly across both ingress paths, server-flag review (§5.5), optional slot affinity (§5.4). | `STABLE_PREFIX_ASSEMBLY` | AFTER §8.3 Phase-1 acceptance met; ship default-on after soak. |
| **2. Anthropic egress** | `anthropic_engine.py`, router backend selection (§4.3), Messages mapping + SSE translation (§6.1), breakpoint policy (§6.2). | `ANTHROPIC_BACKEND_ENABLED` | §8.3 Phase-2 acceptance; noted consolidation agreed. |
| **3. Unify & consolidate** | Move noted's Claude path behind agent_server; `/v1/models` backend field; docs. | `CACHE_ROUTING_ENABLED` default-on | noted routes `claude-*` through agent_server; single Anthropic client in the stack. |

Each phase is independently revertible (flag off = prior code path), consistent
with the `LLAMA_SERVER_URL` / `FORCE_VOICE_INJECTION` rollback discipline
already in the codebase.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Stable-prefix change alters an agent's behavior (date moved/removed) | Default option **B** (keep date, move to tail); regression suite asserts answer parity; flag-gated. |
| Slot-affinity assumes a slot API the pinned build lacks | Verify against the digest-pinned build first; fall back to `--cache-reuse` auto-match; never hard-depend. |
| Larger `--cache-reuse` raises VRAM/latency | Measure per §8; revert flag; KV stays F16; `c` drop to 65536 already the OOM lever. |
| Anthropic SSE→OpenAI translation drifts from local shape | Same `regression_suite` consumers gate it; reuse existing `_ThinkingSplice`. |
| `cache_control` markers on volatile blocks waste Anthropic breakpoints | Planner skips spans below min cacheable size / on late blocks (§6.2). |
| Two Anthropic clients (noted + agent_server) diverge | Phase 3 consolidation; until then agent_server's path is flag-off. |
| Secret handling | `ANTHROPIC_API_KEY` env-only, never in `call_log`/`http_log`/dashboard. |

## 11. Open Questions
1. Does the **digest-pinned** llama.cpp build expose per-request slot pinning
   (`id_slot`) usable from the HTTP API, or only auto longest-prefix match?
   (Gate §5.4 on this.)
2. Default timestamp option — A, B, or C (§5.1)? (Recommend B; needs owner ok
   since it touches every agent's prompt.)
3. Hermes' exact on-the-wire shape — does it POST Anthropic-native to `/v1`, or
   an OpenAI body with `cache_control` smuggled into blocks? (Confirms the §4.2
   detector; both are supported, but the default backend mapping differs.)
4. Consolidation ownership — who moves noted's `dispatch_claude` behind
   agent_server, and on what timeline? (Phase 3 dependency.)
5. Is per-conversation Anthropic breakpoint reuse worth the write-cost on short
   threads, or only enable breakpoints above a turn/length threshold?

---

## 12. Appendix — file touch-map (anticipated)
- **New:** `app/cache_routing.py` (canonical model + ingress parser + planner),
  `app/anthropic_engine.py` (egress), `tests/cache/*`.
- **Edit:** `app/openai_compat.py` (`_build_messages` stable-prefix + per-block
  strip; router hook in the dispatch at
  [openai_compat.py:307](../../app/openai_compat.py#L307)), `app/main.py`
  (Socket.IO assembly parity; backend wiring), `app/llm_engine_server.py`
  (`generate_stream` assembly parity; share normalizer),
  `app/call_log.py` (cache metrics), admin status/dashboard (cache tile),
  `docker-compose.adapter.yml` (`--cache-reuse` retune if measured),
  `documents/llama_server_model_notes.md` (record findings),
  `README.md` (backend + caching section).
- **Unchanged:** chat templates, `agent_config.json` schema (Anthropic models,
  if listed, would be a new neutral group or `backend:"anthropic"` flag —
  decide in Phase 2), embedders/reranker.
```
</content>
