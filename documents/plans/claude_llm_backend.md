# Claude LLM backend for agent_server — centralized local↔Claude routing

**Status:** PLAN (not built). Investigation + design only; nothing implemented.

**Goal:** Let agent_server's consumers (cv, noted, …) use Claude instead of the
local Gemma model, selectable per-request and/or via a single global switch, with
one API key and one adapter — reusing the native Claude adapter noted **already has**.

## What already exists (verified 2026-08-14)

- **agent_server** forwards every LLM call through a **single global engine**
  (`LLMEngine` base → `LlamaServerEngine`, selected in `build_engine_or_raise()`,
  [app/main.py:272-291](../../app/main.py)). Two surfaces: the OpenAI REST proxy
  (`_LlamaServerProxy.create_chat_completion`) and the Socket.IO async path
  (`LlamaServerEngine.generate_stream`). All consumers share this one engine.
- **noted already implements dynamic local↔Claude switching**, per-consumer:
  - `LLMRouter` (`noted/backend/app/managers/llm_router.py`) routes by model-id
    prefix (`claude-*` → Anthropic, else local), **same interface** as the local
    manager so `llm.py` is untouched. `select_model()`: a cloud pick is in-memory;
    a local pick calls agent_server's active-model switch. `health()` **merges**
    local + Claude models into one dropdown list.
  - `AnthropicLLMManager` (`noted/backend/app/managers/anthropic_llm_manager.py`)
    is a **native Messages API adapter** (`POST api.anthropic.com/v1/messages`,
    `x-api-key` + `anthropic-version`, raw `aiohttp` — **not** the OpenAI-compat
    layer, **not** the official SDK). It already translates Anthropic SSE:
    `thinking_delta`→`<think>…</think>`, `text_delta`→content, `tool_use`→tool-call
    events; handles `/think` directives and Anthropic's strict user/assistant
    alternation.
  - **noted's Claude path is DIRECT — it bypasses agent_server entirely.** Only
    *local* picks go through agent_server.
- **cv has no Claude path at all** — it only talks to agent_server.

So the "native Claude adapter" is ~90% written already (in noted); the design
question is **where the router lives**, not how to call Claude.

## Two architectures

### Option A — Centralize in agent_server (recommended)

agent_server itself gains the ability to talk to **either** llama-vision (local)
**or** Anthropic. Clients keep calling agent_server unchanged; they never hold an
Anthropic key or adapter.

End-to-end for a request:
1. **Client (cv/noted/…)** POSTs to agent_server `:7701` with a `model` field —
   unchanged. Claude via (a) `model: "claude-sonnet-5"` per-request, or (b) normal
   agent/model id while the **active backend** is set to Claude.
2. **Router engine** (implements `LLMEngine`, mirrors noted's `LLMRouter`) inspects
   the resolved model id per call: `claude-*` → `AnthropicEngine`; else →
   `LlamaServerEngine` → llama-vision. Sits behind **both** surfaces (REST proxy +
   Socket.IO `generate_stream`).
3. **`AnthropicEngine`** translates OpenAI-shaped request → Anthropic Messages API
   and the response/SSE → OpenAI-shaped chunks. Ported from noted's
   `AnthropicLLMManager`. **Emit `reasoning_content` deltas (from `thinking_delta`)
   rather than inline `<think>`** so the *existing* `_ThinkingSplice` /
   `_splice_nonstreaming` + voice-injection run identically for both backends and
   the noted/cv parsers stay unchanged.
4. **Selection (model-selectable):** Claude models join agent_server's
   `GET /v1/models` list next to the local ones. Two levers: **per-request**
   (`model: claude-…`) and a **global switch** — extend
   `POST /admin/api/active-model` to accept a Claude id, flipping the active
   backend for every consumer (this is the original "single parameter for
   everyone"). No llama-vision restart is needed to select Claude (unlike local
   model switches) — a Claude pick is an in-memory routing change.
5. **One API key, one place.** agent_server holds `ANTHROPIC_API_KEY` + the Claude
   model list; clients never see it.

Effect: **cv gets Claude for zero code changes.** noted can later **drop** its
direct-to-Anthropic path and treat Claude as more agent_server models (one path,
one key), or keep its direct path and coexist. agent_server's call logging,
dashboard, GeoIP client tracking, and voice injection automatically cover Claude
traffic too.

### Option B — Replicate noted's pattern into cv (per-consumer)

Give cv its own `LLMRouter` + `AnthropicLLMManager` copy, like noted. Smaller and
isolated, but: duplicated adapter + API key per consumer, **no single switch**, and
every future consumer repeats the work. noted already proves it works; this just
spreads it.

### Option C — Just fix noted's

If only noted needs Claude for now: update its stale model list + legacy thinking
config (below) and touch nothing else. Least work; leaves cv without Claude and no
central switch.

**Recommendation: Option A**, and the decisive reason is **abstraction, not
convenience.** The point of centralizing is *where the abstraction boundary lives*:

- With A, the boundary is **agent_server's API**. Clients speak one stable contract
  (OpenAI `/v1/chat/completions` + Socket.IO) forever; the concrete LLM (local Gemma,
  Claude Sonnet, a future model) is swapped behind it at the `LLMEngine` seam with
  **zero client code change, ever**. That seam already exists for exactly this.
- With B (and with noted today), **every client is coupled to the concrete backend**:
  its own SDK, its own key, and model-id-prefix routing that leaks "is this Claude?"
  into the client. Adding a third backend later means editing every client again.
  noted's current design is therefore the anti-pattern for this goal — centralizing
  pulls that vendor knowledge back behind agent_server, and noted would just send a
  model id and delete its `AnthropicLLMManager`.

Secondary benefits (one key + one adapter, cv gets Claude free, shared logging/
dashboard/voice) follow from the same move. noted's adapter is the template, so the
marginal build cost over B is modest.

## Buy vs build — use LiteLLM inside agent_server (don't hand-write the adapter)

The abstraction that keeps clients unchanged is the **OpenAI-compatible boundary**,
which **agent_server already is**. Any-language clients (cv, noted, a JS app via the
standard `openai` npm package) keep speaking `/v1/chat/completions` + Socket.IO; when
the backend flips local→Claude behind that boundary, **no client changes a line.**

- **Provider abstraction inside agent_server = LiteLLM (as a library), not a
  hand-written adapter.** Verified 2026-08: LiteLLM exposes a **unified
  `reasoning_content`** for Claude thinking in OpenAI shape (streaming + non-stream)
  — feeds the existing `_ThinkingSplice` — and fronts a **local llama-server** via
  `model="openai/<name>"` + `api_base=http://llama-vision:8500`. So the engine
  becomes a thin `litellm.acompletion(model=…, api_base=…, …)` call; the "router" is
  just "pass the model string through" (`anthropic/claude-sonnet-5` vs
  `openai/gemma-4`). This **deletes** the two-way translation surface and the
  gotchas (thinking config, tool round-trips, refusal shape) and gives
  fallbacks/retries/cost-tracking for free.
  - Cost: LiteLLM is a heavier pure-Python dep than a raw-httpx adapter (many
    transitive deps), but no CUDA — fine for the slim image. Verify it emits
    `reasoning_content` in the exact streaming shape `_ThinkingSplice` consumes, and
    map current models to `thinking:{type:"adaptive"}` (LiteLLM also accepts a
    provider-agnostic `reasoning_effort`).
- **LiteLLM Proxy (standalone gateway) is redundant here** — agent_server is already
  the gateway/orchestrator (agents, memory, STT/TTS, voice, logging). Stacking a
  second proxy in front duplicates that. Prefer LiteLLM *in-process*.
- **LangGraph is the wrong layer for this goal** and is explicitly ruled out: it is
  agent *orchestration*, its provider abstraction is really LangChain's chat models
  at the code level, and its **native client contract is assistants/threads/runs
  (`@langchain/langgraph-sdk`), not OpenAI** — adopting it would force every client
  (JS included) onto a new SDK, defeating the no-client-change abstraction. OpenAI
  facades exist (`langgraph-openai-serve`) but only re-add the contract agent_server
  already provides. Revisit LangGraph only if the goal becomes stateful multi-step
  agent graphs — a different problem.

## Design detail (Option A, LiteLLM-backed)

- **`LiteLLMEngine(LLMEngine)`** + a sync `_LiteLLMProxy.create_chat_completion`
  (mirrors the `_LlamaServerProxy` shape) so `openai_compat.py` stays **unchanged**.
  Internally calls `litellm.acompletion` / `litellm.completion`, selecting provider
  by the resolved model string. Emit OpenAI-shaped chunks incl. `reasoning_content`
  so `_ThinkingSplice` / voice-injection run identically for both backends.
  - Fallback if LiteLLM's shape proves awkward: a hand-written `AnthropicEngine` on
    raw `httpx` (native `/v1/messages`, no new dep), ported from noted's
    `AnthropicLLMManager` — kept as Plan B, not the default.
- **Router engine** replaces the single global engine in `build_engine_or_raise()`;
  gated by a selector (`LLM_BACKEND`/config) so the whole thing is additive and
  reverts by flipping back to `LlamaServerEngine`.
- **Reuse** `_ThinkingSplice`, `_splice_nonstreaming`, `_maybe_inject_voice_chunk`
  unchanged by emitting OpenAI-shaped chunks with `reasoning_content`.
- **Translation surface (the real work + tests):**
  - Request: hoist system message → top-level `system`; OpenAI `tools[].function`
    → Anthropic `tools[].input_schema`; `tool_calls`/`tool` role → `tool_use`/
    `tool_result` blocks; sampling temp 0–1, `top_p`, `top_k` (Anthropic native),
    `max_tokens`, `stop_sequences`; drop `min_p`/`repeat_penalty`; base64
    `image_url` → Anthropic image blocks (native path handles base64 cleanly).
  - Response/SSE: `text`→content, `thinking`→`reasoning_content`, `tool_use`→
    `tool_calls`; map `stop_reason` (end_turn→stop, tool_use→tool_calls,
    max_tokens→length, **refusal→handle**).
- **Prompt caching:** put a `cache_control` breakpoint on the (large, stable) agent
  system prompt — where the cache-aware plan finally pays off (compat layer can't).

## Reuse gotchas (carried from noted's code)

1. **Stale model list** — noted hardcodes `claude-sonnet-4-6 / opus-4-6 /
   haiku-4-5`. Use current tiers: **`claude-sonnet-5` (recommended default) /
   `claude-opus-5` / `claude-haiku-4-5`**, all user-selectable in `/v1/models`.
2. **Legacy thinking config** — noted sends `thinking:{type:"enabled",
   budget_tokens:8000}`, which is **deprecated and 400s on Claude 4.7/5 models**.
   Must become `thinking:{type:"adaptive"}` (+ `display:"summarized"` so
   `reasoning_content` is populated for the `<think>` UI). noted's Claude path is
   therefore likely already broken against current models until updated.
3. **`refusal` stop reason** — Claude 5 classifiers can decline (HTTP 200,
   `stop_reason:"refusal"`); handle before reading content.

## Phasing

- **Phase 1 — `AnthropicEngine` + `_AnthropicProxy`** (httpx, native `/v1/messages`),
  ported from noted's manager, emitting `reasoning_content`; current models +
  adaptive thinking + refusal handling. Unit-test the two-way translation
  (esp. tool-calling round-trips).
- **Phase 2 — Router engine + selector** in `build_engine_or_raise()`; Claude models
  in `/v1/models`; extend `active-model` switch to accept Claude ids (no llama-vision
  restart on a Claude pick). Verify cv and noted both work against Claude through
  agent_server on both surfaces (REST + Socket.IO).
- **Phase 3 (optional) — noted migration:** point noted's Claude picks at agent_server
  and retire its direct `AnthropicLLMManager` path (single key, single adapter).
- **Cross-cutting:** decide which agents/consumers may egress to Claude (cost +
  data-egress gate); add per-backend cost/usage to the dashboard.

## Risks / limitations to state plainly

- **External paid egress from a central hub** — agent_server would send prompt +
  context to Anthropic; gate which consumers/agents are allowed, and surface cost.
- **Bigger test surface than noted's single path** — must be solid on REST *and*
  Socket.IO, and on tool-calling round-trips.
- **Latency** — noted gains one extra local hop if it routes through agent_server
  vs its current direct call (~ms, negligible).
- **The agents' prompts are Gemma-tuned** (voice/citation conventions); Claude runs
  them fine but some scaffolding becomes unnecessary — a prompt re-tune is optional
  follow-up, not required.
