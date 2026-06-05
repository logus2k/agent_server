# Reliable Citations — Spec & Implementation Plan

Make RAG citations reliable across **all** chat models (Gemma, Qwen,
SmolLM3, Granite), by never asking the model to reproduce fragile tag
strings. A general numbered-reference scheme fixes every model; an
optional Granite-native path layers on top for Granite's trained
grounded-citation behaviour.

Status: **PLANNED** — not yet implemented. Authored 2026-06-05.

## Problem

The CV's evidence carries citation tags the model is asked to copy
verbatim into its answer:

- `[E:VB_01]` (entity), `[markdown_chunk:7a9f]` (doc chunk),
  `[R:src>type>tgt]` (relationship).

These are long, arbitrary, hard-to-reproduce strings, and **LLMs are
bad at emitting arbitrary hex/IDs verbatim**. The base `cv_assistant`
prompt even warns about it ("copy the entire tag verbatim … label-based
citations produce 404s"). Observed failure rate is high and **model-
agnostic**: SmolLM3 ~1/4 raw, Granite-3.3 ~2/3, and **even Gemma fails
frequently**. So this is structural, not a small-model quirk — prompting
alone won't fix it.

The citation **resolver** (cv-backend / client) maps these exact tags to
clickable sources; a garbled or dropped tag = a broken or missing
citation.

## Core idea

The model should only ever emit a **simple integer reference** (`[1]`,
`[2]`, …) — which any model can produce reliably — and the **server
maps that integer back to the real tag** before the resolver/client
sees it. The fragile hex never touches the model.

Both phases below share one server-side primitive:

> a per-request map `{ N -> real_tag }` built when evidence is
> assembled, plus a post-processor that rewrites the model's simple refs
> into real tags.

## Phase 1 — Numbered references (general, all models)

Addresses the systemic failure for every model.

1. **Evidence assembly (cv-backend).** When building the evidence block
   appended to the user prompt, assign each retrieved chunk a small
   integer id and present it as `[1] <chunk text>` (replacing, not
   alongside, the raw tag so the model isn't tempted to copy hex). Build
   `cite_map = {1: "E:VB_01", 2: "markdown_chunk:7a9f", ...}` for the
   request.
2. **Prompt (per-family CV prompts).** Instruct: "End each sentence that
   uses evidence with its source number, e.g. `[1]`. Use only the
   numbers shown in the evidence." (Trivial for any model.)
3. **Post-process (cv-backend, on the answer stream).** Rewrite each
   `[N]` -> the real tag via `cite_map`, before the client renders / the
   resolver runs. Streaming-safe via a small stateful rewriter (mirror
   `_ThinkingSplice`'s held-tail approach) or by rewriting on sentence
   boundaries.
4. **Validate / repair.** Drop out-of-range `[N]`; never fabricate a
   citation for an uncited sentence; log citation coverage per turn.

Outcome: the model emits only `[1]`; the fragile tag is reattached
server-side. Works uniformly for gemma / qwen / smollm3 / granite.

## Phase 2 — Granite-native citations (Granite only, layered)

Leverages Granite's *trained* grounded-citation behaviour, reusing the
same `cite_map`.

1. **Pass evidence via Granite's `documents` + `controls` API.** When the
   selected model is Granite, send
   `chat_template_kwargs: {"documents": [{"doc_id": "1", "text": "..."}, ...],
   "controls": ["citations"]}`. agent_server already forwards
   `chat_template_kwargs` verbatim, so this is mainly a cv-backend change
   to emit documents/controls when targeting Granite (instead of, or in
   addition to, the text evidence block).
2. Granite emits `<|start_of_cite|>{document_id: N}fact<|end_of_cite|>`
   plus an ordered citation list, grounded by its template.
3. **Post-process.** Map `document_id N` -> real tag via the same
   `cite_map`; convert `<|start_of_cite|>…` into the client's expected
   citation format.
4. **Bonus:** `controls: ["hallucinations"]` makes Granite append a
   numbered risk list of unsupported sentences — a free signal to flag
   or suppress Granite's tendency to embellish beyond the evidence.

## Phase 3 — Verify

Measure **citation coverage** (% of evidence-based sentences carrying a
valid, resolvable tag) across all 4 models, before/after, on a fixed
question set (factual, deep-dive, decline). Target: near-100% *valid*
tags, since the model now only emits integers.

## Design decisions

- **Where post-processing lives: cv-backend.** It owns evidence
  assembly, the tag vocabulary, the citation resolver, and the
  client-facing stream. agent_server stays model-neutral (it only
  forwards `chat_template_kwargs`).
- **Phase 1 is the foundation** (universal win); Phase 2 is an optional
  Granite upgrade that reuses Phase 1's `cite_map`.
- **Scope:** this fixes the *garbled / dropped tag* failure mode. It does
  NOT fix *wrong-source selection* (model cites the wrong chunk) — a
  smaller, separate problem.

## File touchpoints

- **cv-backend** (`~/env/assets/cv/backend/main.py`): evidence assembly
  (number the chunks, build `cite_map`), and the streaming answer
  post-processor (`[N]` -> tag). Phase 2: emit `documents`/`controls`
  and map `document_id`.
- **agent_server** (`data/prompts/cv_assistant_system_prompt*.txt`,
  per-family): change the citation instruction to "cite by number".
  No agent_server *code* change needed for Phase 1 (it already forwards
  `chat_template_kwargs`); Phase 2 likewise rides the existing
  passthrough.

## Open questions

- Streaming rewrite granularity: token-level held-tail vs sentence-
  boundary buffering (latency vs simplicity).
- Whether to show the model *only* the numbers, or numbers + a hidden
  mapping (numbers only is cleaner / less tempting to copy hex).
- Granite `controls` exact shape (`["citations"]` list vs
  `{"citations": true}`) — confirm against the live template at build.
