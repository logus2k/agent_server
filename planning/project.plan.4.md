Love it—let’s slot in a small “B+” phase before persistent memory so we harden what we’ve just built without churning code later.

# Updated Plan (compact)

## Phase A — Baseline (✅)

FastAPI + Socket.IO, GPU llama-cpp, streaming, agent presets loaded from `agents/*.agent.json`, prompts as files, worker pool, and server-controlled memory.

## Phase B — Server-controlled thread memory (✅)

* Client sends only `{ agent, text, thread_id }`.
* Server enforces `memory_policy` from agent preset:

  * `none` → ignore history.
  * `thread_window` → require `thread_id`, use in-proc thread buffer.
* UI has no memory controls.

---

## Phase B+ — Hardening (new, before C)

1. **Preset schema lock & validation**

   * Allowed `memory_policy`: **only** `"none"` or `"thread_window"`.
   * Fail fast on unknown keys/paths or missing prompt files.
   * Clear startup log listing loaded agents and their prompts.

2. **Thread-window tuning (still in-proc)**

   * Add per-agent **`memory.thread_window.max_context_tokens`** (the single knob we keep).
   * Robust token budgeting: include only as much history as fits; oldest first out.

3. **Strict request contract**

   * If agent policy is `thread_window` and `thread_id` is missing → immediate 400 via Socket.IO error event (already in place).
   * If `agent` name is unknown → immediate error with list of valid agents.

4. **Run-level metrics (lightweight)**

   * Log per run: `runId`, agent, thread\_id (if any), input/output token counts, latency, truncated? (yes/no).
   * Keep in logs only (no endpoint yet).

5. **Concurrency & resilience checks**

   * Verify multiple clients/threads don’t cross-contaminate memory.
   * Confirm cancel/timeout leave no dangling tasks and worker is returned to pool.

**Exit criteria for B+:**

* Startup fails loudly on misconfigured presets.
* Long answers never silently truncate history inclusion.
* Logs show token/latency metrics per run.
* Manual test proves thread isolation across different `thread_id`s.

---

## Phase C — Persistent memory (Postgres) (next)

* Pluggable `MemoryStore` with **Async SQLAlchemy** (Postgres).
* Same `thread_window` policy, but backed by DB so conversations survive restarts.
* Server maps `(agent, thread_id)` → history rows; same preamble/append workflow.

---

## What you already have vs. what changes in B+

* **Already done:** server-owned memory control, thread\_window strategy, UI cleaned, streaming intact.
* **Add in B+:** preset validation, single memory knob (`max_context_tokens`), robust token budgeting, clearer errors, and basic metrics.

If that matches your intent, I’ll prep the small diffs (schema validator + budgeting + logging) in the next step.
