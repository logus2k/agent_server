# tools/

Reusable helpers for evaluating and integrating new GGUF chat models into the
agent_server / llama-vision stack. Both were built while integrating Nemotron and
Ministral; keep them here so model evaluation isn't re-invented each time.

## `gguf_meta.py` — GGUF metadata reader

Pure-stdlib reader of a GGUF file's key/value header. No model load, no GPU.
Prints architecture, name, context length, block count, tokenizer model, and a
chat-template summary (whether it contains `<think>`, `enable_thinking`, `/think`)
plus the template head — the fast way to decide how a model exposes reasoning
before wiring it into `agent_config.json`.

```bash
python3 tools/gguf_meta.py /path/to/model.gguf
```

## `model_check.sh` — model readiness harness

One-shot readiness check for a new chat model. Run after the GGUF lands in
`data/models/`. It:

1. dumps GGUF metadata (via `gguf_meta.py`),
2. isolated-loads the model inside the `llama-vision` container on a spare port
   (8599) with GPU + flash-attn, waiting for `/health`,
3. probes whether llama.cpp folds the model's reasoning into the
   `reasoning_content` channel (so agent_server's `_ThinkingSplice` wraps it to
   `<think>`) or leaks the raw `[THINK]`/`<think>` tags inline,
4. kills the test server and reports GPU back to baseline.

```bash
bash tools/model_check.sh
```

The script currently globs `Ministral*Reasoning*.gguf` in `data/models/` (line ~8);
edit that glob for a different model file. Container name `llama-vision` and spare
port `8599` are hardcoded.

> See [[agent-server-model-switch]] in memory for the per-family thinking-toggle
> findings these tools produced.
