# Agent Server Configuration (Quick Reference)

This document summarizes the **server-side configuration** the current code supports. Use it to understand how agents, memory, and models are wired at runtime.

---

## Files & Folders

- `app/agent_config.json` — global service config (runtime, memory, models).
- `app/agents/*.agent.json` — one file per agent preset (router, topic, …).
- `prompts/` — prompt text files (system prompts). When referenced from
  `app/agents/*.agent.json`, use paths **relative to that JSON** (e.g. `../prompts/...`).

---

## agent_config.json (schema)

```jsonc
{
	"runtime": {
		"pool_size": 2,                 // number of model workers to start
		"per_request_timeout_s": 0      // 0 disables; otherwise cancels long runs
	},
	"memory": {
		"strategies": {
			"thread_window": {
				"max_context_tokens": 1024   // token budget for rolling context
			}
		}
	},
	"models": [
		{
			"name": "jan-nano-128k-Q4_K_M",
			"path": "./models/jan-nano-128k-Q4_K_M.gguf",
			"active": true,               // exactly ONE model must be active
			"mode": "chat",
			"system_prompt": "",          // optional default; agents override
			"params": {
				"n_ctx": 8192,
				"n_threads": 0,
				"n_gpu_layers": -1,
				"temperature": 0.6,
				"top_k": 40,
				"top_p": 0.9,
				"min_p": 0.1,
				"max_tokens": 512
			}
		}
	]
}
