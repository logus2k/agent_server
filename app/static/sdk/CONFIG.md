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
	// `models` is grouped by task; exactly ONE entry in `chat` is active.
	// Each entry self-describes; the llama.cpp adapter generates the
	// llama-server preset from this. See documents/llama_server_model_notes.md.
	"models": {
		"chat": [
			{
				"active": true,                 // exactly ONE active chat model
				"name": "Gemma 4 E4B IT Q4 KXL GGUF",
				"model_id": "gemma-4",          // forward id + generated [section] header
				"family": "gemma",              // gates model-specific response handling
				"active_backend": "llama_cpp",
				"context": 131072,
				"reasoning": true,
				"vision": true,
				"system_prompt": "",            // optional default; agents override
				"sampling": { "temperature": 1, "top_k": 64, "top_p": 0.95, "min_p": 0, "max_tokens": 131072 },
				"backends": {
					"llama_cpp": {
						"model_file": "/agent_server_models/gemma-4-E4B-it-UD-Q4_K_XL.gguf",
						"projector": "/agent_server_models/mmproj-F16.gguf",
						"options": { "n-gpu-layers": -1, "flash-attn": "on", "jinja": true,
						             "chat-template-file": "/agent_server_models/chat_template_gemma-4.jinja",
						             "ctx-checkpoints": 0 }
					}
				}
			}
		],
		"embedding": [ /* bge-m3   — active_backend + backends.llama_cpp.options */ ],
		"reranking": [ /* bge-reranker */ ]
	}
}
