# agent_server — local LLM serving stack

A self-hosted, OpenAI-compatible LLM serving stack with a web admin dashboard.
It runs local **GGUF** models on your GPU via `llama.cpp`, and lets you switch
the active model, change its context size, swap the vision adapter, inspect
clients, and register new model files — all from a browser.

Published images (Docker Hub, `logus2k`):

| Image | Role |
|-------|------|
| [`logus2k/agent-server`](https://hub.docker.com/r/logus2k/agent-server) | OpenAI-compatible API + admin dashboard (port **7701**) |
| [`logus2k/agent-server-llama-adapter`](https://hub.docker.com/r/logus2k/agent-server-llama-adapter) | "llama-vision" — the `llama.cpp` server that actually loads & serves the GGUFs (port **8500**) |

> The model **weights are not baked into the images** (they're tens of GB).
> You download the GGUFs yourself into `./data/models` (see step 2).

---

## Prerequisites

- **NVIDIA GPU** (the default model set targets a 24 GB card) + recent driver.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
  so containers can use the GPU.
- **Docker Engine** with **Compose v2** (`docker compose`).
- **`curl`** (for the download script) and disk space for the models
  (~5 GB for the default model; ~30 GB for all of them).

Verify the GPU is visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick start

```bash
# 1. Get these files (docker-compose.yml, download_models.sh, agent_config.json,
#    README.md) into an empty install directory, then cd into it.

# 2. Download the models into ./data/models
chmod +x download_models.sh
./download_models.sh                 # everything (~30 GB)
#   …or just enough to boot the default model:
./download_models.sh --required-only # gemma-4 + embedders (~7 GB)

# 3. Put the config where both services expect it
mkdir -p data && cp agent_config.json data/agent_config.json

# 4. Pull images and start
docker compose pull
docker compose up -d
docker compose logs -f          # watch llama-vision load the model (~30–60 s)
```

Then open the dashboard: **http://localhost:7701/admin/**

Test the API:

```bash
curl http://localhost:7701/v1/models
curl http://localhost:7701/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4","messages":[{"role":"user","content":"hello"}]}'
```

---

## Directory layout

After download, your install directory looks like:

```
.
├── docker-compose.yml
├── download_models.sh
├── agent_config.json          # copy this to data/agent_config.json
└── data/
    ├── agent_config.json       # the running config (single source of truth)
    ├── agents/                 # (optional) *.agent.json presets — empty by default
    └── models/
        ├── gemma-4-E4B-it-UD-Q4_K_XL.gguf
        ├── mmproj-F16.gguf
        ├── chat_template_gemma-4.jinja
        ├── Qwen3.5-4B-UD-Q5_K_XL.gguf
        ├── …
        ├── bge-m3-q8/bge-m3-Q8_0.gguf
        └── bge-reranker-v2-m3-q8/bge-reranker-v2-m3-Q8_0.gguf
```

---

## Models & download addresses

`download_models.sh` fetches all of these for you. The table is the full
manifest if you prefer to download manually. **Required** = needed to boot the
default active model (`gemma-4`) + the always-resident embedders. The other
chat models are **optional** — only needed before you switch to them.

### Required

| Model | Capacity | Size | File / address |
|-------|----------|------|----------------|
| **gemma-4** (active, vision) | 64K ctx | 4.8 GB | `gemma-4-E4B-it-UD-Q4_K_XL.gguf`<br>https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf |
| ↳ vision adapter | — | 0.9 GB | `mmproj-F16.gguf`<br>https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf |
| ↳ chat template | — | tiny | save as `chat_template_gemma-4.jinja`<br>https://huggingface.co/google/gemma-4-E4B-it/resolve/main/chat_template.jinja |
| **bge-m3** (embedding) | 8K ctx | 0.6 GB | `bge-m3-q8/bge-m3-Q8_0.gguf`<br>https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf |
| **bge-reranker** (reranking) | 8K ctx | 0.6 GB | `bge-reranker-v2-m3-q8/bge-reranker-v2-m3-Q8_0.gguf`<br>https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf |

### Optional switchable chat models

| Model | Capacity | Size | File / address |
|-------|----------|------|----------------|
| qwen3.5 (4B) | 64K ctx | 3.0 GB | `Qwen3.5-4B-UD-Q5_K_XL.gguf`<br>https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q5_K_XL.gguf |
| qwen3.5-9b | 64K ctx | 5.6 GB | `Qwen3.5-9B-UD-Q4_K_XL.gguf`<br>https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf |
| smollm3 | 64K ctx | 1.8 GB | `SmolLM3-Q4_K_M.gguf`<br>https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf |
| granite-3.3 (2B) | 64K ctx | 2.9 GB | `granite-3.3-2b-instruct-UD-Q8_K_XL.gguf`<br>https://huggingface.co/unsloth/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-UD-Q8_K_XL.gguf |
| nemotron (4B) | 64K ctx | 2.6 GB | `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`<br>https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf |
| ministral (3B) | 64K ctx | 3.4 GB | `Ministral-3-3B-Reasoning-2512-Q8_0.gguf`<br>https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF/resolve/main/Ministral-3-3B-Reasoning-2512-Q8_0.gguf |

> Note the two embedder files are renamed/placed into subfolders, and the gemma
> chat template is downloaded as `chat_template.jinja` and saved as
> `chat_template_gemma-4.jinja`. The download script handles this for you.

The same addresses are stored per-model in `agent_config.json` (`download_url`)
and shown under each model name in the admin dashboard.

---

## Using the dashboard

Open **http://localhost:7701/admin/**:

- **Dashboard** — active model, GPU/VRAM, recent calls, memory threads, and a
  **Switch active model** panel with tabs: **LLM / Embeddings / Reranking /
  Vision Adapter**. Pick one and **Activate** (restarts the stack ~40 s).
  Adjust the active model's **context** with the slider.
- **Discovered models** — drop any new `.gguf` into `data/models` and it appears
  here; **Register…** creates a config entry (metadata auto-detected) so you can
  activate it.
- **Clients** — connected sockets + recent API callers, with optional GeoIP.
- **Agents** — manage agent presets (`data/agents/*.agent.json`).

You can also switch without the UI by editing `data/agent_config.json`
(`"active": true` on one model per category) and running
`docker compose restart`.

---

## Notes

- **One model resident at a time.** Only the *active* chat model is loaded
  (plus the always-resident embedder + reranker). Switching unloads the old one
  and loads the new — that's the ~40 s restart. Keeps VRAM bounded.
- **VRAM & context.** Bigger context = more VRAM (F16 KV cache). The defaults
  target a 24 GB card; raise/lower context with the slider, and if a model fails
  to load, pick a smaller context or a smaller quant.
- **`/var/run/docker.sock`** is mounted into `agent_server` so the dashboard can
  restart the containers to apply switches. This is powerful — only run on a
  trusted host. Remove that volume line to disable in-UI switching (editing the
  config + `docker compose restart` still works).
- **Auth is open by default.** To require a bearer token, set
  `OPENAI_API_KEY` in the `agent_server` service environment.
- **Ports:** `7701` = API + dashboard, `8500` = llama.cpp router (optional to
  expose).
- **Platform:** images are `linux/amd64` and require an NVIDIA GPU.

---

## Updating

```bash
docker compose pull          # get newer logus2k/* images
docker compose up -d
```
