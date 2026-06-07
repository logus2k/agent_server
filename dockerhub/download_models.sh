#!/usr/bin/env bash
# Download the GGUF models for the agent_server serving stack into ./data/models.
#
#   ./download_models.sh                 # everything (default active model + all switchable)
#   ./download_models.sh --required-only # only what's needed to boot the default (gemma-4 + embedders)
#
# Safe to re-run: existing complete files are skipped; partial downloads resume.
# Requires: curl.
set -euo pipefail

DEST="$(dirname "$0")/data/models"
MODE="${1:-all}"

dl() {  # dl <url> <relative-dest-path>
  local url="$1" out="$DEST/$2"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    echo "  ✓ exists  $2"
    return
  fi
  echo "  ↓ $2"
  curl -L --fail --retry 3 -C - -o "$out" "$url"
}

echo "==> Required (boots the default active model: gemma-4 + embedders)"
dl https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf  gemma-4-E4B-it-UD-Q4_K_XL.gguf
dl https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf                  mmproj-F16.gguf
dl https://huggingface.co/google/gemma-4-E4B-it/resolve/main/chat_template.jinja                    chat_template_gemma-4.jinja
dl https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf                    bge-m3-q8/bge-m3-Q8_0.gguf
dl https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf bge-reranker-v2-m3-q8/bge-reranker-v2-m3-Q8_0.gguf

if [ "$MODE" = "--required-only" ]; then
  echo "Done (required only). Switch to other models after downloading them with --all."
  exit 0
fi

echo "==> Optional switchable chat models (needed only before you activate them)"
dl https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q5_K_XL.gguf                         Qwen3.5-4B-UD-Q5_K_XL.gguf
dl https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf                         Qwen3.5-9B-UD-Q4_K_XL.gguf
dl https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf                               SmolLM3-Q4_K_M.gguf
dl https://huggingface.co/unsloth/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-UD-Q8_K_XL.gguf granite-3.3-2b-instruct-UD-Q8_K_XL.gguf
dl https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
dl https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF/resolve/main/Ministral-3-3B-Reasoning-2512-Q8_0.gguf Ministral-3-3B-Reasoning-2512-Q8_0.gguf

echo "Done. All models in $DEST"
