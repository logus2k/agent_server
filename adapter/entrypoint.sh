#!/bin/sh
# llama.cpp adapter entrypoint.
#
# On every container boot, regenerate the llama-server `--models-preset`
# INI from agent_config.json (the single source of truth), then exec the
# real llama-server. The generated preset lives ONLY here, in the
# container's ephemeral /tmp — it never exists on the host or in git.
#
# Switching the active model becomes: edit data/agent_config.json, restart
# this container (it reloads VRAM with the new preset) + agent_server.
set -eu

CONFIG="${AGENT_CONFIG:-/agent_server/app/data/agent_config.json}"
PRESET="/tmp/llama-router-models.ini"

echo "[adapter] generating $PRESET from $CONFIG"
python3 /adapter/llama_cpp_preset.py --config "$CONFIG" --out "$PRESET"
echo "[adapter] ---- generated preset ----"
sed 's/^/[adapter] /' "$PRESET"
echo "[adapter] ---------------------------"

# Locate the llama-server binary shipped in the upstream base image. In the
# pinned image it is /app/llama-server (NOT on PATH), but probe PATH first
# in case a future base changes that. A plain for/if loop so it stays safe
# under `set -e` regardless of which candidate matches.
LLAMA_BIN=""
for cand in "$(command -v llama-server 2>/dev/null || true)" /app/llama-server /llama-server; do
	if [ -n "$cand" ] && [ -x "$cand" ]; then
		LLAMA_BIN="$cand"
		break
	fi
done
if [ -z "$LLAMA_BIN" ]; then
	echo "[adapter] FATAL: llama-server binary not found on PATH, /app, or /" >&2
	exit 1
fi

echo "[adapter] exec $LLAMA_BIN --models-preset $PRESET $*"
exec "$LLAMA_BIN" --models-preset "$PRESET" "$@"
