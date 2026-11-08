#!/bin/sh
# llama.cpp adapter entrypoint.
#
# On every container boot, regenerate the llama-server `--models-preset`
# INI from agent_config.json (the single source of truth), then exec the
# real llama-server. The generated preset lives ONLY here, in the
# container's ephemeral /tmp — it never exists on the host or in git.
#
# Switching the active model becomes: edit data/config/agent_config.json, restart
# this container (it reloads VRAM with the new preset) + agent_server.
set -eu

CONFIG="${AGENT_CONFIG:-/agent_server/app/data/config/agent_config.json}"
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

# llama-server behind the hold proxy (adapter/hold_proxy.py; Cortex register #344): the proxy takes the port asked for
# (--port, 8500) and llama-server listens on loopback 8501. Without hold_proxy.py: llama-server as before.
if [ -f /adapter/hold_proxy.py ]; then
	PORT=8500
	ARGS=""
	skip=""
	for a in "$@"; do
		if [ -n "$skip" ]; then
			[ "$skip" = "port" ] && PORT="$a"
			skip=""
			continue
		fi
		case "$a" in
			--port) skip="port" ;;
			--host) skip="host" ;;
			*) ARGS="$ARGS $a" ;;
		esac
	done
	echo "[adapter] $LLAMA_BIN --models-preset $PRESET$ARGS --host 127.0.0.1 --port 8501, behind the hold proxy on $PORT"
	# shellcheck disable=SC2086 - the compose's own args, split as the shell received them (none holds a space)
	"$LLAMA_BIN" --models-preset "$PRESET" $ARGS --host 127.0.0.1 --port 8501 &
	exec python3 /adapter/hold_proxy.py --listen "$PORT" --upstream 8501 --pid $!
fi
echo "[adapter] exec $LLAMA_BIN --models-preset $PRESET $*"
exec "$LLAMA_BIN" --models-preset "$PRESET" "$@"
