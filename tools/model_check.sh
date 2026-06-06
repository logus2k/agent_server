#!/usr/bin/env bash
# One-shot Ministral readiness check. Run AFTER the gguf lands in data/models/.
# Tests: metadata, b9487 load (CPU+GPU), and CRITICALLY whether llama.cpp parses
# the model's [THINK]...[/THINK] into the reasoning_content channel (so
# agent_server's splice folds it to <think>) or leaks it inline.
set -u
HOST=~/env/assets/agent_server/data/models
F=$(ls "$HOST"/Ministral*Reasoning*.gguf 2>/dev/null | head -1)
[ -z "$F" ] && { echo "NOT FOUND in $HOST"; exit 1; }
CONT_PATH="/agent_server_models/$(basename "$F")"
echo "=== file: $F"; ls -lh "$F"

echo; echo "=== 1) GGUF metadata (arch / context / template) ==="
python3 /tmp/gguf_meta.py "$F" 2>&1 | head -20

echo; echo "=== 2) isolated load on b9487 (GPU + flash-attn, spare port 8599) ==="
docker exec llama-vision sh -lc "rm -f /tmp/mtest.log; (timeout 150 /app/llama-server --model '$CONT_PATH' --ctx-size 4096 --n-gpu-layers -1 --flash-attn on --jinja --reasoning-format auto --port 8599 --no-webui > /tmp/mtest.log 2>&1 &) ; sleep 1; echo spawned"
echo "  waiting for load..."
for i in $(seq 1 40); do
  code=$(docker exec llama-vision sh -lc "curl -s -m2 -o /dev/null -w '%{http_code}' http://localhost:8599/health 2>/dev/null")
  [ "$code" = "200" ] && { echo "  loaded after ~${i}s"; break; }
  sleep 2
done
docker exec llama-vision sh -lc "grep -iE 'model loaded|error|unsupported|arch|unknown|reasoning' /tmp/mtest.log | head -10"

echo; echo "=== 3) [THINK] reasoning-parse probe (does it surface reasoning_content, or leak [THINK] inline?) ==="
docker exec llama-vision sh -lc "curl -s -m60 http://localhost:8599/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Think briefly first.\"}],\"stream\":false,\"max_tokens\":200}'" \
  | python3 -c "
import sys,json
d=json.load(sys.stdin); m=d['choices'][0]['message']
rc=m.get('reasoning_content') or ''
c=m.get('content') or ''
print('reasoning_content present:', bool(rc), '| len', len(rc))
print('content has [THINK] inline (BAD):', '[THINK]' in c)
print('content has <think> inline:', '<think>' in c)
print('reasoning_content[:160]:', repr(rc[:160]))
print('content[:200]:', repr(c[:200]))
"

echo; echo "=== 4) cleanup (kill the test server) ==="
docker exec llama-vision sh -lc "pkill -f 'port 8599' 2>/dev/null; sleep 1; echo killed"
echo "=== GPU back to baseline? ==="; nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
echo; echo "VERDICT GUIDE:"
echo " - reasoning_content present + content clean  -> COMPATIBLE (splice folds to <think>, like the others)"
echo " - [THINK] leaks inline in content            -> needs --reasoning-format tuning or a family-specific strip"
