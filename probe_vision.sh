#!/usr/bin/env bash
# Smoke-test the llama-vision sidecar with a known image.
# Prereq: docker compose -f docker-compose.llama-vision.yml up -d
# and the server has finished loading (look for "HTTP server listening"
# in `docker logs llama-vision`).
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8500/v1/chat/completions}"
IMAGE="${IMAGE:-/home/logus/env/assets/noted/frontend/images/cat.png}"
PROMPT="${PROMPT:-Describe this image in 2-3 sentences.}"

if [[ ! -f "$IMAGE" ]]; then
  echo "Test image not found: $IMAGE" >&2
  exit 1
fi

EXT="${IMAGE##*.}"
case "${EXT,,}" in
  png) MIME="image/png" ;;
  jpg|jpeg) MIME="image/jpeg" ;;
  webp) MIME="image/webp" ;;
  *) MIME="application/octet-stream" ;;
esac

PAYLOAD_FILE=$(mktemp /tmp/probe_vision.XXXXXX.json)
trap 'rm -f "$PAYLOAD_FILE"' EXIT

python3 - <<PY
import base64, json
with open("$IMAGE","rb") as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": """$PROMPT"""},
            {"type": "image_url", "image_url": {"url": f"data:$MIME;base64,{b64}"}}
        ]
    }],
    "max_tokens": 1024,
    "temperature": 0.2
}
with open("$PAYLOAD_FILE","w") as f:
    json.dump(payload, f)
PY

echo "POST $ENDPOINT (image=$IMAGE)"
echo "---"
RESP=$(curl -sS --max-time 600 "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  --data-binary @"$PAYLOAD_FILE")
echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
msg = r['choices'][0]['message']
print('model:', r.get('model'))
print('finish:', r['choices'][0].get('finish_reason'))
print('usage:', r.get('usage'))
t = r.get('timings') or {}
if t:
    print(f\"timings: prompt {t.get('prompt_n')}t in {t.get('prompt_ms'):.0f}ms ({t.get('prompt_per_second',0):.1f} t/s), \"
          f\"predict {t.get('predicted_n')}t in {t.get('predicted_ms'):.0f}ms ({t.get('predicted_per_second',0):.1f} t/s), \"
          f\"cached {t.get('cache_n')}\")
if msg.get('reasoning_content'):
    print('---reasoning---')
    print(msg['reasoning_content'])
print('---content---')
print(msg.get('content') or '(empty)')
"
