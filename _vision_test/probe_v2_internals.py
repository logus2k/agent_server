"""
Run inside the agent_server_v2 container via `docker exec` to compare
Llama construction variants and isolate the per-request overhead source.
Each variant constructs a fresh Llama, runs the same image+prompt 3 times,
reports timing for each call.
"""
import base64, sys, time
from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaRAMCache
from app.chat_handlers.gemma4_vision import Gemma4VisionChatHandler

MODEL = "/agent_server/app/data/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
MMPROJ = "/agent_server/app/data/models/mmproj-F16.gguf"
IMAGE = "/agent_server/app/data/probe_cat.png"  # bind-mounted in
PROMPT = "Describe this image in 2-3 sentences."


def make_messages():
    with open(IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]
    }]


def run_variant(name, **llama_kwargs):
    print(f"\n{'='*60}\nVARIANT: {name}\n{'='*60}", flush=True)
    print(f"  llama_kwargs: {llama_kwargs}", flush=True)
    handler = Gemma4VisionChatHandler(clip_model_path=MMPROJ, verbose=False)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=MODEL,
        n_gpu_layers=-1,
        chat_handler=handler,
        verbose=False,
        **llama_kwargs,
    )
    if llama_kwargs.get("_use_kvcache"):
        llm.set_cache(LlamaRAMCache(capacity_bytes=2 << 30))
    t1 = time.perf_counter()
    print(f"  load_time={t1 - t0:.3f}s", flush=True)

    msgs = make_messages()
    for i in range(3):
        t0 = time.perf_counter()
        r = llm.create_chat_completion(
            messages=msgs, max_tokens=256, temperature=0.2,
        )
        t1 = time.perf_counter()
        ct = r["choices"][0]["message"].get("content", "")
        u = r.get("usage") or {}
        print(
            f"  run {i+1}: elapsed={t1 - t0:.3f}s "
            f"prompt_tok={u.get('prompt_tokens')} comp_tok={u.get('completion_tokens')} "
            f"first_50={ct.strip()[:50]!r}",
            flush=True,
        )
    del llm
    del handler


# Baseline matching the test container (the "fast" reference)
run_variant(
    "BASELINE: n_ctx=8192, no FA, no presence_penalty, no kv-cache",
    n_ctx=8192,
)

# Add flash_attn (one variable changed)
run_variant(
    "+flash_attn=True",
    n_ctx=8192,
    flash_attn=True,
)

# Add LlamaRAMCache (uses our marker var; constructor unchanged)
run_variant(
    "+LlamaRAMCache(2GB)",
    n_ctx=8192,
    flash_attn=True,
    _use_kvcache=True,
)

# Big context (production-like)
run_variant(
    "PROD-LIKE: n_ctx=131072, FA, KV cache",
    n_ctx=131072,
    flash_attn=True,
    _use_kvcache=True,
)
