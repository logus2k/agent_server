"""
Probe whether llama-cpp-python (built against llama.cpp b9006) supports
Gemma 4 vision through its Python API on a single Llama() instance.

Run inside the vision-test container with:
  - /models/gemma-4-E4B-it-UD-Q4_K_XL.gguf  (main model)
  - /models/mmproj-F16.gguf                  (E4B-matched projector)
  - /test/cat.png                            (test image)

Probes, in order:
  1. Print versions and bundled llama.cpp commit (if available).
  2. Enumerate chat handler classes in llama_cpp.llama_chat_format,
     looking for anything Gemma-specific.
  3. If a Gemma handler exists, use it. Otherwise try Llava16ChatHandler
     and Qwen25VLChatHandler as sanity probes (likely wrong protocol,
     but cheap to confirm).
  4. Send the cat image and print the model's response.

Outcomes:
  - Gemma handler present + works  -> integrate via single Llama() instance.
  - No Gemma handler, fallback fails -> need to write Gemma4VisionChatHandler.
  - Process aborts on load          -> bundled llama.cpp version too old
                                       or mtmd not built into the wheel.
"""
from __future__ import annotations
import base64
import json
import sys
import traceback
from pathlib import Path

SEP = "=" * 64

def banner(title: str) -> None:
    print()
    print(SEP)
    print(title)
    print(SEP)

# --- Step 1: versions ---
banner("Step 1: versions")
import llama_cpp
print(f"llama_cpp.__version__ = {getattr(llama_cpp, '__version__', '?')}")
try:
    import importlib.metadata as md
    print(f"distribution version  = {md.version('llama-cpp-python')}")
except Exception as e:
    print(f"(could not read distribution version: {e})")

# --- Step 2: handler enumeration ---
banner("Step 2: chat handlers exposed")
from llama_cpp import llama_chat_format
handlers = sorted(name for name in dir(llama_chat_format) if "ChatHandler" in name)
for h in handlers:
    print(f"  - {h}")

gemma_named = [h for h in handlers if "gemma" in h.lower()]
print()
print(f"Handlers with 'gemma' in name: {gemma_named or '(none)'}")

# --- Step 3: pick a candidate handler ---
banner("Step 3: pick handler")
candidates: list[str] = []
if gemma_named:
    candidates = gemma_named
else:
    # Fallbacks worth probing even if they likely don't speak Gemma's
    # image-token protocol. They use mtmd internally so loading proves
    # the multimodal stack is present.
    for fallback in ("Qwen25VLChatHandler", "Llava16ChatHandler", "Llava15ChatHandler"):
        if fallback in handlers:
            candidates.append(fallback)

if not candidates:
    print("ABORT: no usable chat handler class found.")
    sys.exit(2)

print(f"Candidates to try: {candidates}")

# --- Step 4: try each handler ---
img_path = Path("/test/cat.png")
if not img_path.exists():
    print(f"ABORT: test image not present at {img_path}")
    sys.exit(2)
img_b64 = base64.b64encode(img_path.read_bytes()).decode()
data_url = f"data:image/png;base64,{img_b64}"

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image in 2-3 sentences."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ],
}]

from llama_cpp import Llama

for handler_name in candidates:
    banner(f"Step 4: try {handler_name}")
    HandlerCls = getattr(llama_chat_format, handler_name)
    try:
        handler = HandlerCls(clip_model_path="/models/mmproj-F16.gguf", verbose=False)
        llm = Llama(
            model_path="/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf",
            chat_handler=handler,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False,
        )
        print(f"  loaded model + handler OK")
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        continue

    try:
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
        )
    except Exception as e:
        print(f"  GENERATE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        del llm
        continue

    msg = resp["choices"][0]["message"]
    finish = resp["choices"][0].get("finish_reason")
    usage = resp.get("usage")
    print(f"  finish_reason: {finish}")
    print(f"  usage: {usage}")

    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if reasoning:
        print()
        print("  --- reasoning_content ---")
        print(reasoning)
    print()
    print("  --- content ---")
    print(content if content else "(empty)")

    if content.strip() or reasoning.strip():
        banner(f"VERDICT: {handler_name} produced output")
        # Heuristic: did it actually describe a cat?
        text = (content + " " + reasoning).lower()
        if "cat" in text and ("orange" in text or "tabby" in text or "ginger" in text or "couch" in text or "sofa" in text):
            print(f"  Output mentions cat + plausible visual details. Vision PROBABLY working.")
        else:
            print(f"  Output produced but does NOT mention cat. Likely wrong protocol; try next handler.")
            del llm
            continue
        sys.exit(0)
    del llm

banner("VERDICT: no candidate handler produced usable output")
print("Need to write a custom Gemma4VisionChatHandler. See Qwen25VLChatHandler as template.")
sys.exit(1)
