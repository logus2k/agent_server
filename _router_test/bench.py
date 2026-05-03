"""Functional + perf benchmark of the router-mode llama-server.

Tests all 3 model types through the single endpoint (port 8501) and
times each. For embeddings, also compares single-text vs batched call
to verify llama-server's native batching beats noted-rag's per-input
loop (the per session memory `feedback_llama_cpp_python_multi_seq_decode_broken`).
"""
import json
import time

import urllib.request

URL = "http://localhost:8501"


def post(path, body):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    t1 = time.perf_counter()
    return data, t1 - t0


print("=== test 1: chat (gemma-4) ===")
r, w = post("/v1/chat/completions", {
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "What is 12*7? One number only."}],
    "max_tokens": 50,
    "temperature": 0,
})
m = r["choices"][0]["message"]
print(f"  wall={w:.3f}s  completion_tokens={r.get('usage',{}).get('completion_tokens')}  content={(m.get('content') or '')[:60]!r}")

print()
print("=== test 2a: embedding 1 text (cold) ===")
r, w = post("/v1/embeddings", {"model": "bge-m3", "input": ["The quick brown fox"]})
print(f"  wall={w:.3f}s  vectors={len(r.get('data',[]))}  dim={len(r['data'][0]['embedding'])}")

print()
print("=== test 2b: embedding 1 text (hot) ===")
r, w = post("/v1/embeddings", {"model": "bge-m3", "input": ["The quick brown fox"]})
print(f"  wall={w:.3f}s")

print()
print("=== test 2c: embedding 50 texts in ONE batched call ===")
texts = [f"sample text number {i} about machine learning topics" for i in range(50)]
r, w = post("/v1/embeddings", {"model": "bge-m3", "input": texts})
print(f"  wall={w:.3f}s  vectors={len(r.get('data',[]))}  per-text={w/len(texts)*1000:.1f}ms  effective {len(texts)/w:.1f} embeddings/s")

print()
print("=== test 2d: same 50 texts as 50 SEQUENTIAL single-text calls (mimics noted-rag's per-input loop) ===")
t0 = time.perf_counter()
for t in texts:
    post("/v1/embeddings", {"model": "bge-m3", "input": [t]})
t1 = time.perf_counter()
seq_total = t1 - t0
print(f"  wall={seq_total:.3f}s  per-text={seq_total/50*1000:.1f}ms  effective {50/seq_total:.1f} embeddings/s")
print(f"  -> batched is {seq_total/w:.1f}x faster than per-text loop")

print()
print("=== test 3: reranking (bge-reranker) ===")
r, w = post("/v1/reranking", {
    "model": "bge-reranker",
    "query": "What is photosynthesis?",
    "documents": [
        "Photosynthesis converts sunlight into energy",
        "The capital of France is Paris",
        "Plants use chlorophyll to absorb light",
    ],
})
print(f"  wall={w:.3f}s")
for x in r.get("results", []):
    print(f"    [{x.get('index')}] score={x.get('relevance_score'):+.4f}")

print()
print("=== test 4: rerank 50 candidates ===")
docs = [f"document number {i}: about photosynthesis and plants" if i % 5 == 0 else f"document {i}: unrelated content about cars" for i in range(50)]
r, w = post("/v1/reranking", {"model": "bge-reranker", "query": "photosynthesis in plants", "documents": docs})
print(f"  wall={w:.3f}s  documents={len(r.get('results',[]))}  per-doc={w/len(docs)*1000:.1f}ms")
top3 = sorted(r.get("results",[]), key=lambda x: -x["relevance_score"])[:3]
print(f"  top3 indices: {[x['index'] for x in top3]}")
