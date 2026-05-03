"""Concurrent load test for agent_server_v2 forwarding mode.

Measures:
  - Single-request baseline (1 in flight)
  - N concurrent requests (default 10)
  - Per-request wall-clock + completion_tokens
  - Aggregate throughput (sum of completion_tokens / max wall-clock)
  - Server-reported t/s if available

Usage:
  python3 concurrent_load.py [N]
"""
import asyncio
import json
import sys
import time

import httpx

URL = "http://localhost:7701/v1/chat/completions"

PROMPTS = [
    "Write a 100-word short story about a robot learning to paint.",
    "List 5 differences between TCP and UDP, briefly.",
    "Explain what a hash table is in 80 words.",
    "Compose a 100-word haiku-inspired poem about autumn rain.",
    "Give me 5 bullet points on the benefits of unit testing.",
    "Write a 100-word elevator pitch for a fictional fitness app.",
    "Explain JSON in 80 words for someone who has never seen it.",
    "Suggest 5 names for a cat with stripes, briefly justify each.",
    "Write 100 words about why bees are important.",
    "List 5 tips for writing readable Python code.",
]


async def one_call(client: httpx.AsyncClient, idx: int, prompt: str):
    body = {
        "model": "general",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.5,
    }
    t0 = time.perf_counter()
    r = await client.post(URL, json=body)
    t1 = time.perf_counter()
    r.raise_for_status()
    data = r.json()
    u = data.get("usage", {})
    return {
        "idx": idx,
        "prompt": prompt[:40],
        "completion_tokens": u.get("completion_tokens"),
        "prompt_tokens": u.get("prompt_tokens"),
        "wall_s": t1 - t0,
        "tps": u.get("completion_tokens", 0) / (t1 - t0) if (t1 - t0) > 0 else 0,
    }


async def main(n: int):
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # --- baseline: one call ---
        print(f"=== baseline: 1 request ===")
        b = await one_call(client, 0, PROMPTS[0])
        print(f"  wall={b['wall_s']:.2f}s  comp_tok={b['completion_tokens']}  tps={b['tps']:.1f}")

        # --- concurrent ---
        print()
        print(f"=== concurrent: {n} simultaneous requests ===")
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_call(client, i, p) for i, p in enumerate(prompts)])
        t1 = time.perf_counter()
        wall_total = t1 - t0
        for r in sorted(results, key=lambda x: x["wall_s"]):
            print(f"  req{r['idx']:2d}: wall={r['wall_s']:5.2f}s  comp_tok={r['completion_tokens']:4d}  per_req_tps={r['tps']:6.1f}  prompt='{r['prompt']}'")
        agg_tokens = sum(r["completion_tokens"] for r in results)
        agg_tps = agg_tokens / wall_total
        per_req_avg = sum(r["wall_s"] for r in results) / len(results)
        per_req_tps_avg = sum(r["tps"] for r in results) / len(results)
        print()
        print(f"  AGGREGATE: {agg_tokens} tokens in {wall_total:.2f}s wall = {agg_tps:.1f} t/s aggregate")
        print(f"  per-request avg wall: {per_req_avg:.2f}s")
        print(f"  per-request avg tps:  {per_req_tps_avg:.1f}")
        print(f"  baseline (1-req) tps: {b['tps']:.1f}")
        print(f"  speedup (agg vs baseline): {agg_tps / b['tps']:.2f}x")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(main(n))
