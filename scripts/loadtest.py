"""Load test for the API layer. Run the API first (uv run sec-agent serve), then:

    uv run python scripts/loadtest.py --requests 500 --concurrency 20

Hits /facts (no model call) and /ask with one question (a cache hit after the first call),
so it measures the API, database and cache, not model speed. Model latency is in the eval results.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import httpx


async def worker(client, queue, latencies, errors):
    while True:
        try:
            method, path, body = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        start = time.perf_counter()
        try:
            resp = await client.request(method, path, json=body)
            if resp.status_code >= 400:
                errors[resp.status_code] = errors.get(resp.status_code, 0) + 1
        except httpx.HTTPError:
            errors["network"] = errors.get("network", 0) + 1
        latencies.append((time.perf_counter() - start) * 1000)


async def main(args):
    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=120) as client:
        warm = await client.post("/ask", json={"question": args.question})
        warm.raise_for_status()
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(args.requests):
            if i % 2:
                queue.put_nowait(("GET", "/facts/AAPL/revenue", None))
            else:
                queue.put_nowait(("POST", "/ask", {"question": args.question}))
        latencies: list[float] = []
        errors: dict = {}
        started = time.perf_counter()
        await asyncio.gather(*[worker(client, queue, latencies, errors) for _ in range(args.concurrency)])
        elapsed = time.perf_counter() - started

    latencies.sort()
    p = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]  # noqa: E731
    print(f"requests     {len(latencies)}  (concurrency {args.concurrency})")
    print(f"throughput   {len(latencies) / elapsed:.1f} req/s")
    print(f"latency      p50 {p(0.50):.0f} ms   p95 {p(0.95):.0f} ms   p99 {p(0.99):.0f} ms   mean {statistics.mean(latencies):.0f} ms")
    print(f"errors       {errors or 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--question", default="What was Apple's revenue in fiscal 2025?")
    asyncio.run(main(parser.parse_args()))
