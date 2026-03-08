from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx

BASE_URL = os.getenv("STRESS_BASE_URL", "http://127.0.0.1:8000")
OUTPUT_PATH = Path("stress_test_results.json")
CONCURRENT_USERS = int(os.getenv("STRESS_USERS", "20"))

SCENARIO = [
    "I want to book an appointment",
    "Maria Gonzalez",
    "routine check-up",
    "Thursday at 10 AM",
    "maria@example.com",
    "yes",
]


async def run_user(client: httpx.AsyncClient, user_index: int) -> list[float]:
    session = (await client.post(f"{BASE_URL}/sessions")).json()["session"]
    latencies = []
    for message in SCENARIO:
        start = time.perf_counter()
        response = await client.post(f"{BASE_URL}/chat", json={"message": message, "session": session})
        response.raise_for_status()
        latencies.append(time.perf_counter() - start)
        session = response.json()["session"]
    return latencies


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(*[run_user(client, idx) for idx in range(CONCURRENT_USERS)])

    flattened = [lat for user in results for lat in user]
    report = {
        "base_url": BASE_URL,
        "concurrent_users": CONCURRENT_USERS,
        "total_requests": len(flattened),
        "avg_latency_s": round(statistics.mean(flattened), 3),
        "p95_latency_s": round(sorted(flattened)[max(0, int(0.95 * len(flattened)) - 1)], 3),
        "max_latency_s": round(max(flattened), 3),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
