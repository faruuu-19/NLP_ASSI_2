from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000")
OUTPUT_PATH = Path("benchmark_results.json")

TEST_TURNS = [
    "I want to book an appointment",
    "Maria Gonzalez",
    "routine check-up",
    "Thursday at 10 AM",
    "maria@example.com",
    "yes",
    "What services do you offer?",
    "Can you recommend a restaurant nearby?",
]


def benchmark_chat_endpoint(runs: int = 3) -> dict:
    turn_results = []

    for _ in range(runs):
        session = requests.post(f"{BASE_URL}/sessions", timeout=10).json()["session"]
        for prompt in TEST_TURNS:
            start = time.perf_counter()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"session": session, "message": prompt},
                timeout=60,
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            payload = response.json()
            session = payload["session"]
            turn_results.append({
                "prompt": prompt,
                "latency_s": round(elapsed, 3),
                "intent": payload["intent"],
                "state": payload["state"],
            })

    grouped = {}
    for item in turn_results:
        grouped.setdefault(item["prompt"], []).append(item["latency_s"])

    aggregated = []
    for prompt, latencies in grouped.items():
        aggregated.append({
            "prompt": prompt,
            "avg_latency_s": round(statistics.mean(latencies), 3),
            "p95_latency_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 3),
            "min_latency_s": round(min(latencies), 3),
            "max_latency_s": round(max(latencies), 3),
        })

    return {
        "base_url": BASE_URL,
        "runs": runs,
        "overall_avg_latency_s": round(statistics.mean(item["latency_s"] for item in turn_results), 3),
        "turn_count": len(turn_results),
        "results": aggregated,
    }


if __name__ == "__main__":
    result = benchmark_chat_endpoint()
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
