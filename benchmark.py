import requests, time, statistics, json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"

TEST_PROMPTS = [
    "Hi, I'd like to book a dental appointment.",
    "My name is John Smith.",
    "I need a routine check-up.",
    "How about this Thursday at 10 AM?",
    "Can I cancel my appointment?",
    "What services do you offer?",
    "Can you recommend a restaurant nearby?",
]

def measure_latency(prompt: str, n_runs: int = 5) -> dict:
    latencies = []
    ttft_list = []

    for _ in range(n_runs):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }

        start = time.time()
        first_token_time = None

        with requests.post(OLLAMA_URL, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if first_token_time is None and chunk.get("message", {}).get("content"):
                        first_token_time = time.time() - start
                    if chunk.get("done"):
                        break

        latencies.append(time.time() - start)
        if first_token_time:
            ttft_list.append(first_token_time)

    return {
        "prompt_preview":   prompt[:45] + "...",
        "avg_latency_s":    round(statistics.mean(latencies), 3),
        "p95_latency_s":    round(sorted(latencies)[int(0.95 * n_runs)], 3),
        "avg_ttft_s":       round(statistics.mean(ttft_list), 3) if ttft_list else "N/A",
        "min_s":            round(min(latencies), 3),
        "max_s":            round(max(latencies), 3),
    }

if __name__ == "__main__":
    print(f"Benchmarking {MODEL}...\n")
    all_results = []

    for prompt in TEST_PROMPTS:
        r = measure_latency(prompt)
        all_results.append(r)
        print(f"✓ \"{r['prompt_preview']}\"")
        print(f"  Avg: {r['avg_latency_s']}s | P95: {r['p95_latency_s']}s | TTFT: {r['avg_ttft_s']}s\n")

    overall = statistics.mean([r['avg_latency_s'] for r in all_results])
    print(f"Overall average latency: {overall:.3f}s")

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to benchmark_results.json")
