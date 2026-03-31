import time
import statistics
import psutil
import os
import numpy as np

from job_recommendation import JobRecommenderEngine


ALPHA = 0.1
N_QUERIES = 300
TOP_K = 10
SEED = 42


def benchmark_query_latency(engine):
    df = engine.df[engine.df["active"] == 1].reset_index(drop=True)
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(df), min(N_QUERIES, len(df)), replace=False)

    latencies = []

    for idx in sample_idx:
        query = df.loc[idx, "combined_text"]

        start = time.perf_counter()
        engine.recommend(query_text=query, top_k=TOP_K)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    return latencies


def benchmark_add_job(engine):
    start = time.perf_counter()
    engine.add_job(
        title="Test Benchmark Role",
        company="Test Corp",
        city="Test City",
        desc="Benchmark testing description"
    )
    end = time.perf_counter()
    return (end - start) * 1000


def main():
    engine = JobRecommenderEngine()
    engine.alpha = ALPHA

    print(f"\nBenchmarking with alpha={ALPHA}\n")

    # Query latency
    latencies = benchmark_query_latency(engine)

    avg_latency = statistics.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    max_latency = max(latencies)

    print("Query Latency (ms)")
    print("------------------")
    print(f"Average: {avg_latency:.2f} ms")
    print(f"p95:     {p95_latency:.2f} ms")
    print(f"Max:     {max_latency:.2f} ms")

    # Add-job time
    add_time = benchmark_add_job(engine)

    print("\nAdd Job Update Time (ms)")
    print("-------------------------")
    print(f"Add-job latency: {add_time:.2f} ms")

    # Memory usage
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    print("\nMemory Usage")
    print("------------")
    print(f"RAM usage: {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()