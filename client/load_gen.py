"""Concurrent load generator for the C++ inference server."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.data_shard import make_dataset
from worker.rpc_client import LineClient, wait_for_health


@dataclass(frozen=True)
class PredictionReply:
    probability: float
    label: int
    model_version: int


@dataclass(frozen=True)
class ServingMetricsReply:
    processed_requests: int
    processed_batches: int
    queue_depth: int
    desired_replicas: int
    average_latency_ms: float
    p99_latency_ms: float


class InferenceClient(LineClient):
    def predict(self, features: np.ndarray) -> PredictionReply:
        values = [float(value) for value in features]
        response = self.request(
            f"PREDICT {len(values)} " + " ".join(f"{value:.17g}" for value in values)
        )
        parts = response.split()
        if len(parts) != 4 or parts[0] != "PREDICTION":
            raise RuntimeError(f"unexpected PREDICT response: {response}")
        return PredictionReply(float(parts[1]), int(parts[2]), int(parts[3]))

    def metrics(self) -> ServingMetricsReply:
        response = self.request("METRICS")
        parts = response.split()
        if len(parts) != 7 or parts[0] != "METRICS":
            raise RuntimeError(f"unexpected METRICS response: {response}")
        return ServingMetricsReply(
            processed_requests=int(parts[1]),
            processed_batches=int(parts[2]),
            queue_depth=int(parts[3]),
            desired_replicas=int(parts[4]),
            average_latency_ms=float(parts[5]),
            p99_latency_ms=float(parts[6]),
        )


def run_load(
    *,
    host: str = "127.0.0.1",
    port: int = 50052,
    requests: int = 200,
    concurrency: int = 32,
    n_features: int = 8,
    seed: int = 11,
) -> dict:
    features, _ = make_dataset(n_samples=max(requests, 16), n_features=n_features, seed=seed)
    client = InferenceClient(host, port)
    wait_for_health(client, timeout=5.0)

    latencies_ms: list[float] = []

    def one_request(i: int) -> PredictionReply:
        local_client = InferenceClient(host, port)
        started = time.perf_counter()
        reply = local_client.predict(features[i % len(features)])
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return reply

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, i) for i in range(requests)]
        for future in as_completed(futures):
            future.result()
    elapsed = time.perf_counter() - started

    observed = client.metrics()
    latencies_sorted = sorted(latencies_ms)
    p99 = latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.99))]
    return {
        "requests": requests,
        "concurrency": concurrency,
        "seconds": elapsed,
        "throughput_rps": requests / elapsed,
        "client_avg_latency_ms": statistics.fmean(latencies_ms),
        "client_p99_latency_ms": p99,
        "server_metrics": asdict(observed),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate inference load")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    summary = run_load(
        host=args.host,
        port=args.port,
        requests=args.requests,
        concurrency=args.concurrency,
        n_features=args.n_features,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
