"""Training entry points for baseline and parameter-server workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.data_shard import iter_batches, make_dataset, shard_dataset
from worker.model import evaluate, gradient, save_weights, train_single_worker
from worker.rpc_client import ParameterServerClient


@dataclass(frozen=True)
class TrainResult:
    worker_id: int
    steps: int
    samples: int
    accuracy: float
    loss: float
    seconds: float


def run_baseline(
    *,
    n_samples: int = 512,
    n_features: int = 8,
    seed: int = 7,
    epochs: int = 80,
    learning_rate: float = 0.5,
    output: str | Path | None = None,
) -> TrainResult:
    features, labels = make_dataset(n_samples=n_samples, n_features=n_features, seed=seed)
    started = time.monotonic()
    weights = train_single_worker(features, labels, learning_rate=learning_rate, epochs=epochs)
    seconds = time.monotonic() - started
    metrics = evaluate(weights, features, labels)
    if output is not None:
        save_weights(output, weights, step=epochs)
    return TrainResult(
        worker_id=0,
        steps=epochs,
        samples=len(features) * epochs,
        accuracy=metrics.accuracy,
        loss=metrics.loss,
        seconds=seconds,
    )


def run_worker(
    *,
    worker_id: int,
    num_workers: int,
    host: str = "127.0.0.1",
    port: int = 50051,
    n_samples: int = 512,
    n_features: int = 8,
    seed: int = 7,
    epochs: int = 80,
    batch_size: int = 0,
    mode: str = "sync",
) -> TrainResult:
    all_features, all_labels = make_dataset(n_samples=n_samples, n_features=n_features, seed=seed)
    features, labels = shard_dataset(
        all_features, all_labels, worker_id=worker_id, num_workers=num_workers
    )
    client = ParameterServerClient(host=host, port=port)
    started = time.monotonic()
    steps = 0
    samples = 0

    for epoch in range(epochs):
        for batch_features, batch_labels in iter_batches(
            features, labels, batch_size=batch_size, shuffle_seed=seed + epoch
        ):
            step, weights = client.pull()
            grad = gradient(weights, batch_features, batch_labels)
            reply = client.push(
                worker_id=worker_id,
                step=step,
                samples=len(batch_features),
                gradient=grad,
            )
            if reply.status == "STALE":
                continue
            if reply.status not in {"QUEUED", "UPDATED"}:
                raise RuntimeError(f"unexpected push status: {reply}")
            if mode == "sync":
                target_step = step + 1
                while True:
                    current_step, _ = client.pull()
                    if current_step >= target_step:
                        break
                    time.sleep(0.002)
            steps += 1
            samples += len(batch_features)

    final_step, final_weights = client.pull()
    metrics = evaluate(final_weights, all_features, all_labels)
    return TrainResult(
        worker_id=worker_id,
        steps=final_step,
        samples=samples,
        accuracy=metrics.accuracy,
        loss=metrics.loss,
        seconds=time.monotonic() - started,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Distributed ML worker")
    parser.add_argument("--baseline", action="store_true", help="run single-process baseline")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.baseline:
        result = run_baseline(
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output=args.output,
        )
    else:
        result = run_worker(
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            host=args.host,
            port=args.port,
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mode=args.mode,
        )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
