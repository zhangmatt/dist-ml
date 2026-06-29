"""Launch a local C++ parameter server and N Python training workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.rpc_client import ParameterServerClient, wait_for_health
from worker.train_worker import run_worker


def _run_worker_entry(kwargs: dict) -> dict:
    return asdict(run_worker(**kwargs))


def _default_binary() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / "param_server"


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run distributed training against C++ PS")
    parser.add_argument("--param-server-bin", default=os.environ.get("PARAM_SERVER_BIN"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--output", default="build/model.weights")
    args = parser.parse_args()

    binary = Path(args.param_server_bin) if args.param_server_bin else _default_binary()
    if not binary.exists():
        raise SystemExit(f"missing parameter server binary: {binary}; run cmake --build build")

    command = [
        str(binary),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dim",
        str(args.n_features + 1),
        "--lr",
        str(args.learning_rate),
        "--workers",
        str(args.workers),
        "--mode",
        args.mode,
    ]

    server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_health(ParameterServerClient(args.host, args.port), timeout=5.0)
        worker_args = [
            {
                "worker_id": worker_id,
                "num_workers": args.workers,
                "host": args.host,
                "port": args.port,
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "mode": args.mode,
            }
            for worker_id in range(args.workers)
        ]
        started = time.monotonic()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_run_worker_entry, worker_args)

        client = ParameterServerClient(args.host, args.port)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        client.save(args.output)
        elapsed = time.monotonic() - started
        summary = {
            "mode": args.mode,
            "workers": args.workers,
            "epochs": args.epochs,
            "seconds": elapsed,
            "samples_per_second": args.n_samples * args.epochs / elapsed,
            "weights": args.output,
            "worker_results": results,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        server.terminate()
        try:
            server.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=2.0)
        stderr = server.stderr.read() if server.stderr is not None else ""
        if server.returncode not in {0, -15, -9} and stderr:
            sys.stderr.write(stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
