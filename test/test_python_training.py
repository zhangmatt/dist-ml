from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from worker.data_shard import make_dataset, shard_dataset
from worker.model import evaluate, train_single_worker


def test_shards_partition_dataset() -> None:
    features, labels = make_dataset(n_samples=20, n_features=3, seed=1)
    shards = [
        shard_dataset(features, labels, worker_id=worker_id, num_workers=4)
        for worker_id in range(4)
    ]
    assert sum(len(shard_features) for shard_features, _ in shards) == len(features)
    assert sum(int(np.sum(shard_labels)) for _, shard_labels in shards) == int(np.sum(labels))


def test_single_worker_baseline_converges() -> None:
    features, labels = make_dataset(n_samples=512, n_features=8, seed=7)
    weights = train_single_worker(features, labels, learning_rate=0.5, epochs=80)
    metrics = evaluate(weights, features, labels)
    assert metrics.accuracy >= 0.9
    assert metrics.loss < 0.35


def test_distributed_training_demo_converges_when_binary_exists(tmp_path: Path) -> None:
    binary = Path("build/param_server")
    if not binary.exists():
        pytest.skip("param_server binary has not been built")

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    except PermissionError:
        pytest.skip("local TCP sockets are unavailable in this sandbox")
    finally:
        probe.close()

    output = tmp_path / "model.weights"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_distributed.py",
            "--param-server-bin",
            str(binary),
            "--port",
            str(port),
            "--workers",
            "2",
            "--epochs",
            "60",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert output.exists()
    summary = json.loads(completed.stdout)
    assert summary["workers"] == 2
    assert min(result["accuracy"] for result in summary["worker_results"]) >= 0.9
