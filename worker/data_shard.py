"""Deterministic data generation and sharding for distributed workers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def make_dataset(
    *,
    n_samples: int = 512,
    n_features: int = 8,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")

    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float64)
    true_weights = np.linspace(-1.5, 1.5, n_features, dtype=np.float64)
    logits = features @ true_weights - 0.15
    logits += rng.normal(0.0, 0.15, size=n_samples)
    labels = (logits >= 0.0).astype(np.int64)
    return features, labels


def shard_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    worker_id: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must be in [0, num_workers)")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    indices = np.arange(len(features))[worker_id::num_workers]
    return features[indices], labels[indices]


def iter_batches(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 0,
    shuffle_seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if batch_size <= 0 or batch_size >= len(features):
        yield features, labels
        return

    indices = np.arange(len(features))
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield features[batch_indices], labels[batch_indices]
