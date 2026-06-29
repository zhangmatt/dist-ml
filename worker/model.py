"""Small NumPy logistic regression model used by workers and clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


def initialize_weights(n_features: int) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return np.zeros(n_features + 1, dtype=np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_proba(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if features.shape[-1] + 1 != weights.shape[0]:
        raise ValueError("feature dimension does not match weights")
    return sigmoid(features @ weights[:-1] + weights[-1])


def predict_label(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return (predict_proba(weights, features) >= 0.5).astype(np.int64)


def gradient(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if len(features) == 0:
        raise ValueError("cannot compute a gradient for an empty batch")
    errors = predict_proba(weights, features) - labels
    grad = np.empty(features.shape[1] + 1, dtype=np.float64)
    grad[:-1] = features.T @ errors / len(features)
    grad[-1] = float(np.mean(errors))
    return grad


def binary_cross_entropy(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(predict_proba(weights, features), 1e-9, 1.0 - 1e-9)
    labels = np.asarray(labels, dtype=np.float64)
    loss = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return float(np.mean(loss))


def evaluate(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> Metrics:
    predictions = predict_label(weights, features)
    return Metrics(
        loss=binary_cross_entropy(weights, features, labels),
        accuracy=float(np.mean(predictions == labels)),
    )


def train_single_worker(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.5,
    epochs: int = 80,
) -> np.ndarray:
    weights = initialize_weights(features.shape[1])
    for _ in range(epochs):
        weights -= learning_rate * gradient(weights, features, labels)
    return weights


def save_weights(path: str | Path, weights: np.ndarray, step: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(weights, dtype=np.float64)
    with path.open("w", encoding="utf-8") as output:
        output.write(f"{len(weights)}\n")
        output.write(" ".join(f"{value:.17g}" for value in weights))
        output.write("\n")
        output.write(f"step {step}\n")


def load_weights(path: str | Path) -> tuple[np.ndarray, int]:
    with Path(path).open("r", encoding="utf-8") as input_file:
        dim = int(input_file.readline().strip())
        values = [float(value) for value in input_file.readline().split()]
        if len(values) != dim:
            raise ValueError(f"expected {dim} weights, found {len(values)}")
        step_line = input_file.readline().strip().split()
        step = int(step_line[1]) if len(step_line) == 2 and step_line[0] == "step" else 0
    return np.asarray(values, dtype=np.float64), step
