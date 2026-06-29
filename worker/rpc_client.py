"""Tiny line-oriented TCP client shared by workers and demos."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PushReply:
    status: str
    global_step: int
    pending_workers: int
    message: str


class LineClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 50051, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, command: str) -> str:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as conn:
            conn.sendall((command + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        response = b"".join(chunks).decode("utf-8").strip()
        if response.startswith("ERROR"):
            raise RuntimeError(response)
        return response


class ParameterServerClient(LineClient):
    def health(self) -> bool:
        return self.request("HEALTH") == "OK"

    def pull(self) -> tuple[int, np.ndarray]:
        response = self.request("PULL")
        parts = response.split()
        if len(parts) < 3 or parts[0] != "WEIGHTS":
            raise RuntimeError(f"unexpected PULL response: {response}")
        step = int(parts[1])
        dim = int(parts[2])
        values = np.asarray([float(value) for value in parts[3:]], dtype=np.float64)
        if len(values) != dim:
            raise RuntimeError(f"expected {dim} weights, received {len(values)}")
        return step, values

    def push(
        self,
        *,
        worker_id: int,
        step: int,
        samples: int,
        gradient: Iterable[float],
    ) -> PushReply:
        values = list(float(value) for value in gradient)
        command = (
            f"PUSH {worker_id} {step} {samples} {len(values)} "
            + " ".join(f"{value:.17g}" for value in values)
        )
        response = self.request(command)
        parts = response.split(maxsplit=3)
        if len(parts) < 3:
            raise RuntimeError(f"unexpected PUSH response: {response}")
        return PushReply(
            status=parts[0],
            global_step=int(parts[1]),
            pending_workers=int(parts[2]),
            message=parts[3] if len(parts) == 4 else "",
        )

    def save(self, path: str | Path) -> str:
        return self.request(f"SAVE {Path(path)}")


def wait_for_health(client: LineClient, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if client.request("HEALTH") == "OK":
                return
        except OSError as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc
        time.sleep(0.05)
    raise TimeoutError(f"service did not become healthy: {last_error}")
