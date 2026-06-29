# Distributed ML Training and Serving Platform

Minimal distributed-systems project for ML training and low-latency serving. The model is intentionally small; the real work is in the infrastructure path: sharded workers, a C++ parameter server, sync/async gradient aggregation, dynamic request batching, p99 latency metrics, and autoscaling decisions.

## What This Shows

- **C++20 systems path:** parameter server, inference server, dynamic batcher, autoscaler, socket protocol, and CMake build.
- **Distributed training:** deterministic data sharding, worker replicas, gradient push/pull, weighted aggregation, sync vs. async SGD.
- **Serving infrastructure:** trained weights loaded into a C++ inference service, concurrent requests batched by size or delay window.
- **Operational signals:** queue depth, processed batches, p99 latency, throughput, desired replica count.
- **End-to-end verification:** baseline training, 2-worker distributed training, serving load test, C++ and Python tests.

This is not a production ML framework. It is a compact, runnable system that demonstrates the infra concepts behind large-scale training and serving.

## Architecture

```text
Python workers
  -> shard data
  -> compute gradients
  -> push/pull weights

C++ parameter server
  -> owns authoritative weights
  -> averages sync gradients or applies async updates
  -> saves trained weights

C++ inference server
  -> loads trained weights
  -> batches concurrent requests
  -> reports latency, queue, and autoscaling metrics
```

## Repo Map

```text
param_server/   C++ gradient aggregation server
serving/        C++ inference server, batcher, autoscaler
worker/         Python data sharding, model, training workers
client/         inference load generator
proto/          intended gRPC API contracts
test/           C++ and Python tests
scripts/        local distributed-training launcher
```

## Run It

Build and test:

```bash
make test
```

Train a baseline model:

```bash
make baseline
```

Run 2-worker synchronous distributed training:

```bash
make train-sync
```

Serve the trained model:

```bash
make serve
```

In another terminal, generate concurrent load:

```bash
make load
```

## Verified Locally

Recent local run:

```text
baseline accuracy:        0.9727
2-worker sync accuracy:   0.9727
serving load:             200 requests, 13 batches
throughput:               ~5.5k requests/sec
client p99 latency:       ~8.45 ms
```

## Design Notes

**Synchronous aggregation:** waits for one gradient per worker at the current step, then applies a sample-weighted average. This is stable and deterministic, but one slow worker can hold the step.

**Asynchronous aggregation:** applies gradients as they arrive with a staleness bound. This improves throughput under uneven workers but can trade off convergence quality.

**Dynamic batching:** the inference server flushes when `max_batch_size` is reached or `max_delay_ms` expires. That exposes the real serving tradeoff: higher throughput at the cost of bounded wait time.

**Autoscaling signal:** the demo computes desired replicas from queue depth and p99 latency rather than CPU average, because latency-sensitive services fail first in the queue.

## Dependency Choice

The API contracts are written as protobufs in `proto/`. The runnable implementation uses a small line-oriented TCP protocol so the project builds cleanly on a laptop without requiring C++ gRPC, Python gRPC tooling, or PyTorch. The boundary is intentionally narrow so gRPC can replace the local protocol without changing the training or serving design.
