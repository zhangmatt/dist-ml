# codex.md — Distributed ML Training & Serving Platform

## Project Goal
Build a system that (1) trains a model across multiple workers using **data-parallel** distribution with a **parameter server** for gradient aggregation, and (2) serves the trained model behind a **low-latency inference service** with request batching and autoscaling. C++ for the performance-critical serving/aggregation path; Python for training orchestration and model code.

## Scope Warning (read first)
This is the largest of the four projects. Do NOT try to build a production ML framework. Build a *minimal but real* end-to-end system that demonstrates the distributed-systems concepts: parameter server, gradient aggregation, sharded data loading, and a batched serving layer. A small model (e.g., logistic regression or a small MLP on MNIST/CIFAR-scale data) is fine — the ML is the vehicle; the *distributed systems* is the point. Build the other three projects first if time is tight.

## Tech Stack
- **Training orchestration / model:** Python (PyTorch for the model + autograd; you implement the *distribution* layer, not the autograd)
- **Parameter server + serving core:** C++20 for the aggregation server and the inference server (the latency-critical pieces)
- **RPC:** gRPC + Protocol Buffers (Python workers ↔ C++ parameter server; clients ↔ C++ serving layer)
- **Optional:** Ray for the worker orchestration alternative; CUDA only if you have a GPU (otherwise CPU is fine and simpler)
- **Build:** CMake for C++; a Python package for workers; docker-compose to launch PS + N workers + serving
- **Testing:** GoogleTest (C++), pytest (Python), an integration test that trains a tiny model to convergence across 2+ workers

## Repository Structure
```
dist-ml/
  proto/
    ps.proto                 # Push(gradients), Pull(weights)
    serving.proto            # Predict(request) -> prediction
  param_server/              # C++
    ps_server.{h,cpp}        # holds authoritative weights; aggregates gradients
    aggregator.{h,cpp}       # sync (all-reduce-style) or async accumulation
    main.cpp
  worker/                    # Python
    train_worker.py          # loads a data shard, computes gradients, push/pull
    data_shard.py            # deterministic dataset sharding across workers
    model.py                 # small MLP / logreg in PyTorch
  serving/                   # C++
    inference_server.{h,cpp} # loads weights, serves Predict
    batcher.{h,cpp}          # dynamic batching of incoming requests
    autoscaler.{h,cpp}       # scale replicas on queue depth / latency
    main.cpp
  client/load_gen.py         # generates inference load to exercise batching/autoscale
  test/
  CMakeLists.txt
  docker-compose.yml
  README.md
```

## Core Architecture
- **Data-parallel training:** the dataset is sharded deterministically across N workers. Each worker holds a full model replica, computes gradients on its shard, and pushes them to the parameter server; it pulls back updated weights for the next step.
- **Parameter server:** a C++ service holding the authoritative weights. It aggregates pushed gradients (synchronous: wait for all workers each step → equivalent to a larger batch; or asynchronous: apply as they arrive → faster but staler). Make the mode a flag and be able to discuss the tradeoff.
- **Serving:** a C++ inference server loads trained weights and answers Predict RPCs. A **dynamic batcher** coalesces concurrent requests into a single batched forward pass (throughput vs. latency tradeoff via a max-batch / max-delay window). An **autoscaler** adds/removes serving replicas based on queue depth or tail latency.

## Implementation Phases (BUILD IN THIS ORDER)

### Phase 1 — Single-Worker Training Baseline
- Python: small model, full dataset, trains to a known accuracy. This is the correctness baseline.
- **Acceptance test:** single-process training reaches target accuracy.

### Phase 2 — Parameter Server + Two Workers (Synchronous)
- C++ PS exposing Push/Pull over gRPC; holds weights.
- Two Python workers, each on half the data, push gradients; PS averages and updates; workers pull.
- **Acceptance test:** 2-worker synchronous training reaches ~the same accuracy as the baseline (validates correct gradient aggregation).

### Phase 3 — Asynchronous Mode + Scaling Workers
- Add async aggregation (apply gradients as they arrive, with a staleness bound).
- Scale to N workers; measure training throughput (samples/sec) vs. worker count.
- **Acceptance test:** throughput rises with workers; document the sync-vs-async accuracy/throughput tradeoff.

### Phase 4 — Inference Server + Dynamic Batching
- C++ serving loads final weights; Predict RPC.
- Dynamic batcher: accumulate requests up to max-batch-size or max-delay, run one forward pass, scatter results.
- **Acceptance test:** under concurrent load, batching increases throughput at a bounded latency cost (show the curve).

### Phase 5 — Autoscaling
- Monitor queue depth / p99 latency; spin serving replicas up/down; load-balance across them.
- **Acceptance test:** load_gen ramps traffic → replicas scale up → latency stays bounded → traffic drops → replicas scale down.

## Key Design Decisions & Interview Defense
- **Why data-parallel (not model-parallel)?** The model fits on one worker; data-parallel is the standard, simplest scaling axis — replicate the model, shard the data, aggregate gradients. Model parallelism is for models too big for one device; mention it as the alternative and when you'd use it.
- **Parameter server vs. all-reduce?** PS is simple and centralized (easy to reason about, but the PS can bottleneck/SPOF); ring all-reduce has no central bottleneck and is what modern frameworks use. Be able to contrast them and say why you chose PS for clarity.
- **Synchronous vs. asynchronous SGD:** sync = deterministic, equivalent to large-batch, but as slow as the straggler; async = higher throughput but stale gradients can hurt convergence. This is the key training-systems tradeoff — know it cold.
- **Why dynamic batching in serving?** GPUs/CPUs are far more efficient on batched matrix ops; batching trades a little per-request latency (the wait window) for large throughput gains. Explain the max-batch / max-delay knobs.
- **Autoscaling signal choice:** queue depth and tail latency are better scaling signals than CPU% for latency-sensitive serving — explain why average CPU misleads under bursty load.
- **Consistency of weights in serving:** how you roll out new weights (atomic swap / versioned models) without dropping in-flight requests.

## Definition of Done
- End-to-end: distributed training produces weights → serving answers batched requests → autoscaler responds to load.
- README documents the sync/async and batching tradeoffs with measured numbers.
- You can whiteboard data-parallel training with a parameter server and explain the batching latency/throughput curve.

## Honest Note
If time is short, it is completely acceptable to ship Phases 1–4 (training + PS + batched serving) and describe autoscaling as in-progress. A smaller system you fully understand beats a sprawling one you can't defend.