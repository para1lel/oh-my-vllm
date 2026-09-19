# oh-my-vllm

A Rust-first inference framework for Qwen3.5-27B-FP8 on a single B200 GPU.

The scheduler and KV cache manager are written in Rust for predictable latency
and low per-step overhead. The model runner stays in Python, delegating directly
to vLLM's `GPUWorker` so every vLLM kernel, attention backend, and CUDAGraph
capture path is reused without modification. The two sides communicate over a
ZMQ DEALER socket using msgpack-encoded messages.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  oh-my-vllm-zmq-worker (Rust binary)                        │
│                                                             │
│  ┌───────────────────┐    ZMQ DEALER (IPC)                  │
│  │  Scheduler        │◄──────────────────────►  Python      │
│  │  (FCFS, chunked   │                          zmq_bridge  │
│  │   prefill, MTP)   │                          +           │
│  ├───────────────────┤                          OhMyVllmWorker
│  │  HybridCoordinator│                          (wraps vLLM │
│  │  (FA + Mamba KV,  │                           GPUWorker) │
│  │   prefix cache,   │                                      │
│  │   LRU pool)       │                                      │
│  └───────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

Qwen3.5-27B has a hybrid layer structure: 16 full-attention layers and 48
GatedDeltaNet (Mamba) layers with `mamba_cache_mode="align"`. The KV cache
coordinator tracks two block-table groups per request and reconciles prefix
cache hits across both groups at each step.

## Quick start

```bash
# Build the Rust binary (requires Rust 1.75+ in the oh-my-vllm conda env).
cargo build --release --bin oh-my-vllm-zmq-worker

# Install the Python worker package.
pip install -e python/

# Run a single prompt (smoke test, loads the full model).
./target/release/oh-my-vllm-zmq-worker \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 4096 \
    run --tokens 1 2 3 4 5 --max-tokens 64

# Synthetic throughput benchmark.
./target/release/oh-my-vllm-zmq-worker \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 4096 \
    bench --batch-size 4 --input-len 2048 --output-len 512 --warmup 2
```

## Performance target

On bs=1/2/4, input=32768, output=4096, the framework targets ±5% of vLLM main
EngineCore throughput on a single B200. See `docs/profiling.md` for methodology
and the benchmark comparison script at `benchmarks/compare_vllm.py`.

## Tests

```bash
# Rust unit tests (scheduler + KV cache).
cargo test --workspace

# GQA accuracy test (requires CUDA).
python tests/test_gqa_accuracy.py
```

## Repository layout

```
crates/
  kv-cache/       Hybrid KV cache coordinator, block pool, prefix cache
  scheduler/      FCFS scheduler with chunked prefill and preemption
  zmq-worker/     Binary: wires Rust scheduler to Python model worker
python/
  oh_my_vllm/
    worker/
      model_runner.py   Thin wrapper around vLLM's GPUWorker
      zmq_bridge.py     ZMQ process entry point (Python side)
      spec_decode.py    MTP speculative decoding helpers
tests/
  test_gqa_accuracy.py  GQA layer accuracy vs CPU FP64 reference
benchmarks/
  compare_vllm.py       Side-by-side throughput comparison with vLLM
docs/
  design.md        Architecture, ZMQ protocol, KV cache data structures
  profiling.md     How to find bottlenecks with flamegraph and torch profiler
```
