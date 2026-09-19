# Profiling guide

Two complementary tools: `cargo flamegraph` for Rust scheduler / ZMQ overhead,
and `torch.profiler` for GPU kernel timing on the Python side.

## Rust: flamegraph

```bash
# Install once.
cargo install flamegraph

# Build with debug symbols in release profile.
# Add to Cargo.toml [profile.release]: debug = 1

cargo flamegraph --bin oh-my-vllm-zmq-worker -- \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 4096 \
    bench --batch-size 4 --input-len 2048 --output-len 512 --warmup 2

# Opens flamegraph.svg in the current directory.
```

Expected hot spots in a healthy run: almost nothing. The Rust scheduler and
ZMQ serialisation should take < 1 ms per step on a B200. If `schedule()` or
`allocate_slots` shows up significantly, look at:

- `FxHashMap` lookup in `block_tables` — should be O(1) but watch for rehash.
- `Vec::extend_from_slice` in block table updates — allocation pressure if
  block tables grow unbounded.
- `msgpack::encode` / `decode` — verify `rmp-serde` is not cloning large token
  id vecs unnecessarily.

## Python: torch profiler

Wrap one benchmark iteration with the profiler to see GPU kernel breakdown:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    with record_function("execute_model"):
        worker_out = worker.execute_model(sched_out)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/trace.json")
# Open trace.json in chrome://tracing or Perfetto UI.
```

Key things to check:

- **Attention kernels:** Qwen3.5 should dispatch 16 calls to `tml_fa4` (full
  attention) and 48 calls to the FlashInfer GDN kernel. If you see
  `scaled_dot_product_attention` with a PyTorch-native fallback, the backend
  selection is wrong — check that `vllm.attention.backends` resolves correctly
  for the model config.
- **GEMM:** should hit DeepGEMM or CUTLASS for FP8 matmuls. A fallback to
  `torch.mm` in FP16 is a sign the FP8 quantisation path did not activate.
- **CUDAGraph replay:** decode steps should show a single `cudaGraphLaunch`
  per step rather than hundreds of individual kernel launches. If not, confirm
  `enforce_eager=False` and that the graph was captured during warmup.

## Comparing against vLLM baseline

Use `benchmarks/compare_vllm.py` for a side-by-side run:

```bash
# Both processes should run on the same GPU and same model weights.
python benchmarks/compare_vllm.py \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --batch-sizes 1 2 4 \
    --input-len 32768 \
    --output-len 4096
```

The script runs vLLM first (using its offline `LLM` API) then oh-my-vllm, and
prints a comparison table with throughput and the percentage difference. The
±5% target is measured on output tokens per second.

## Interpreting a gap

If oh-my-vllm is slower than vLLM:

1. Check the flamegraph for Rust-side overhead first — it should be negligible.
2. Run the torch profiler on both and diff the CUDA kernel timings. Equal
   kernel times with worse wall-clock usually means ZMQ or Python serialisation
   overhead.
3. Unequal kernel times (especially slower attention) means the backend
   selection or CUDAGraph coverage differs. Compare the profiler output for
   the attention kernels specifically.
4. If the gap is in prefill (long input), check chunked prefill: vLLM's default
   `max_num_batched_tokens` may differ from ours. Match the value and re-run.
