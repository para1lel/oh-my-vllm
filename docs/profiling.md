# Profiling guide

Start with timestamped logs, then profile the part responsible for the gap.
Keep profiling separate from throughput acceptance and never commit raw traces.

## Host diagnostics

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=debug OH_MY_VLLM_LOG_LEVEL=DEBUG target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-diagnostic.ipc bench --batch-size 1 --input-len 32768 --output-len 256 --warmup 1 --repetitions 1
```

Correlate run_id and step_id. Compare Rust scheduling time/free-block count,
RPC round trip, Python decode, host execute and encode/send. A large difference
between RPC and worker execution suggests transport/wakeup overhead; it does not
by itself identify which thread or kernel caused the delay. DEBUG I/O changes
latency, so validate improvements again with normal logging.

A CPU-only echo worker can isolate transport costs, but cannot establish model
throughput. CPU affinity experiments must match baseline and framework settings;
record the CPU mask and investigate variance.

## Rust flamegraph

If cargo-flamegraph and perf are already available, run them through the same
wrappers. For symbols, use an isolated build with release debug info enabled;
do not replace a binary used by another benchmark.

```bash
scripts/with-gpu.sh scripts/with-env.sh env CARGO_TARGET_DIR=/tmp/oh-my-vllm-profile-target CARGO_PROFILE_RELEASE_DEBUG=1 cargo flamegraph --output /tmp/oh-my-vllm-flamegraph.svg --bin oh-my-vllm-zmq-worker -- --socket /tmp/oh-my-vllm-flamegraph.ipc bench --batch-size 1 --input-len 32768 --output-len 256 --warmup 1 --repetitions 1
```

Inspect scheduling/allocation, unnecessary token-history copies, block-table
updates, serialization and runtime wakeups. Treat any expected overhead budget
as a hypothesis until measured.

## Python and CUDA profiling

Wrap selected worker calls in a diagnostic harness, using the conda oh-my-vllm Python
and GPU/environment wrappers. torch.profiler with CPU and CUDA activities can
separate kernel work from host launch/wait time; cProfile only explains Python
and host-call time. A diagnostic wrapper passed as OH_MY_VLLM_WORKER_PYTHON must
accept the interpreter-style arguments: -m oh_my_vllm.worker.zmq_bridge --socket URI.

```python
from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    worker_out = worker.execute_model(sched_out)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/oh-my-vllm-trace.json")
```

Use initialization logs and the actual backend configuration to identify kernels.
The current probe wraps the independent paged-attention and GDN prefill/recurrent
calls; do not require a historical FA kernel name. Check FP8 GEMM dispatch,
CUDAGraph coverage, prefill chunk boundaries, draft counts and any runtime JIT
warnings. Different kernel paths require investigation, not an automatic claim
that one named backend is correct for every version.

## Matched acceptance

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

Repeat for mtp and prefix. The script emits JSON with raw repetitions, medians
through the ratio, configuration and identity; it exits unsuccessfully if any
row is below95%. Consult testing.md for cache/draft validation, interference
handling and variance requirements. Profiling traces are not acceptance runs.
