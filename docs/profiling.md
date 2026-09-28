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

Use `benchmarks/ttft.py` as documented in testing.md. It emits raw repetitions,
medians, ratios, stability, configuration and identity, and exits unsuccessfully
if any gate fails. Profiling traces are not acceptance runs.

## Custom kernels

TileFoundry static cost/memory/roofline analysis is a development hypothesis,
not measured kernel time. Profile the actual model to attribute throughput/TTFT
gaps. Keep temporary operator tuning scripts and reports outside the repository.
Formal measurements audit FlashInfer, Triton, TileLang and `TVM_FFI_CACHE_DIR`
cache trees, including text artifacts; unset or missing roots fail the audit.
The repaired audit has not yet been exercised by a new full 12-row collection.
Finish all compilation before measured repetitions. Kernel faults
surface at the next host sync; use `compute-sanitizer` on a reduced case to
attribute them (audit KRN-09).

## CUDA/TileLang operator observations

Use `python -m development.kernels.observations` through the GPU/environment
wrappers to collect a formal case's two complete backend operations. See
cuda-development.md for the command. The report combines separately labeled
TileFoundry HIR estimates and Nsight counters/resources. It verifies immutable
reference/source identity and counter completeness. This profiler workflow is
separate from `benchmarks/kernels.py` CUDA Event/Graph timing and never supplies
performance acceptance samples. Raw profiler CSV remains temporary outside Git.
