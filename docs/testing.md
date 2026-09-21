# Testing Guide — oh-my-vllm

## Environments and CPU checks

All commands use scripts/with-env.sh to set PYTHONPATH and CARGO_TARGET_DIR.
The framework uses conda oh-my-vllm; GPUWorker and baseline use conda vllm.

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python -m unittest discover -s tests -p test_runtime_tools.py
scripts/with-env.sh python -m unittest discover -s tests -p test_benchmark_tools.py
scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python -m unittest discover -s tests -p test_scheduler_adapter.py
scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python -m unittest discover -s tests -p test_bridge_logging.py
scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python -m unittest discover -s tests -p test_bridge_abort.py
```

Rust tests cover pool/free/hash/prefix accounting, chunked prefill, arrivals,
recompute preemption, MTP acceptance/rejection, private prefix-hit states,
speculative slot migration over multiple blocks, and complete release.
CPU benchmark tests check invalid MTP configuration and descendant cleanup.

## GPU selection and real text

Every single-GPU command waits for any idle B200 using scripts/with-gpu.sh.
Use a unique socket per run. Do not terminate unrelated GPU processes.

```bash
scripts/with-env.sh cargo build --release -p oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

Add --num-speculative-tokens 4 for MTP; add --context-repeats 100 to cross a
784-token boundary. Combine --prefix-hit with --context-repeats 100 to seed the
prompt first and require a nonzero cache hit before decoding the real answer.
--binary selects an isolated build for this text smoke. Both ordinary and MTP paths have produced coherent Chinese.
Fixed output limits deliberately ignore EOS, as does the throughput baseline.

## Actual-path FP64 reference probe

Add these environment variables to the real-text command (after with-env.sh):

```bash
env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON=/data0/shared/dongwu.chen/oh-my-vllm/tests/probe_worker.py
```

For MTP also set OH_MY_VLLM_PROBE_MTP=1 and pass --num-speculative-tokens 4.
The probe instruments a selected real FlashInfer GQA layer and target GDN
prefill/packed decode/fused MTP calls. It never substitutes production kernels.
It checks outputs and recurrent states, including each speculative state.
Required coverage categories must appear before shutdown can succeed. This is
single-request, eager diagnostic coverage; never enable it in throughput runs.

References use actual rounded inputs in CPU FP64. BF16 output checks use
atol=rtol=0.03. Recurrent state checks require normalized RMS error <=1% and
maximum absolute error <=2% of the reference peak; both metrics are logged.
Near-zero elementwise state relative errors are unstable. These checks diagnose
individual kernel/state paths, not full-model FP8 token equality. The historical
standalone SDPA test is supplementary and does not establish actual-path coverage.

## Feature combinations and preemption

The Rust bench command accepts --arrival-interval (steps between arrivals),
--prefix-hit (seed before timed batch), --warmup and --repetitions. Global
--scheduler-blocks can shrink the Rust pool for deterministic memory pressure
without changing physical GPU allocation. Verify preemptions >0 in its result;
a constrained pool alone does not prove that preemption happened.

Examples with 256 physical blocks and max-model-len 8192:

- Ordinary: scheduler-blocks 10, batch-size 2, input-len 2048, output-len 1024.
- MTP/prefix/arrivals: num-speculative-tokens 4, batch-size 2, input-len 2048,
  output-len 128, prefix-hit, arrival-interval 3.

Results report exact output counts, cache hits, preemptions, proposed and accepted
drafts. Feature smoke runs may use warmup 0/repetitions 1; they are not performance
acceptance measurements.

For request isolation through recompute, run two distinct real Chinese prompts
with staggered arrivals and a constrained pool:

```bash
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

Each request has 2048 input and 1024 output tokens. The script requires observed
preemptions, checks the beginning and tail for the correct city, and prints full
text for inspection. MTP additionally requires accepted drafts; prefix mode
requires `initial_prefix_hit_tokens > 0`, which excludes cache hits from recompute
after preemption. These are semantic smoke checks, not full-model equivalence.
Rust `run --prompt-file PATH` accepts one whitespace-separated token-ID request
per line; `--arrival-interval 3` admits the next request three steps later.
CPU cancellation tests cover running, waiting and preempted requests and the
finished-only notification that clears Python registration, adapter and Worker state.

## Matched performance acceptance

```bash
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

Repeat with --mode mtp and --mode prefix. Defaults are input32768/output4096,
1024 physical blocks, one warmup and three measurements. Each engine runs in its
own process on the same selected GPU with matching input IDs, sampling, cache
state and model configuration. Prefix seeding occurs outside the timer; both
engines must report exactly batch_size*32144 cached tokens for the default input.
MTP uses four drafts and BF16 SSM in both engines (ADR-003), and both must report
actual drafts. The timer includes request submission through completion.

Results preserve starting HEAD, dirty status/diff hash, binary hash, vLLM version,
configuration and raw repetitions. A dirty source is never labeled as clean HEAD.
Timeout cleanup terminates the entire owned engine process group. Median framework
throughput must reach >=95% in every mode/batch, with additional repetitions when
variance is material. See acceptance.md for the completed nine-row matrix and
its precise execution settings.

The driver requires a single GPU UUID and polls GPU compute clients throughout
each engine run. A process outside the owned engine process group invalidates
the measurement and causes owned-engine cleanup. Other GPU users are never killed.
This supplements the cooperative lock: unrelated jobs do not honor that lock.
Polling cannot exclude arbitrarily short interference between samples; investigate
variance and retain evidence before declaring acceptance.

Use --binary /absolute/path/to/worker for an isolated runtime experiment. The
actual executable is copied into the private benchmark directory and its hash
must match the captured identity before either engine starts. Do not modify Python
worker/vLLM sources during a run. CPU tests cover monitor ownership and rejection
of numeric GPU IDs, in addition to timeout cleanup and invalid MTP configuration.

For a CPU-affinity experiment, place taskset -c CPU before the benchmark Python
command, after the environment/GPU wrappers. Both engines inherit the same CPU
mask, which is recorded as cpu_affinity. Compare complete matching configurations;
a CPU echo/host-timing improvement alone does not establish target throughput.

## Serving CPU, client and GPU coverage

```bash
scripts/with-env.sh cargo build
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES='' scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python -m unittest discover -s tests -p 'test_serving*.py'
```

Rust tests cover incremental reasoning/XML, literal strings, request mapping and
response state. test_serving_worker.py uses the real tokenizer/XGrammar for
strict constraints, valid/invalid speculative-prefix rollback, reasoning-end
crossings, EOS, stop and UTF-8. test_serving_http.py starts the real Rust server
and a scripted CPU ZMQ worker: both API representations, usage, tool history,
stored continuation/deletion/expiry, length truncation, cancellation, concurrency,
bad requests and fatal worker errors. These are not GPU/model evidence.

See serving.md for scripts/agentic-acceptance.py. With a real UUID-pinned B200 and
MTP4 service, both OMP providers must independently execute the documented task,
return tool results and produce a file-grounded answer. Also exercise strict JSON
and tool constraints with MTP, check nonzero actual draft proposals, inspect
acceptance and latency/throughput logs, and check cleanup after cancellation.
The current CUDA/NVML host failure prevents that final acceptance; record failures
in handoff.md instead of treating scripted output as a pass.
