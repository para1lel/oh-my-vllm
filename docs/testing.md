# Testing Guide — oh-my-vllm

## Independent migration checks

Use the independent environment/cache settings in development.md. CPU tests cover
`test_batch_plan.py`, `test_mtp_plan.py` and `test_independent_sampler.py`; CUDA tests
cover `test_independent_kernels.py`, `test_independent_decode_attention.py`,
`test_decode_graph.py` and `test_elementwise.py`. Actual-model probes use
`tests/probe_worker.py` as `OH_MY_VLLM_WORKER_PYTHON`, together with
`OH_MY_VLLM_ENFORCE_EAGER=1`; MTP adds `OH_MY_VLLM_PROBE_MTP=1` and
`--num-speculative-tokens 4`. Probe tolerances below are unchanged.

The independent path has passed ordinary/MTP text, MTP prefix/preemption batches,
12 Chat/Responses constraint cases and service lifecycle. The independent nine-row
frozen-baseline matrix passes. Both final oh-my-pi workflows completed with MTP4,
real source reads and tool-result follow-ups. See acceptance.md for answer-review caveats and timings.

## Environments and CPU checks

All commands use scripts/with-env.sh to set PYTHONPATH and CARGO_TARGET_DIR.
All tests and the model Worker use conda oh-my-vllm. Baseline data is frozen
and must not be regenerated. The old adapter and its tests have been removed.

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python -m unittest discover -s tests -p test_runtime_tools.py
scripts/with-env.sh python -m unittest discover -s tests -p test_benchmark_tools.py
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_*.py'
scripts/with-env.sh python -m unittest discover -s tests -p test_bridge_logging.py
scripts/with-env.sh python -m unittest discover -s tests -p test_bridge_abort.py
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
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
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
prefill/recurrent verification calls and MTP paged decode. It never substitutes production kernels.
It checks outputs and recurrent states, including each speculative state.
Required coverage categories must appear before shutdown can succeed. This is
single-request, eager diagnostic coverage; never enable it in throughput runs.
Graph capture is disabled, so diagnostic CPU copies observe real requests only.

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
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

Each request has 2048 input and 1024 output tokens. The script requires observed
preemptions, checks the beginning and tail for the correct city, and prints full
text for inspection. MTP additionally requires accepted drafts; prefix mode
requires `initial_prefix_hit_tokens > 0`, which excludes cache hits from recompute
after preemption. These are semantic smoke checks, not full-model equivalence.
Rust `run --prompt-file PATH` accepts one whitespace-separated token-ID request
per line; `--arrival-interval 3` admits the next request three steps later.
CPU cancellation tests cover running, waiting and preempted requests and the
finished-only notification that clears Python registration, sampling, MTP and Worker state.

## Frozen-baseline performance acceptance

The benchmark launches only the framework and validates an exact workload match
against `bench/baseline/2026-09-19-acceptance.json`, including its fixed SHA256.
An identical copy is allowed; changed or substitute data is rejected. There is no baseline execution
code path. Use two warmups and three measured repetitions per row:

```bash
scripts/with-gpu.sh scripts/with-env.sh taskset -c 8 python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 --warmup 2 --repetitions 3 --output /tmp/independent-ordinary-bs1.json
```

Run ordinary, mtp and prefix for bs1/2/4. The original rows used CPU cores8..16 in
that order; use each row's original mask and record it. Defaults are32768 input,
4096 output and1024 capacity units (341 logical blocks). MTP uses four proposals
and BF16 GDN state; ordinary/prefix use FP32 GDN state. Controlled prefix hits must
be exactly batch_size*32144. Prefix seeding occurs outside the timer. The timer
includes registration through completion notification. The deterministic token-ID
workload, sampling, cache policy and context limits must match frozen metadata.

The driver records original artifact hash, source HEAD/status/diff, every Python
source hash, executable hash, current independent runtime versions, GPU UUID,
CPU mask and raw measurements. It executes a hash-verified private binary copy.
Do not edit Python sources during a measurement. Final acceptance requires a clean
implementation commit; diagnostics are explicitly separate. Median throughput must
be >=95% in each of all nine rows, with follow-up measurements for material variance.

GPU clients are monitored throughout each run. External same-GPU processes invalidate
that run and trigger cleanup of the owned process group; unrelated users are never
killed. UUID pinning and cooperative locking do not replace this monitoring. Polling
cannot exclude arbitrarily short interference, so investigate unexplained variance.
Timeout cleanup includes child workers. CPU tests cover these ownership rules and
reject numeric GPU IDs, invalid MTP settings and accidental legacy interpreters.

## Serving CPU, client and GPU coverage

```bash
scripts/with-env.sh cargo build
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_serving*.py'
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
Historical V2 acceptance is recorded in acceptance.md; current independent
acceptance status is in handoff.md. Reproduce constrained cases with scripts/serving-acceptance.py and
inspect per-request MTP counters alongside saved responses. Scripted output alone
is never GPU evidence.

## Strict Rust formatting

```bash
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh python -m unittest discover -s tests -p test_rust_style.py
scripts/with-env.sh pre-commit run cargo-fmt --all-files
scripts/with-env.sh pre-commit run rust-line-width --all-files
```

`rustfmt.toml` sets stable Rust 2024 formatting, 100-character width, Unix
newlines and multiline if/else and let/else expressions. Pre-commit checks
formatting instead of silently rewriting it. A separate hard-width hook reads
the same configuration and checks tracked/unignored new `.rs` files, including
macros, comments and string literals; tabs expand using rustfmt's tab size.
`rustfmt` alone may leave `json!` bodies beyond max_width untouched. Split their
fields manually and use `concat!` to wrap long literals without changing values.
Configuration and hook changes trigger both Rust style hooks.

## Independent operator and model probes

Run these in the new conda environment. Separate FlashInfer/Triton cache roots
prevent a successful run from silently reusing artifacts compiled in the old
environment. The model probe is a short eager diagnostic, not a service or
performance acceptance run. The independent Worker is the only execution path;
the complete performance matrix passes. See acceptance.md for final service and
agentic evidence.

```bash
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p test_independent_sampler.py
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p test_serving_worker.py
scripts/with-gpu.sh scripts/with-env.sh env \
  FLASHINFER_WORKSPACE_BASE=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent \
  TRITON_CACHE_DIR=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent/triton \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MAX_JOBS=8 \
  python -m pytest tests/test_independent_kernels.py -x -q
scripts/with-gpu.sh scripts/with-env.sh env \
  FLASHINFER_WORKSPACE_BASE=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent \
  TRITON_CACHE_DIR=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent/triton \
  OMP_NUM_THREADS=1 MAX_JOBS=8 python scripts/probe-independent-model.py
```

The operator tests use CPU references for block-scaled FP8, paged GQA, gated
delta recurrence, causal convolution, RMS normalization and partial rotary
embedding. They include multiple FP8 rows, non-contiguous logical attention
pages, 784/785-token boundaries and per-candidate MTP state snapshots. They do
not replace the actual-model FP64 probes or final service/performance acceptance.
