# Testing Guide — oh-my-vllm

All commands go through `scripts/with-env.sh`, which sets the conda environment,
`PYTHONPATH`, `CARGO_TARGET_DIR` and independent kernel cache roots. GPU commands also
go through `scripts/with-gpu.sh`, which waits for an idle B200 and pins its UUID.
Use a unique socket per run. Stop every GPU process you own when the run ends.

Known test gaps are tracked in [audit-2026-09-23.md](audit-2026-09-23.md)
(EVD-03…EVD-11). Do not cite a check listed there as evidence for the property it
fails to cover.

## Static and CPU checks

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/test.sh cpu
```

`scripts/test.sh` builds the Rust HTTP binary, then runs pytest through the
project environment. `cpu` hides CUDA and deselects cases marked `gpu`;
`full` acquires and pins an idle B200. Pytest is the single Python test entry
point. The former `unittest discover` command imported these pytest-style
modules without running them:

- `test_independent_kernels.py`
- `test_independent_decode_attention.py`
- `test_elementwise.py`
- `test_greedy_batch.py`
- `test_proposal_graph.py`
- `test_gqa_accuracy.py`

They are collected in full mode, with GPU cases explicitly marked in
`pytest.ini`/test modules. The mixed greedy-batch and validation-guard modules
mark only their GPU cases, so their CPU cases still run in `cpu` mode.

Rust tests cover:

- Pool, free-list, hash and prefix accounting
- Chunked prefill and arrivals
- Recompute preemption
- MTP acceptance and rejection
- Private prefix-hit states
- Speculative-slot migration
- Separate FA/GDN pools
- Complete release

The CPU Python tests cover:

- Batch planning and the sampler
- Serving preparation (real tokenizer/XGrammar)
- The HTTP server with a scripted worker
- The bridge
- Benchmark and measurement tooling
- The operator-comparison statistics and frozen-reference integrity

The scripted worker (`tests/fixtures/serving_worker.py`) produces fake output. It
is never model or performance evidence.

## Full GPU suite

```bash
scripts/test.sh full
```

The historical default-CUDA run on the working tree after `ef07b2b`, later
committed as `030f60f`, passed 174 tests plus 31 subtests with no skips
(`bench/baseline/2026-09-22-cuda-features.json`). It was not a separate
clean-commit run of `030f60f`.
After `96e4ecc`, the full B200 suite passed 211 tests and 32 subtests; six
context-boundary placeholders skipped. Those skips do not verify
REQ-CONTEXT-001 (EVD-07), and `d33b844` did not rerun the full GPU suite.

The operator tests compare each kernel with CPU FP64 references. They cover:

- Block-scaled FP8 and paged/grouped GQA
- Gated delta recurrence, causal convolution, RMS and partial rotary
- Fused attention preparation
- Non-contiguous pages
- 784/785-token boundaries
- Page addresses beyond int32
- Per-candidate MTP snapshots
- Graph replay

Set `OH_MY_VLLM_KERNEL_BACKEND=tilelang` to run the applicable tests against
the frozen comparison backend. `test_gqa_accuracy.py` specifically verifies
owned CUDA paged decode against CPU FP64, including a two-page 784/785 boundary;
it is skipped under the TileLang backend. The graph tests also have independent
CPU FP64 references (EVD-04/05, `4ed62ab`).

## Real text and actual-path FP64 probes

```bash
scripts/with-env.sh cargo build --release -p oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

Options:

- `--num-speculative-tokens 4`: enable MTP.
- `--context-repeats 100`: cross a 784-token boundary.
- `--prefix-hit` together with `--context-repeats 100`: seed the prompt, and
  require a cache hit.
- `--binary`: select an isolated build.

Fixed output limits ignore EOS, as the throughput workload does.

To add FP64 probes of the real kernel calls, append these variables after
`with-env.sh`:

```bash
env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON=/data0/shared/dongwu.chen/oh-my-vllm/tests/probe_worker.py
```

For MTP, also set `OH_MY_VLLM_PROBE_MTP=1` and pass `--num-speculative-tokens 4`.
`scripts/probe-independent-model.py` runs the same short eager model probe.

- **What the probe instruments:** real GQA, GDN prefill and recurrent
  verification, and MTP paged decode. It never substitutes production kernels.
- **References:** CPU FP64 references on the actual rounded inputs.
- **Tolerances:** BF16 outputs use atol=rtol=0.03. Recurrent states require
  NRMSE ≤ 1% and maximum absolute error ≤ 2% of the reference peak.
- **Coverage requirement:** the required coverage categories must appear before
  shutdown succeeds.
- **Scope:** these are eager, single-request diagnostics, not a full-model token
  equality claim. Graph capture is disabled, so diagnostic CPU copies observe real
  requests only. Near-zero elementwise state relative errors are unstable and are
  not a criterion.

Never enable probes or eager mode in performance runs.

## Feature combinations and preemption

The Rust `bench` command accepts `--arrival-interval`, `--prefix-hit`, `--warmup`
and `--repetitions`. The global `--scheduler-blocks` option shrinks the Rust pool
to force preemption. Check `preemptions > 0` in the result, because a small pool
alone does not prove preemption happened.

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

Each command runs two distinct Chinese prompts with staggered arrivals. Each
request has 2048 input and 1024 output tokens.

The script checks:

- Observed preemptions.
- The correct city at both the beginning and the tail of each output.
- Accepted drafts (MTP).
- `initial_prefix_hit_tokens > 0` (prefix mode).

It also prints the full generated text for inspection. These are semantic smoke
checks.

`run --prompt-file PATH` reads one token-ID request per line.

## Frozen-baseline performance acceptance (REQ-PERF-001/002)

**Tool.** `benchmarks/ttft.py` is the acceptance tool. It launches only this
project's worker, and on every call it enforces:

- At least 2 warmups and 5 repetitions.
- 4096 outputs per request and zero preemptions.
- Exact prefix-hit counts.
- Matching CPU affinity, GPU model and driver.
- The frozen baseline SHA.
- A steady-state compilation/capture audit.
- A throughput ratio ≥ 0.95, a TTFT ratio ≤ 1.10, and `(max-min)/median` ≤ 10%
  for both metrics on both engines.

The candidate's scheduler configuration and FA/GDN capacities are read from
worker logs and allocated device tensors, then checked against the requested
values (EVD-02); a fresh 12-row run using this path remains pending. Use
`benchmarks/ttft.py` for acceptance. `benchmarks/compare_vllm.py` now rejects
fewer than five measured repetitions or >10% spread, so its historical nine-row
baseline with three repetitions cannot pass the current gates (EVD-01).

**Baseline input.** The frozen baseline rows are embedded in
`bench/baseline/2026-09-22-refreshed-enginecore.json` under `rows[].artifact`.
Extract a row byte-identically, formatted as `json.dumps(artifact, indent=2)`
plus a trailing newline. Its SHA-256 must equal `rows[].sha256`. Then run the
row under the CPU cores recorded in `artifact.hardware.cpu_affinity`:

```bash
scripts/with-env.sh python - <<'EOF'
import hashlib, json
rows = json.load(open("bench/baseline/2026-09-22-refreshed-enginecore.json"))["rows"]
for row in rows:
    text = json.dumps(row["artifact"], indent=2) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == row["sha256"]
    open(f"/tmp/baseline-{row['label']}.json", "w").write(text)
    print(row["label"], ",".join(map(str, row["artifact"]["hardware"]["cpu_affinity"])))
EOF
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh taskset -c 8-15 \
  python benchmarks/ttft.py --baseline /tmp/baseline-ordinary-32768-1.json \
  --output /tmp/candidate-ordinary-32768-1.json
```

The defaults are `--num-gpu-blocks 4200`, `--mamba-blocks 128`, 2 warmups and
5 repetitions. The script exits nonzero if any gate fails. It runs
`target/release/oh-my-vllm-zmq-worker` under the repository root, so a custom
`CARGO_TARGET_DIR` is not supported. Prefix rows must report exactly
`batch_size * 32144` hit tokens (`(32768-1)//784*784` per request).

**Rules for a row to count as acceptance:**

- Build from a clean commit, and do not edit sources while a run is in progress.
- Retain every attempt. If an external GPU process appears, the run is invalid.
  Investigate any spread failure, then repeat the complete set.
- Never select individual repetitions.
- Never rerun the vLLM baseline without explicit user authorization.

**Where results go.** Record accepted matrices in `bench/baseline`, following
`2026-09-22-cuda-framework.json`.

## Maximum context (REQ-CONTEXT-001)

Run the six boundary rows (ordinary and MTP4, batch 1/2/4, 258048 input plus 4096
output). Global options go before the `bench` subcommand; use `--num-speculative-tokens 0`
for ordinary and `4` for MTP:

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker \
  --socket /tmp/boundary-mtp-4.ipc --num-gpu-blocks 4200 --mamba-blocks 128 \
  --max-model-len 262144 --num-speculative-tokens 4 \
  bench --batch-size 4 --input-len 258048 --output-len 4096 --warmup 0 --repetitions 1
```

Each run must show:

- No OOM.
- `preemptions == 0`.
- Recorded peak memory: the worker logs `max_reserved_bytes` at shutdown
  (`worker/model_runner.py:380-385`).

No automated test covers this yet (EVD-07).

`scripts/long-context-acceptance.py` exercises 131072-token strict-JSON requests
and prefix reuse against a running MTP4 service. Confirm MTP activity from the
server log.

## Serving

```bash
scripts/with-env.sh cargo build
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_serving*.py'
```

These tests cover:

- Rust unit tests for the reasoning/XML parser, request mapping and response
  state.
- `test_serving_worker.py`: strict constraints, speculative-prefix rollback,
  reasoning-end crossings, EOS, stop and UTF-8, using the real tokenizer and
  XGrammar.
- `test_serving_http.py`: the real Rust server with the scripted CPU worker,
  covering both APIs, usage, tool history, stored responses, truncation,
  cancellation, concurrency and error handling.

Real GPU acceptance against an MTP4 service uses these scripts. See
[serving.md](serving.md) for their commands and required log inspection.

- `scripts/serving-acceptance.py`: 12 constraint cases.
- `scripts/serving-lifecycle.py`: thinking levels, stored responses, mixed
  batches and disconnect.
- `scripts/agentic-acceptance.py`: oh-my-pi on both APIs.

Scripted output is never GPU evidence.

## Operator comparison (REQ-KERNEL-002)

`benchmarks/kernels.py` runs the formal static cases defined in
`development/kernels/cases.py`. It compares CUDA with frozen TileLang using
paired, interleaved timing. See [cuda-development.md](cuda-development.md) for
commands and the decision rule.

`test_kernel_comparison.py` tests the decision statistics.
`test_kernel_reference.py` tests the frozen-reference hashes and backend
selection.

Before timing each case, the collector checks the frozen TileLang and CUDA
operation returns and written cache/state slots, including FP8 scale layout and
per-slot recurrent-state limits. A mismatch aborts collection; the comparison
result is recorded per row. `test_kernel_output_verification.py` tests the
comparator and representative GPU fixtures (EVD-09, `ea0aa4e`). Independent
CPU FP64 tests remain separate. The current clean full matrix is pending.

## Strict Rust formatting

```bash
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh python -m unittest discover -s tests -p test_rust_style.py
scripts/with-env.sh pre-commit run cargo-fmt --all-files
scripts/with-env.sh pre-commit run rust-line-width --all-files
```

`rustfmt.toml` sets:

- Stable Rust 2024 formatting
- A 100-character width
- Unix newlines
- Multiline if/else and let/else expressions

A separate hard-width hook applies the same limit, with tabs expanded, to
macros, comments and string literals, which rustfmt may leave long. Split `json!`
bodies into fields and use `concat!` for long literals.
