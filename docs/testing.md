# Tests and acceptance procedure

Use the environment in [development](development.md).
Set `OH_MY_VLLM_MODEL` for tokenizer or target-model integration tests.
Use [requirements](requirements.md) for active gates and [acceptance](acceptance.md) for historical results.

## CPU and GPU suites

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/check_docs.py
scripts/test.sh cpu
scripts/test.sh full
```

`scripts/test.sh` builds the test binary and supplies `OH_MY_VLLM_TEST_BINARY`.
CPU mode selects `not gpu` with GPU visibility empty.
Full mode must use a model location before selection of an idle UUID-pinned B200.
CPU tokenizer/HTTP integrations skip explicitly when the model is unset.
Pure validation tests stay active without a checkpoint.

Selected pytest arguments follow `cpu` or `full`.

Tests include scheduler transactions, cache ownership, cancellation, RPC framing, serving, inference-path FP64 references, and compiled model units.
GPU coverage includes all six maximum-context cases and changed-metadata graph replay.
Full-model drift tests use their documented model-level bounds in addition to strict operator-level tolerances.
Use scripted workers for protocol regressions and the target checkpoint for GPU/agentic acceptance.
Record warnings and skipped tests with their reasons.

## Accuracy

Use independent references with CPU FP64 computation on rounded inference inputs.
Tests must include prefill, decode, boundary writes, strided views, grouped MTP verification, and speculative rollback.
BF16 operator output uses `atol=rtol=0.03`.
Recurrent state must have NRMSE at most 1% and maximum absolute error at most 2% of reference peak.
Examine each written slot and unchanged protected state.

For fused chains, include all rounding, scale, copy, and reduction behavior.
See `tests/probe_worker.py` and the applicable kernel tests.

## Offline execution and numerical probes

Build the release binary before these commands. Set the environment and model from [development](development.md).

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

Use `--num-speculative-tokens 4` for MTP4.
Use `--context-repeats 100 --prefix-hit` to cross a 784-token boundary. The second execution must have a prefix hit.
Use `--binary` to select a different build.
Offline fixed output ignores EOS.

`run --prompt-file PATH` reads whitespace-delimited token IDs, one request per line, with at most 32 requests.
`--arrival-interval N` delays each admission by N scheduling steps.
`--prefix-hit` first seeds the prompts and then must have positive cache hits.
The global `--scheduler-blocks` limits the Rust pool. Its value must be at most the reported worker capacity.
The `bench` subcommand also has `--warmup` and `--repetitions`.

These two-request probes use 2048 input and 1024 output tokens per request:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

They must have positive preemption counts and the correct city at the start and end of each output.
MTP4 must accept drafts. Prefix mode must have positive initial prefix hits.
Examine the printed text as a semantic smoke test.
A small pool alone does not show preemption.

For inference-path numerical diagnosis, use the executable probe as the worker interpreter:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" python scripts/smoke-text.py --socket /tmp/probe-text.ipc --max-tokens 32
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" OH_MY_VLLM_PROBE_MTP=1 python scripts/smoke-text.py --socket /tmp/probe-mtp.ipc --max-tokens 32 --num-speculative-tokens 4
scripts/with-gpu.sh scripts/with-env.sh python scripts/probe-independent-model.py --max-tokens 32
```

The worker probe instruments model GQA, GDN prefill/recurrent verification, and MTP attention calls without replacement kernels.
It compares rounded inference inputs with independent references through CPU FP64 computation.
Eager mode disables capture so diagnostic copies observe inference requests.
The direct model probe supplies a short eager model diagnosis without Rust scheduling.
Keep these probes and eager mode disabled for performance acceptance.

## Release binary for acceptance

The TTFT harness reads repository `target/release/oh-my-vllm-zmq-worker`.
Set this directory explicitly. The harness must use the candidate binary from this build:

```bash
export CARGO_TARGET_DIR="$PWD/target"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

Use this build before framework, context, or service acceptance.
Set `EVIDENCE_DIR` to a writable external directory for full raw records and logs.

## Formal operator gate

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --output "$EVIDENCE_DIR/operators.json"
```

Set `EVIDENCE_DIR` to a writable evidence directory in an external directory.
The harness uses static cases from `development/kernels/cases.py`.
It verifies full output and written caches before timing.
Each case must have three rounds of twenty interleaved pairs and faster CUDA medians in all rounds.
The hierarchical-bootstrap one-sided 95% bound on time saved must be positive.

Each timing sample uses one hundred graph repetitions.
Restore persistent state between cases and rounds.
Normalization, Q/K, recurrent, and convolution verification must have one fast and zero generic host dispatches.
Do not substitute graph replay counts for host dispatch counts.
Use [kernel development](kernels.md) for provenance and profiling separation.

## Framework performance

Use `benchmarks/ttft.py` for the current twelve-row protocol.
It accepts one full baseline row from original evidence with hardware, configuration, warmups, repetitions, and measurement audit.
The refreshed historical artifact groups rows. Extract its baseline row from original evidence with all its fields unchanged.
Use the kept full record on the measurement host or collect a new matching baseline on another server.

Set `BASELINE_COLLECTION` to the full refreshed collection before this extraction.
The serialization and hash assertion keep each archived row's byte identity.
Do not supply a portable summary.

```python
import hashlib
import json
import os
from pathlib import Path

collection = json.loads(Path(os.environ["BASELINE_COLLECTION"]).read_text())
assert "artifact_kind" not in collection, "use the full original collection"
output = Path(os.environ["EVIDENCE_DIR"])
output.mkdir(parents=True, exist_ok=True)
for row in collection["rows"]:
    raw = (json.dumps(row["artifact"], indent=2) + "\n").encode()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    (output / f"baseline-{row['label']}.json").write_bytes(raw)
    print(row["label"], row["artifact"]["hardware"]["cpu_affinity"])
```

Portable summaries are rejected before comparison.

Set `EVIDENCE_DIR`, the model, and matching CPU affinity before this command:

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/ttft.py --baseline "$EVIDENCE_DIR/baseline-ordinary-32768-1.json" --output "$EVIDENCE_DIR/candidate-row.json"
```

Select each other row by its label in `--baseline`. The harness uses that row's mode, input/output lengths, and batch.
The extraction prints labels and CPU affinity. Use matching CPU affinity for collection.

Repeat for ordinary/MTP4/prefix-hit at input 32768, output 4096, batches 1/2/4, and ordinary input 131072 at the same batches.
The candidate uses 4200 historical capacity units and 128 GDN slots by default.
This gives 1400 FA slots.
Make sure that observed worker and scheduler capacities match the requested settings.
Baseline capacity must keep the same workload resident without preemption.

Match CPU affinity and GPU model, memory, and driver.
The comparator also must use matching model/configuration and pinned baseline source identity.
Use two full warmups and five repetitions.
Record each request's TTFT and fixed output count.
Apply 95% throughput, 110% TTFT, and 10% spread limits to each row.

A failed spread must have investigation and a full repeat.
Keep all rejected full attempts and interrupted runs.

The collector records source, binary, Python file hashes, package versions, loaded CUDA module, and full logs.
Log/cache audit must show steady-state measurement without observed compilation or capture.
Available logs and cache records cannot exclude silent in-memory recompilation.
Disable diagnostics that alter execution during performance runs.
Never present GPU-only prefill timing as EngineCore TTFT.

`benchmarks/compare_vllm.py` is a historical nine-row tool with an immutable original hash check.
Its early three-repetition artifact does not meet the current five-repetition rule.
It must use explicit `--baseline-json` and never starts vLLM.
Do not use its old acceptance as the current performance denominator.

## Isolated baseline collection

The baseline collector is the only authorized vLLM-dependent tool.
Use an isolated interpreter, checkout, and cache.
Do not invoke it through the project environment wrapper.
Set `BASELINE_PYTHON` and `BASELINE_CHECKOUT` to your isolated installation:

```bash
scripts/with-gpu.sh "$BASELINE_PYTHON" benchmarks/baseline/enginecore.py --checkout "$BASELINE_CHECKOUT" --model "$OH_MY_VLLM_MODEL" --mode ordinary --batch-size 1 --input-len 32768 --output-len 4096 --output "$EVIDENCE_DIR/baseline-row.json"
```

The collector must use the pinned commit and an unchanged checkout.
It records reported capacities and full measurement identity.
A new source revision must have an independently approved baseline policy.
Keep baseline caches isolated from project caches.

## DSpark comparison

Set `OH_MY_VLLM_DRAFT_MODEL` to the validated local checkpoint and build the current release binary.
Keep full attempts in an external directory. Use a new output path for each attempt:

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/speculative.py --binary target/release/oh-my-vllm-zmq-worker --raw-dir "$EVIDENCE_DIR/dspark-attempts" --output "$EVIDENCE_DIR/dspark-comparison.json" --portable-output "$EVIDENCE_DIR/dspark-comparison-portable.json"
scripts/with-env.sh python benchmarks/speculative.py --input "$EVIDENCE_DIR/dspark-comparison.json" --output "$EVIDENCE_DIR/dspark-recheck.json"
```

The collector starts one comparison worker per batch, with the same target weights and physical target caches.
Each mode loads its draft model before warmup.
Each attempt resets prefix reuse.
Use batches 1/2/4, input 32768, output 4096, synthetic IDs, greedy sampling, and ignored EOS.
EngineCore throughput includes registration, prefill, transport, and cleanup.

Each batch has three rounds.
Each round has two full warmups per mode and five measured pairs with alternating order.
Report each mode's throughput median and spread in each round.
Record the DSpark median difference from native MTP4 and the one-sided 95% hierarchical paired-bootstrap lower bound on throughput gain.
TTFT and its spread are reported.

This comparison reports DSpark throughput, spread, and confidence bounds.
Do not add a DSpark throughput or TTFT gate for this comparison.

The raw artifact keeps each attempt, log hashes, source/binary/Python identities, packages, hardware, capacities, and checkpoint file hashes.
Compare loaded draft configuration and weight hashes with the requested checkpoint bytes. They must agree.

Records include compile/capture audit, peak GPU memory, and scheduled/accepted drafts.
The paired audit checks all forty-two phase markers and each measured interval.
Subsequent warmups keep their own compilation/capture interval.
Logs and the last cache mtimes show observed activity. They cannot exclude all silent in-memory compilation.

`--dspark-confidence-threshold` selects fixed or shorter proposals and is recorded with the observed worker configuration.
Its initial setting is `0.2`. Use `0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.
Compare each setting through its scheduled/accepted draft counts and measured performance.
Acceptance rate is total accepted drafts divided by total scheduled drafts in the measured pairs.

Use `bench --prompt-file PATH` for experiments with text or tool histories different from formal inputs.
The file has one request per line, with whitespace-separated token IDs.
Its row count must equal `--batch-size`. Each row length must equal `--input-len`.
The command checks these constraints before model or GPU loading.
Keep text sources, token hashes, templates, and sampling settings with the experiment records.

`spec-bench` keeps the fixed synthetic inputs for the formal comparison.
It does not accept `--prompt-file`. Use different inputs for threshold experiments and that comparison.

Verified drafts are scheduled candidates. Returned next-step proposals have a different diagnostic counter.
The portable output is a derived summary. It cannot replace original evidence for comparison.
Keep the original twelve vLLM performance gates active on the source after implementation.

Current-source DSpark performance acceptance results are not available.

## Context and service acceptance

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/context_boundary.py --binary target/release/oh-my-vllm-zmq-worker --raw-dir "$EVIDENCE_DIR/boundary-logs" --output "$EVIDENCE_DIR/boundary.json"
```

The six rows use 258048 input and 4096 output, ordinary/MTP4, and batches 1/2/4.
Make sure that output is full, source identities match, and workers stop.
Reject OOM or preemption.
Record peak allocated/reserved GPU memory and draft counters.
Raw boundary logs must stay in an external directory.

Add the three DSpark boundary cases and keep ordinary/MTP4:

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/context_boundary.py --binary target/release/oh-my-vllm-zmq-worker --modes ordinary mtp4 dspark --raw-dir "$EVIDENCE_DIR/boundary-dspark-logs" --output "$EVIDENCE_DIR/boundary-dspark.json"
```

Supply the DSpark path through `OH_MY_VLLM_DRAFT_MODEL` or `--draft-model`.
These nine rows keep the same full-output, no-OOM, no-preemption, and memory-record contracts.

Start a dedicated MTP4 service in one terminal with the release binary and a unique IPC path.
Use Serving DEBUG logs for mixed-batch and cancellation evidence:

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug target/release/oh-my-vllm-zmq-worker --socket /tmp/service-acceptance.ipc --max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128 --num-speculative-tokens 4 serve > "$EVIDENCE_DIR/server.log" 2>&1
```

Run these commands from another terminal after service readiness.
Install the OMP executable on `PATH`, or supply its path with `--omp`.
Use the same explicit URL for all clients. The long-context tool has a different default port.

```bash
export SERVICE_URL="http://127.0.0.1:8000/v1"
scripts/with-env.sh python scripts/serving-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/constraints"
scripts/with-env.sh python scripts/serving-lifecycle.py --api chat --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle-chat.json"
scripts/with-env.sh python scripts/serving-lifecycle.py --api responses --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle-responses.json"
scripts/with-env.sh python scripts/long-context-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/long-context.json"
scripts/with-env.sh python scripts/agentic-acceptance.py --api chat --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/agentic-chat"
scripts/with-env.sh python scripts/agentic-acceptance.py --api responses --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/agentic-responses"
```

The lifecycle tool must find a shared mixed-request batch and worker cancellation/release in the supplied log.
Examine positive MTP proposals and accepted drafts in the server log.
Stop the service and its worker immediately after these tests.

The long-context gate checks strict JSON content, input more than 131072, repeated prefix reuse, and positive MTP proposals per request.
Complete OMP tasks with the target checkpoint through each API as specified in [service contracts](serving.md#verification).
The tools must read files, and the model must receive their results.
Script success alone is insufficient.

Repeat the service gates with a different DSpark worker:

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug target/release/oh-my-vllm-zmq-worker --socket /tmp/dspark-service-acceptance.ipc --max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128 --speculative-mode dspark serve > "$EVIDENCE_DIR/dspark-server.log" 2>&1
```

Supply `--speculative-mode dspark` and its `--server-log` to each acceptance script.
Use different evidence paths for each mode and API.
Run lifecycle checks with `--api chat` and `--api responses`.
The constraint and long-context scripts test the two APIs.
Each completed response ID must agree with the active mode and scheduled/accepted counts in the server log.

The OMP script forwards client HTTP through a temporary recording proxy without body changes.
It keeps client JSONL, forwarded requests, response IDs, and tool-result hashes.

The two source reads must complete without error. Their results must enter a subsequent model request before its last answer.
The last answer must cite the source files and use their results.
Examine answer meaning after the automated checks.
Stop the DSpark service and its worker after these tests.

Service scripts reserve each output file or directory before requests. An existing result path fails.
They keep `passed=false` until the applicable contracts pass.
Each received HTTP response is saved before content or log checks.
Failures keep received responses and an explicit error record.
Lifecycle streams keep the consumed SSE frames before disconnect.

## Tutorial and cleanup

Use the checks in [code-journey](../code-journey/README.md#verification).
Examine desktop/mobile screenshots and all thirteen routes after tutorial changes.

Run each GPU program through `scripts/with-gpu.sh`.
Stop all owned processes and descendants after completion or failure.
Make sure that GPU processes stopped, listeners closed, and IPC cleanup completed.
Keep only explicitly user-requested services active. Put their host details in `LOCAL.md`.
