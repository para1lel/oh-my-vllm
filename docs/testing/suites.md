# CPU and GPU suites

## CPU and GPU suites

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/test.sh cpu
scripts/test.sh full
```

`scripts/test.sh` builds the test binary and supplies `OH_MY_VLLM_TEST_BINARY`.
CPU mode selects `not gpu` with GPU visibility empty.
Full mode must use a model location before selection of an idle UUID-pinned B200.
CPU tokenizer/HTTP integrations skip explicitly when the model is unset.
Pure validation tests stay active without a checkpoint.

The FlashInfer artifact tests use the pinned cubin package without CUDA initialization.
They load checked B200 manifests, headers, and a binary kernel with downloads disabled.
They also reject incorrect artifact bytes and package version mismatches.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_flashinfer_artifacts.py -q
```

These CPU checks do not prove model timing or cold-start acceptance.
Keep new project caches for GPU acceptance. Do not install a prewarmed project or FlashInfer JIT cache.

CPU executable diagnostics tests check executable lifetime, owner intersections, pending callbacks, and explicit incomplete coverage.
They keep diagnostic snapshots without a strong executable reference.
The HTTP regression sends nonempty owner measurements through Python msgpack, the Rust decoder, and HTTP diagnostics.
It checks tuple lengths `0/1/2`, owner intersections, and zero worker restarts.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_executable_diagnostics.py tests/test_worker_lifecycle.py -q
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_serving_http.py::HttpTests::test_executable_diagnostics_cross_msgpack_rust_and_http_boundaries -q
```

Q/K GPU tests include FP64 output, FP32 edge values, and changed-input graph replay at the short dispatch boundary.
Keep their original numerical tolerances and the frozen TileLang comparison.

Each diagnostic source-audit task must have a new successful source read.
Grep output and earlier history do not satisfy the existing per-item source-read gate.

The CPU SDK fixture keeps full request bodies and the number of copies of each body.
With completed history, two automatic summary requests can have different sequences.
Other requests must keep the same sequence. The CPU fixture tests each sequence.

Scripted HTTP fixtures have a `30`-second timeout for initial service readiness.
This wait includes Python imports and the ZMQ connection. Request, recovery, and shutdown assertions keep their existing deadlines.

Selected pytest arguments follow `cpu` or `full`.
Full mode uses two new pytest processes, with the context-boundary module first.
The boundary collector's GPU ownership check runs without CUDA contexts from earlier tests.
The partition plugin keeps selected node IDs and their duplicate counts. Empty groups do not hide failures.

Each process uses the original collection scope and pytest filters before path partition.
This keeps `--ignore` and `--ignore-glob` effective for the boundary module.

Use `full --evidence-dir "$EVIDENCE_DIR/python-tests"` for a new external evidence directory.
Without this option, the launcher prints a new temporary directory and keeps its files after exit.
Each started pytest process has a log and process birth records. Keep JUnit results when pytest writes them.
Do not treat missing group reports as completed test evidence.
`tests/test_python_test_partition.py` compares collection results with the original pytest node multiset without GPU execution.

The existing worker lifecycle test checks model, callable, and graph-pool reference release on CPU.
The six target-model boundary rows check normal process exit before the unchanged worker shutdown deadline.
Removal of the final `torch.cuda.empty_cache()` call uses these behavior checks, without a test that repeats the source statement.

Tests include scheduler transactions, cache ownership, cancellation, RPC framing, serving, inference-path FP64 references, and compiled model units.
GPU coverage includes all six maximum-context cases and changed-metadata graph replay.

The benchmark ownership check queries only the selected GPU UUID through `nvidia-smi -i`.
Its query timeout stays `10` seconds. Query failures stop the collection.
Process acceptance still checks exact owned ancestry and birth identity.
CPU regressions check selected-device isolation, foreign processes, birth reuse, and query failures.

Linux can return `EINVAL` when a process is reaped after kernel PID lookup and before pidfd binding.
The [Linux 6.8 implementation](https://github.com/torvalds/linux/blob/v6.8/kernel/fork.c#L2178) checks for a thread-group leader after lookup.

Collection checks the process birth again after this error.
The observer does not register a missing or changed birth. The same birth still causes failure.
Other `pidfd_open` errors keep their previous handling.
CPU tests check child exit, PID reuse, and these failure conditions.

`tests/test_graph_capture.py` checks effective allocator settings and workspace lifetime fences on CPU.
`tests/test_graph_memory.py` checks allocator release with live shared graphs and outputs.
Its temporary allocation is more than `total_size - allocated_size` for each existing default-pool segment.
The allocator can release all storage in the new segment while other segments keep live tensors.
A new case keeps a live tensor and at least `128` MiB of free storage in the same default-pool segment.
The owner, pool, zero-eviction, `32` MiB release, and exact replay checks stay active.

`tests/test_dspark_context_graph_gpu.py` includes the production 29-by-25600 BF16 context projection.
Its warmup uses the capture stream before pool capture.
It resets unrelated GEMM graphs, applies allocator pressure, and replays changed inputs.
It checks all five cache outputs at `atol=0.03`, `rtol=0.03`, exact untouched-row canaries, and unchanged workspace budgets.

Full-model drift tests use their documented model-level bounds in addition to strict operator-level tolerances.
Use scripted workers for protocol regressions and the target checkpoint for GPU/agentic acceptance.
Record warnings and skipped tests with their reasons.
