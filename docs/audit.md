# Audit status and open work

The source audit at `030f60f` included project Rust, Python, kernels, tests, scripts, benchmarks, and development tools.
It identified 51 findings.
The closed-finding index keeps useful contracts and fix identities.
Daily remediation narratives stay in Git history.
Performance evidence applies to its measured source, as specified in [acceptance](acceptance.md).

## PY-04: Graph memory and shape changes

Status: partial/open, severity P2.
Graph families use bounded graph caches with shared pools, admission, eviction, free-memory guards, and capture restoration.
The resident-graph 36k-to 40k transition completed with less than 4 GiB free memory.
Same-shape 262144 ordinary/MTP4 executions also completed without graph eviction or recapture.
These probes do not include cross-shape 262k eviction/recapture or all late-capture memory conditions.

A future test must force eviction and recapture with measured memory pressure.
Record before/after keys, counters, persistent-state equality, peak memory, and process cleanup.
Review the workload before implementation to keep in the supported memory envelope.

## PY-06: Host synchronization

Status: partial/open, severity P2.
Metadata and readback are batched, but dependent host waits stay in the execution chain.
The pinned-D-to-H ABBA experiment found no material full-call gain.
Keep the current validated path until a full-step overlap candidate shows correctness and measured benefit.
A trace or asynchronous API call alone does not show overlap or throughput improvement.

## KRN-06: Streaming store candidate

Status: open with a rejected optimization, severity P2.
The old KRN-06 candidate used streaming stores for native residual output.
Its measurements showed no stable net full-call gain. The project rejected that proposed replacement.

Current guarded store paths for native residual and DSpark hidden RMS stay in use.
Make a decision about a new candidate only with a new hypothesis and full-operation measurements.

## Other evidence limits

Dynamo's 4096 limits and 256-variant warnings do not show bounded compiler memory with indefinite shape changes.
The log/cache audit cannot exclude silent in-memory recompilation.

Phase, operator, boundary, and service evidence keeps its measured source scope.
The [acceptance index](acceptance.md) keeps source scope and all original failed records.
The fifteen-row phase-latency collection passed. See the acceptance index for its source and review scope.

## Closed finding index

These 48 findings have recorded fixes.
Their current tests and historical acceptance have different scopes.
Use the cited commits for implementation context and [testing](testing.md) for current verification.

| ID | Severity | Closed contract | Fix commits |
|---|---|---|---|
| SRV-01 | P0 | Request failure isolation | `d0d286a`, `96e4ecc` |
| SRV-02 | P0 | Worker lifetime | `487f8de` |
| SRV-03 | P0 | HTTP error classes | `d0d286a`, `d33b844` |
| SRV-04 | P0 | Length output and tool completion | `d0d286a` |
| EVD-01 | P1 | Repetition and spread gates | `487f8de` |
| EVD-02 | P1 | Observed configuration | `487f8de` |
| KRN-01 | P1 | Grouped query bounds | `eb5ff62`, `96e4ecc` |
| KRN-02 | P1 | KV slot bounds | `96e4ecc` |
| KRN-03 | P1 | Empty split merge | `96e4ecc` |
| KRN-04 | P1 | Flat index bounds | `eb5ff62`, `952a01d`, `312c54b` |
| KRN-05 | P1 | Q/K contracts | `487f8de` |
| PY-01 | P1 | FA page ownership | `487f8de` |
| PY-02 | P1 | FP8 scale packing | `eb5ff62` |
| SCH-01 | P1 | Transactional output validation | `487f8de` |
| SCH-02 | P1 | Reference and free-queue guards | `487f8de` |
| SCH-03 | P1 | Unaligned prefill progress | `42197f7` |
| SCH-04 | P1 | Capacity admission | `487f8de` |
| SCH-05 | P1 | Interior null blocks | `42197f7` |
| SRV-05 | P1 | RPC correlation and timeout | `96e4ecc` |
| SRV-06 | P1 | Fire-and-forget and preparation isolation | `d0d286a`, `96e4ecc` |
| SRV-07 | P1 | Service output ownership | `96e4ecc` |
| PY-03 | P2 | Graph admission and eviction | `5772834`, `2837c49` |
| PY-05 | P2 | Compact MTP metadata | `17413e5` |
| PY-07 | P2 | Cached penalties and top-k selection | `503ea27` |
| SRV-08 | P2 | Background preparation | `efdef09` |
| SRV-09 | P2 | Incremental tool parsing | `c082937` |
| SRV-10 | P2 | Timed stream backpressure | `b1761e9` |
| SCH-06 | P2 | Priority preemption | `67deb85` |
| SCH-07 | P2 | Incremental prefix hashing | `5a3b703` |
| KRN-07 | P2 | Observable alignment copies | `f4aacca` |
| KRN-08 | P2 | Fast/generic dispatch counters | `40e3e57` |
| KRN-09 | P2 | CUDA error observation | `94c3473` |
| KRN-10 | P2 | Launch configuration and page-table reuse | `37f8cc9`, `231f066` |
| EVD-13 | P2 | Native cache audit | `895d57b`, `487f8de` |
| EVD-03 | P3 | Unified pytest discovery | `6da6427` |
| EVD-04 | P3 | Inference-path GQA reference | `4ed62ab` |
| EVD-05 | P3 | Independent graph expectations | `4ed62ab` |
| EVD-06 | P3 | Pinned reference identity | `895d57b` |
| EVD-07 | P3 | Context and HTTP evidence | `bd8e21e`, `8dfc97b` |
| EVD-08 | P3 | Cancellation ownership tests | `6da6427` |
| EVD-09 | P3 | Full operator output gate | `ea0aa4e` |
| EVD-10 | P3 | Deterministic fixture seeds | `6da6427` |
| EVD-11 | P3 | Text-cache and root audit | `487f8de` |
| EVD-12 | P3 | Compiler/module provenance | `f34b661`, `9648012` |
| MNT-01 | P3 | Live CUDA factory arguments | `be84176` |
| MNT-02 | P3 | Dtype, epsilon, and empty-input contracts | `04540bd` |
| MNT-03 | P3 | Numerical dispatch contracts | `0394727` |
| MNT-04 | P3 | Protocol and shutdown maintenance | `96e4ecc`, `ff1f944` |


## Change rules

Add a failing regression where practical, then make the fix and a review by another agent.
Keep numerical tolerances and active gates unchanged.
Do affected formal operator cases again after device implementation changes.
Do affected framework rows again before you claim continued performance acceptance.
Update this status and [handoff](handoff.md), with open limitations.

## Owned CUDA operator audit

Opus 5.5 examined owned CUDA entry points, Python wrappers, semantic operations, and production callers.
A different agent checked and corrected the audit items against source guards, dispatch, graph restoration, and measurements.
The [operator audit](../bench/evidence/2026-10-10-owned-cuda-audit.json) gives the full inventory, source hashes, corrections, and verification limits.

This audit uses `a6b9bd2` and `a5cc0ef`, before the CUDA header split.
Checks of the current header layout have different records.

The native CUDA module has 25 FFI exports. The `groupwise_fp8` module has one other export.
`variant_launch_count` counts host dispatches. The other 25 exports do GPU computation.
The 27 `__global__` definitions in `operators/` do not include CUTLASS template instantiations.
The audit counts semantic operations, FFI exports, kernel definitions, and compiled variants in different groups.

| Item | Contract | More verification |
|---|---|---|
| FP8 GEMM workspace | The GEMM code checks the necessary workspace size before initialization and execution. | Keep the capacity guard after configuration changes. |
| FP8 scales and accumulation | Selected projections with 624 through 2496 rows use K-major scales. Their mainloop has FP32 accumulation in K-group sequence. | Examine the tile, number of pipeline stages, and scale layout together. |
| GDN copy and PDL | Selected views use the value copy with input strides before chunk GDN prefill. The consumer call supplies no PDL launch parameter. | Examine the two launches before a test that changes the trigger position. |
| DSpark draft attention | All seven draft rows read the full committed context plus all seven draft KV rows. The draft block uses bidirectional attention. | Keep the reference mask and the mathematical attention scale `1/sqrt(128)`. |
| Owned decode attention | Four ungrouped queries use two shared-memory buffers for the next tile transfer. Other selected paths use one buffer. | Compare full attention operations before a replacement. |
| Graph and stream lifetime | The capture code restores persistent state. The caller stream records a dependency on the GDN branch in `finally`. | Keep `PY-04` until the memory-pressure test passes. |
| Dispatch counts | The counter counts host calls, not graph replays. | Keep verification and replay counts in different groups. |

The different agent checked the 12 initial audit items and found no new production defect.
The corrected audit shows source guards, incorrect statements, external candidates, and performance changes that have no applicable measurement.
Production selection stays the same after this audit.

### Operator tuning

1. Use source instruction observations for 624 through 2496 FP8 rows before pipeline stage or register tests.
2. Measure full RMS/SiLU, quantization, scale handling, and GEMM operations for a packed FP32 candidate.
3. Examine the GDN producer and consumer launch contracts together before a PDL integration change.
4. Check each candidate with the source for this revision in operator, framework, capacity, and service tests before selection.

The external packed candidate groups two FP32 values in `float2` for `__ffma2_rn`.
It had equal outputs in 18 cases. Ten speed tests passed and eight failed.
The diagnostic selection has seven GEMM shapes and nine full operations that used less time in the measurements.
This diagnostic selection uses a GEMM shape only if all its measured output paths pass.
These paths include output without a gate and output with a gate.

The measured rows are 624, 1248, and 2496. These tests do not cover two-SM tiles or large MN scales.
Before production selection, check these selected diagnostic results with the source for this revision.

Profiler measurements identify the source, shape, kernel, and replay.
Eligible warps have units of warps for each active scheduler cycle.
Scoreboard samples do not give an instruction's percentage of execution time.
Because cache can supply TMA requests, TMA traffic and HBM traffic are different measurements.
Low occupancy, without other measurements, does not identify the latency cause or show that a different tile is faster.
