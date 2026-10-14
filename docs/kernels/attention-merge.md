# Attention split merge

The [owned attention source](../../python/oh_my_vllm/kernels/cuda_backend/operators/decode_attention.cuh) divides the 128-split weighted sum across eight warps and four value tiles.
The maximum and weight denominator keep the previous reduction order. The FP32 value sum uses a different grouping.
The final BF16 boundary and numerical tolerances stay equal. Empty heads produce exact zero.

## Numerical checks

The [merge contract test](../../tests/numerical/test_attention_merge_replay.py) compares the complete weighted mean with an independent FP64 reference.
Five shapes test cancellation, one dominant split, mixed empty splits, and empty heads.
Checks cover fresh writes, output guards, immutable inputs, and changed inputs on a non-default stream during graph replay.

Main passes five fresh memchecks, 115 selected GPU tests, and all 539 operator diagnostic cases.
CPU passes 1682 tests and 153 subtests without skips. The full numerical and framework reports keep `diagnostic=true` and `passed=false`.
The [evidence summary](../evidence/attention-merge/verification.json) binds sources, modules, original results, and independent reviews.

## Complete operation comparison

Each sample executes the partial attention and merge with eight independent KV caches larger than L2.
The context is 32768 tokens, with 24 query heads and four KV heads. Batches are one, two, and four.
The full FP64 attention reference checks all eight inputs before measurement and after changed-input replay.
Destination guards, full writes, input bytes, and caller-stream behavior also pass.

Three rounds use twenty interleaved pairs with one hundred complete operations per sample.
Mean gains are 3.557, 4.912, and 10.629 microseconds. One-sided 95% bootstrap lower bounds are 3.544, 4.882, and 10.552 microseconds.
Mean latency decreases are approximately 9.13%, 7.24%, and 8.82%.
These device intervals exclude query preparation and host scheduling.

## Comparison module isolation

The first two-module batch-four memcheck fails at a CUDA launch. Its original log and empty result file stay available.
Symbol evidence identifies a shared GNU-unique function-local flag. Its once-only configuration can skip shared-memory opt-in for the second module.

A fresh experimental build adds `-Xcompiler=-fno-gnu-unique` to the two comparison libraries.
Read-only address checks prove distinct flags. All 482 device function bodies and resources equal the corresponding original build.

All six fresh whole-operation memchecks then pass, across the two PDL settings.
Production keeps its original compiler flags. Main uses one module and passes its own fresh memory and numerical checks.
This experiment does not claim a production two-module defect.

## Framework scope

Same-GPU ordinary 32768/4096 batch-one collections each use two warmups and five measurements, with separate caches and the same release binary.
Source and checkpoint identities match before and after collection. Steady-state audits pass without preemption or process leaks.

Prefill medians are 1.896502455 and 1.901747655 seconds. Decode medians are 34.079205395 and 33.769918883 seconds.
Candidate ratios are 4.305 and 2.181. The two latency gates fail. All spread checks pass.

The observed decode decrease is 0.908%, with a 0.277% prefill increase. These sequential sets do not prove whole-model causal attribution.

The full fifteen-workload, 27-latency-gate collection and the two final-source book soaks are pending.
Operation gains do not supply framework acceptance or a critical-path saving.
