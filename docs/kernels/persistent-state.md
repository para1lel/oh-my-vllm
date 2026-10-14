# Persistent GDN precision state

The precision scan keeps FP32 state in registers across chunks when all chunks of a sequence and head have precision route `1`.
Other sequences use the existing mixed-route loop. The route check stays on the device.
The token recurrence, eight-lane reduction order, FP32 state, and BF16 output boundaries stay equal.
The private BF16 state mirror is not read on this path, so its repeated writes are removed.

Use the [scan](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_chunk.cuh), [precision operations](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_precision.cuh), and [critical tests](../../tests/numerical/test_gdn_prefill_persistent.py).
The [source-bound summary](../evidence/persistent-state/verification.json) binds original sources, loaded modules, verification, measurements, and failed attempts.

## Operator measurements

The comparison uses three contiguous GDN input snapshots from one instrumented repeated-token model workload.
Each complete operation has three rounds of twenty interleaved pairs, with one hundred complete operations in each graph sample.
Normalization, preparation, projection, and scan enter GPU event time.
Host allocation, capture, and dispatch do not enter that time.

| Invocation | CUDA baseline | Persistent state | Decrease |
| --- | ---: | ---: | ---: |
| 0 | 8.433 ms | 8.388 ms | 0.53% |
| 15 | 16.049 ms | 14.830 ms | 7.60% |
| 47 | 16.075 ms | 14.846 ms | 7.65% |

All three confidence lower bounds on time saved are positive.
The snapshots do not identify physical layer IDs, checkpoint byte hashes, or other prompt distributions.
The comparison uses repository CUDA, not an independent long FP64 reference.
Contiguous snapshots do not test the original packed model strides.

A sixteen-lane candidate improves the first input but slows the other two. It is rejected.
Keep all its original measurements. Do not select only the improved input.

## Critical verification

Twelve external fresh-process memchecks pass with zero errors.
All 539 operator numerical cases pass independent output, state, rounding, and mutation checks without interference.
This uncommitted-source collection is diagnostic. Keep the original `passed=false` and `selected_passed=true` values.
The 105 selected GPU tests also pass.

Five new tests check precision route `1` in all active chunks.
They check almost complete cancellation with a two-token tail, padded and misaligned rows at 783/785 tokens, and live replay at 16/48 heads.
Changed sequence offsets keep an empty sequence. Each operation keeps its inputs, initial state, padding, and output guards.
The independent reference and existing numerical/state tolerances stay active.

The newly built main module passes all five memchecks and 588 numerical tests with six subtests.
Its module has the same reviewed build inputs as the external candidate, with a separate binary identity.
Mixed and changed-route tests stay active.

Six maximum-context cases pass before a CPU timeout fixture repair. Runtime sources stay equal.
The repaired fixture clears GPU visibility only for its CPU subprocesses.
The complete non-boundary repeat passes 2308 tests and 153 subtests.
This partition verification is diagnostic. Original failures stay available.

## Framework verification

The ordinary 32768-input/4096-output batch-one comparison uses the same GPU, checkpoint bytes, and release binary.
Each collection has two full warmups and five measurements, with unchanged source throughout collection.
Separate caches pass the steady-state audit.

Prefill median decreases from 1.959 to 1.899 seconds, a 3.06% improvement.
Decode medians are 33.970 and 33.947 seconds.
The candidate's phase ratios are 4.300 and 2.192. The latency checks fail the two-times gate. The spread checks pass.
This is measured prefill benefit for this workload, with no complete-matrix or final-source acceptance claim.

The first baseline and candidate collections fail the steady-state audit because shared cache files change during measurement.
The two original complete attempts stay available. Cache times alone do not identify the writer.
The next complete pair uses separate native, Triton, compiler, and driver caches.

These diagnostic results do not supply final-source acceptance.
All fifteen framework rows, capacity/service checks, and subsequent one-hour workloads await final-source verification.
Keep the numerical tolerances and theoretical equations equal.
