# Performance contracts

## REQ-PERF-001: Phase latency

| Mode | Input tokens | Output tokens | Batch sizes |
|---|---:|---:|---|
| ordinary, MTP4, DSpark | 32768 | 4096 | 1, 2, 4 |
| MTP4 prefix-hit | 32768 | 4096 | 1, 2, 4 |
| MTP4 | 131072 | 4096 | 1, 2, 4 |

These combinations make fifteen rows.
Use synthetic token IDs, greedy sampling, and fixed output counts with ignored EOS.
Reset prefix reuse before each repetition.
For prefix-hit rows, seed 32144 reusable tokens per request.
Keep each full workload resident with zero recompute preemptions.

Record submission, first kept token, and last kept token for each request.
Prefill is submission to the first token. Decode is the first token to the last token.
For each repetition, use the maximum request prefill interval and maximum request decode interval.
Use each request's own boundaries when phases overlap in a batch.

Include registration, queue time, scheduling, Python, transport, and sampling.
Exclude loading, tokenization, HTTP, and warmup.


Record full batch time, cleanup time, TPS, acceptance rate, and GPU diagnostics independently.

Except for prefix-hit prefill, the wall-time median must be at most two times the theoretical lower-bound median.
The wall-time spread must be at most 10%.
Record the prefix-hit prefill latency ratio, wall time, and theoretical bound without a latency gate.
Its 10% spread check stays active. Prefix-hit decode keeps the two-times limit.

## REQ-PERF-002: Theoretical model and measurement

Use a fixed semantic DAG with data, token, layer, and persistent-state dependencies.
Independent branches can run in parallel.
CUDA stream order supplies no extra dependency.
Record effective queries, KV lengths, state access, drafts, verification, and rollback for each executed step.
Canonical rules must stay the same when kernels split or fuse.

The theoretical lower bound is the maximum of these limits:

- The semantic DAG critical path.
- Necessary HBM traffic divided by the official single-GPU bandwidth.
- Necessary work on each shared execution resource divided by its official peak rate.

Include parameter reads, scales, model computation, GPU sampling, KV/state updates, drafts, and rejected verification rows.
Use the precision and execution resource of each operation.
Count all seven DSpark backbone rows when the algorithm executes seven rows.
Keep padding, transfers that occur again and again, and unnecessary computation in overhead diagnostics.
They must not increase the theoretical bound.

Use single-GPU, dense B200 peaks: FP8 Tensor 4.5 PFLOPS, BF16 Tensor 2.25 PFLOPS, FP32 75 TFLOPS, and FP64 37 TFLOPS.
Use HBM bandwidth 7.7 TB/s.
Use published instruction rates for other known execution resources.
Record hardware identity, peak sources, operation rules, and model version.
Measured rates and empirical efficiency factors cannot increase the bound.

Use ideal cache reuse and fused intermediate storage on chip.
Use finite storage capacities to calculate minimum necessary HBM traffic.
Unpublished on-chip bandwidth is infinite in this relaxation.
Identify all GPU operations. An unidentified operation blocks acceptance.

Use two full warmups and five measured repetitions per row.
Calculate a new bound from each repetition's executed schedule.
Apply these two independent checks to prefill and decode:

```text
median(wall_time) <= 2 * median(theoretical_lower_bound)
(max(wall_time) - min(wall_time)) / median(wall_time) <= 0.10
```

For prefix-hit prefill, record the first check's result without a failure gate.
Apply the second check to each phase.

Keep complete failed sets and interrupted attempts with their reasons.
Investigate a failure. Run the affected workload again with two full warmups and five measurements.
Do not select samples.

Compilation, graph capture, interference, or source changes invalidate measured work.
Connect each attempt to source, binary, environment, CUDA module, configuration, and checkpoint identities.
Portable summaries cannot replace original records for formal verification.
