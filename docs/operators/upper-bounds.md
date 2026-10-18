# Physical upper bounds

## Scope

No qualifying physical upper bound below 0.50 is available for any operator family and execution mode.
The ordinary gate is 50% in eager execution and CUDA Graph replay.
A 75%-of-U alternative applies only after complete physical analysis proves that 50% is unreachable.
Full numerical coverage and measured family efficiency of at least 0.75U are also necessary.
This alternative supplies no operator acceptance by itself.
The [24 HTTP configurations](../performance.md) each have a hard 50% gate.

The [workload](../../development/kernels/workload.py#L323)<!-- source: def modeled_cases(profiles=PROFILES): --> supplies 75 cases for seven serving boundaries.
The two execution modes give 14 family-mode decisions.
Keep the original work, precision, named tolerances, cold-weight policy, and serving boundaries from the [evaluation procedure](evaluation.md).

## Family calculation

For case i, R_i is its fixed roofline time, T_i is its measured duration, and calls_i is its invocation count.
The [work calculation](../../development/kernels/work.py#L50)<!-- source: def work_for(shape, *, queries=None, state_writes=None): --> defines useful computation and necessary boundary traffic.
The [aggregator](../../development/kernels/roofline.py#L61)<!-- source: def expected_efficiencies(observations, required_cases, source_id): --> uses:

`p_i = calls_i / sum_j calls_j`

`eta_family = sum_i p_i * (R_i / T_i)`

The sum in p_i covers the same operator family. A ratio of summed times is a different metric.

A physical proof must give valid positive time floors for all cases and the selected execution mode:

`T_i >= L_i > 0`

`eta_family <= U_family = sum_i p_i * R_i / L_i`

The choice `L_i = R_i` gives `U_ref = 1` in the reference accounting model.
This choice is not a physical certificate. It does not prove a universal physical ratio of at most one.
Do not clamp a physical `R_i/L_i` at one.

Use the fixed dense B200 peaks and [intermediate precision](model.md#intermediate-precision).
Tensor precision times add on their shared resource. Independent resource floors combine with `max`.
Only proof of mandatory serial execution can justify adding complete memory and compute times.
Padding, recomputation, and unnecessary copies cannot increase useful work.
All necessary launches and temporary traffic are part of the measured operation.

## GDN state path

The [paired state recurrence](../../python/oh_my_vllm/kernels/cuda_backend/gdn_value_pair.cuh#L64)<!-- source: gdn_value_pair_kernel(const __nv_bfloat16 *qkv, __nv_bfloat16 *pool, const int *starts, --> has this longest path in its original rounded FP32 graph:

| Operation | Consecutive rounded nodes |
|---|---:|
| Decay multiply | 1 |
| Ordered recall FMA | 4 |
| Original reduction levels | 5 |
| Subtraction | 1 |
| Beta multiply | 1 |
| State-update FMA | 1 |
| Total | 13 |

RN means round to nearest, with ties to even.

The 13 nodes give source dependency depth, not a universal native instruction count or duration.
Independent batch, head, value-row, and packed-component chains do not multiply this path.
Output dot products, head RMS, and output gates do not feed the next state.
Prefill Q/K normalization and beta/decay are computed before the carry.
The [prepared operands](../../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L203)<!-- source: __device__ inline void gdn_prefetch_rows(unsigned char *storage, const __nv_bfloat16 *qkv, --> already supply scaled Q and gate coefficients.
BF16 checkpoint stores do not replace the live FP32 carry or prove a complete HBM stall before the next token.

Let q_i be the query length and tau a certified minimum delay for each rounded parent edge.
A conditional span is:

`span_i = 13 * q_i * tau`

The longest modeled query has 32144 tokens and 417872 nodes on this path.
No certified edge delay or clock cap supports a physical duration for the path.
A proof must also address legal alternate realizations and input-dependent shortcuts.

The workload gives q1 probability 0.737516005121639, q5 probability 0.24583866837387963, and prefill probability 0.016645326504481434.
The q1 share contains embedding cases. A restriction on prefill alone leaves the reference remainder 0.9833546734955185.
That remainder is not an attainable-efficiency claim.

The conditional equation for all 24 GDN cases is:

`sum_i p_i * R_i / max(R_i, 13*q_i*tau) = 0.50`

Its solution is tau=1.96191198467 microseconds per rounded edge.
This is a required proof threshold, not a measured or assumed hardware latency.
The equation supplies no physical U until its time floors have independent physical support.

## Evidence for a physical proof

Current registers, spills, occupancy, tiles, layouts, launch counts, and KV-slot choices are properties of one implementation.
They do not limit all legal implementations.

Fastest observed runtime is at least the unknown optimum. It cannot serve as a physical lower time bound.

Clock readings, event quantum, host gaps, and unfitted model units cannot supply the missing certificate.
Packed FP32 semantics do not fix a minimum native delay.
Low useful work per case does not prove hardware impossibility.
The normalization path also has no certified delay or clock cap.

A future proof has four parts:

1. Bind shapes, calls, work, precision, and source identity.
2. Prove time floors for all legal implementations and applicable inputs in each execution mode.
3. Calculate the invocation-weighted family U without a clamp.
4. Apply the alternative only after unreachability proof and complete numerical and timing coverage.

The [measurement rules](evaluation.md#measurement) apply after the proof. They do not change the physical-proof conditions.
