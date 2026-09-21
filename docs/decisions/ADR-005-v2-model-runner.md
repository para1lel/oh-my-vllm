# ADR-005: Use V2 Model Runner exclusively

Date: 2026-09-21. Status: accepted, implemented and verified.

## Decision

Keep GPUWorker as the Python boundary and require its V2
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`. Rust retains scheduling, logical
KV allocation and authoritative accepted token history. Remove the old runner
recovery branch and unused legacy MTP helpers; provide no legacy fallback.
Do not edit vLLM source. Implement integration gaps in this repository.

V2 removes preempted requests before adding new requests and only appends cached
block updates. Therefore resumption uses NewRequestData with full accepted
`prefill_token_ids` and replacement block tables. Rust sends full history only at
admission/resumption, excluding drafts; normal decode remains incremental.
The original prompt is retained separately so output budgets and generated-token
counts survive recompute. Prefix hits retain their computed-token positions.

V2's default draft handler avoids copying unconstrained drafts and returns -1
placeholders. Rust needs real unsigned IDs. A local handler subclass passes a
shallow copy of the batch with the transfer flag enabled to the existing CUDA
copy/event implementation. The actual runner's grammar/sampling flags are untouched.
This adds D2H/event overhead for plain MTP and must pass real performance checks.

Hybrid cache pages also require V2's new_block_ids_to_zero contract. Rust records
only actual fresh allocations in BlockPool; cached references and live speculative
slots are excluded. Python expands logical allocations to all reserved physical
stride IDs and invokes the existing V2 zeroing path before execution. Cache tensors
are also cleared in place after startup warmup, retaining CUDA graph addresses.
Actual FP64 probes exposed NaN attention output from stale warmup contents; clearing
restored finite output within the original tolerances. Runtime allocation zeroing
additionally prevents stale mixed-layout bytes after page recycling.

## Acceptance

Preserve all offline and serving features, MTP4 constraints, real OMP tasks,
prefix hits, preemption/recompute and cancellation. Keep existing actual-kernel
FP64 tolerances; full-model token equality is not required. Inspect HTTP logs for
abnormal performance without adding a strict serving throughput gate.

Run only framework measurements for the nine ordinary/MTP/prefix × bs1/2/4
32768→4096 workloads. Compare against the frozen EngineCore measurements in
bench/baseline/2026-09-19-acceptance.json; do not rerun or select a more favorable
baseline. Record artifact hash, runtime source identity and binary identity.
The editable vLLM source is at 039b2ad67da6d64f7c1835c4738c7fb545ad37fc while its
package metadata still reports g6376c601e, matching the historical version label.
This is a historical comparison, not a same-source paired benchmark.

All nine historical performance rows and real FP64/text/MTP/serving/OMP checks
passed. See [acceptance evidence](../acceptance.md) for measured results and
source-identity limitations.
