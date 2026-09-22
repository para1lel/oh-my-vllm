# Completed plan — TileLang, TTFT and long context

1. Updated and froze the isolated official vLLM main baseline and its environment.
2. Implemented per-request TTFT and independent FA/GDN cache capacities.
3. Replaced all seven owned Triton kernel modules with TileLang; integrated the
   development-only personal TileFoundry fork and offline autotuning workflow.
4. Tuned required shapes until all twelve rows passed throughput >=95%, TTFT
   <=110%, and the user-revised10% stability gate. Preserved original raw sets.
5. Passed unchanged correctness, six262144-token ordinary/MTP4 boundary rows,
   long prefix, constraints, lifecycle and both actual oh-my-pi providers.
6. Added25 Chinese companions and synchronized final evidence. Final document
   readback, review, main commit/push and GPU cleanup verification close the task.

See acceptance.md and handoff.md for current evidence and answer limitations.

# Implementation status

The independent runtime migration (ADR-006) is implemented: stable dependencies
are pinned in conda oh-my-vllm; model loading, GPU state and required kernels are
owned by this project or independent libraries. The old adapter and all vLLM
runtime/build/test dependencies are removed. Rust retains serving, scheduling
and logical KV ownership; the ZMQ protocol and block 784 remain unchanged.

FP64 operators and actual-model probes, ordinary/MTP text, prefix/preemption,
constraints, lifecycle and CUDA Graph regressions pass. All nine original-baseline
performance rows pass. Both final oh-my-pi workflows completed after documentation reconciliation;
results and generated-answer caveats are recorded in acceptance.md and handoff.md.

Earlier GPUWorker/V2 stages are historical milestones, not active implementations.
Their evidence and decisions remain under bench/baseline and docs/decisions.
Future architectures, NVIDIA backends, single-node multi-GPU and local DSpark
extensions are described in ADR-006, without placeholder interfaces or claims of
current support. CPU KV swap, PD/multi-node, LoRA and multimodal execution remain
out of scope. Continue independent reviews, local pre-commit checks, main-only
commits and prompt cleanup of task-owned GPU programs.
