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
