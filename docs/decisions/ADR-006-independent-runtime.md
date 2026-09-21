# ADR-006: Own the GPU execution runtime

Date: 2026-09-21. Status: user-approved; implementation in progress.

## Decision

Keep Rust serving, scheduling and logical KV ownership and Python GPU execution.
Replace vLLM integration with small project-owned implementations and independent
libraries. Selected upstream code may be adapted with its provenance and license;
copying the framework wholesale is not acceptable. Final builds, tests and services
must work without the vLLM package, source checkout, compiled operators, old conda
environment or its build caches. Transitional stages may still use the old adapter.

Select recent compatible stable dependencies in order, starting with the GPU
toolchain and tensor runtime. Pin the verified combination. System driver and
working CUDA/compiler tools are allowed; higher-level libraries live in conda
oh-my-vllm. Avoid nightly dependencies unless this decision is explicitly revised.

Investigate the highest-risk boundaries first: FP8 weight/scale semantics, GQA,
GDN recurrence and MTP state rollback. Then integrate model loading, request/cache
state, sampling, graph execution and service utilities. Migrate FP64 observation
points alongside their actual execution paths, retaining independent references.

Relevant correctness tests and independent review gate each milestone. Targeted
performance checks accompany hot-path changes; full nine-row acceptance is needed
at execution-chain completion and final acceptance. Intermediate performance gaps
must be recorded. Compare only with the original 2026-09-19 frozen EngineCore
baseline at >=95%; do not rerun native vLLM or add a V2-relative gate.

## Design references

- https://docs.vllm.ai/en/latest/design/model_runner_v2/: persistent request state,
  incremental updates, GPU metadata and explicit graph lifecycle.
- https://docs.vllm.ai/en/latest/design/vllm_ir/: separate operator semantics from
  implementation. Do not build a general compiler IR for this single-model task.
- https://docs.vllm.ai/en/latest/design/cuda_graphs/: separate graph capture from
  compilation, with stable buffers and explicit execution shapes.

## Future extensions (not current implementation requirements)

Keep model/weight mapping, device kernels, cache state and execution flow readable
and localized. Different models and NVIDIA GPUs can add concrete implementations
when needed. Single-node multi-GPU would add rank-local weights/state and
collectives around the same Rust scheduling contract. PD disaggregation and
multi-node execution are out of scope. Do not add speculative empty interfaces.

DSpark refers primarily to /data0/shared/Qwen3.8-27B-DSpark. Its config declares a
five-layer BF16 GQA draft model, target feature layers [5,19,33,47,61], confidence
and Markov heads, seven proposals and eight verification tokens including bonus.
Draft block_size=7 and training_block_size=16 are not the KV page size of 784.
Future integration needs independent draft state, target hidden features and
variable verification/commit/rollback handling. Exact feature extraction and token
mapping must be verified then; the checkpoint README is not local acceptance.
