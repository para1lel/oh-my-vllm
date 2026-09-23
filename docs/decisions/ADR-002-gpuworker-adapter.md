# ADR-002: Adapt the installed GPUWorker without importing its scheduler

Accepted 2026-09-19 under the approved Rust scheduling/Python runner boundary.
**Superseded** by ADR-005 and then ADR-006 (2026-09-21): the vLLM adapter and
physical-stride remapping no longer exist. Retained as history.

The installed vLLM is an editable checkout at `/data0/shared/dongwu.chen/vllm`
(version metadata: 0.28.1rc1.dev691+g6376c601e.cu131). Its worker class is `Worker`;
initialization uses a current VllmConfig context, device/load, memory profiling,
KV configuration, initialize_from_config, and compile_or_warm_up_model.
EngineArgs normalizes configuration only; no vLLM scheduler is instantiated.
Text-only mode avoids initializing the out-of-scope vision encoder.

The actual physical KV layout is three Mamba groups and one FA group, in that
order. Rust retains two logical kinds and a shared block-ID pool. Python maps its Mamba table into distinct physical IDs for each group and maps
FA by spec kind, never by assumed group order. Physical groups share tensor
storage, so repeating the same ID across Mamba groups corrupts layer states.
With stride=max(number of physical groups of each kind), logical block b maps
to b*stride+group_offset; null always maps to 0. Rust's shared pool keeps FA and
Mamba logical IDs distinct. Ready returns floor(physical_capacity/stride), so
Rust cannot allocate a block outside the physical pool. This reserves unused
addresses for FA blocks; a weighted physical allocator could reclaim them later.
With 1024 physical blocks, 341 logical slots cover the roughly 200 live
logical blocks needed by four 36864-token ordinary requests; long-workload
execution is still being verified.
The logical and physical block sizes are asserted to stay 784. Post-load
platform setup is required to set Mamba block size and page padding.

SchedulerAdapter tracks worker admission separately from num_computed_tokens:
a prefix-hit request is still new to the worker. Running updates append new
block IDs; resumed requests replace tables. Result rows are matched by req_id,
not scheduling order. Empty token lists represent intermediate prefill, and
multi-token lists preserve all accepted speculative outputs. Completion is
flushed to the worker even when no model tokens remain scheduled.

Real inference has passed a 16-token smoke. Full functional, actual-path accuracy
and performance acceptance are tracked separately in handoff.md. Runtime
probes must not be enabled for performance measurements.
