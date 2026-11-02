# DeepSeek V4.1 adapter

DeepSeek-V4.1-Flash uses a four-rank Python worker group.
Rust owns service control, admission, scheduling, logical state, and publication.
Python owns checkpoint loading, tensors, model computation, and DSpark state.
The worker boundary uses the existing ZMQ DEALER and msgpack protocol.

The adapter binds the public configuration, inference configuration, tokenizer assets, and checkpoint metadata.
It reads weight shards during load without a full startup hash pass.
It supports the released 48-shard safetensors layout, FP8 projections, FP4 routed experts, sparse attention, Engram tables, and Hyper-Connections.
Prefill rows are split before native operators receive them.
Nonzero multi-token prefill uses the fused chunk path.

The current functional path supports text target decode and greedy DSpark5 verification.
Short checkpoint checks use TEP4 on GPUs `4,5,6,7`.
The target path uses Eager execution.
The shared CUDA Graph implementation is not connected to this multi-rank adapter.

The supported public capability is text.
Image preparation is kept out of the public capability until the tokenizer, image processor, rank execution, and HTTP checks are connected.
The current DEP4 command selects the TEP4 executor as a functional fallback.
It does not provide DEP4 request ownership or expert dispatch.

The physical cache keeps model pages behind the Rust logical block tables.
It supports prefix reuse, partial block copies, host snapshots, and restoration.
Accepted-state pages keep target windows, MTP windows, and compressor tails at an exact endpoint.
Restoration rejects a different endpoint before it clears an active slot.
FA prefix copies contain only compressed KV, index keys, and Engram history.
DeepSeek still uses Eager execution because coordinated multi-rank Graph replay is not connected.

Verification keeps private state for each possible accepted endpoint before it overwrites recurrent storage.
Commit publishes only accepted states and scheduled checkpoints.
Prefill exports known checkpoint states before it releases their hidden rows.
Verification keeps at most five additional hidden rows for accepted MTP seed state.
Window storage grows with checkpoint capacity, not with FA token-page capacity.

Native [projection preparation](../../../python/oh_my_vllm/models/deepseek_v41/kernel.py) shares compatible encoded weight storage.
Packed scales and padded weights keep separate storage.
The reference path can still use the raw parameters.
The [quantization contract](quantization.md) defines activation and KV formats, scale rounding, and in-place behavior.

CUDA sequence staging uses demand-mapped storage for compressed KV, index keys, and Engram history.
Each request slot maps only its active context range.
The logical tensor address stays fixed during growth.
Restoration maps the required range before the state copy.
Slot release waits for device work before it releases physical storage.

CPU reference execution keeps dense tensors.
Models with mapped storage do not support tensor device or dtype conversion.

Index score tensors use query chunks with a 256 MiB budget.
This budget excludes inputs, selected results, and native library workspaces.
FP16/BF16 index rank sums use FP32 and return to the score dtype after the sum.
Candidate filters keep one boolean per block.
Consumers expand only the current query chunk to token positions.

Use the [HTTP procedure](performance.md) for the approved 38-case matrix.
Use the [AgentX procedure](agentx.md) for the three strategy cases.
The procedures define measurements. They do not add performance gates.

## GPU execution

Use four GPUs apart from GPU 0.
The current validation preference is GPUs `4,5,6,7`.
Set the four GPU UUIDs through the GPU wrapper or the ignored `LOCAL.md` file.

Build the demand-mapped storage module before starting the worker:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build --module paged
```

The default worker context is bounded for functional startup.
Set `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` to the requested physical context before a long-context check.
The descriptor keeps the released 1048576-token model limit.
The physical cache setting must cover the requested prompt and output.

## State

The Python worker keeps model cache state in rank-local tensors and physical pages.
Rust logical cache transfers return completion records for the protocol seam.
DeepSeek-specific DEP4 ownership and coordinated Graph replay stay in progress.
Nonzero multi-token prefill uses the fused chunk path.
Target and MTP seed chunks keep total native rows below the operator limit.

Short TEP4 evidence does not prove TP4, DEP4, long context, images, Graphs, or SOTA.
These paths must have separate checks.
