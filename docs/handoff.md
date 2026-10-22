# Work state

Use `nightly` for the planned DeepSeek extension from base `b99701a`.
Read the [architecture proposal](deepseek-v41/design.md), [phase plan](deepseek-v41/plan.md), and [validation plan](deepseek-v41/validation.md).
The implementation is still Qwen-only. No DeepSeek runtime changes are present.

The architecture and measurement matrix have approval.
Execute all phases without additional phase approval.
Phase 0 is active: freeze checkpoint identities, tensor layouts, model mathematics, and independent numerical references.

The [checkpoint tools](deepseek-v41/checkpoint.md) provide full byte identities and a lazy CPU FP64 weight reader.
The [model contract](deepseek-v41/model.md) and [state contract](deepseek-v41/state.md) give the local mathematical sources and subsequent reference gates.
The [memory planner](deepseek-v41/memory.md) specifies tensor placement and storage budgets for all three four-rank strategies.
The checkpoint tools also supply independent CPU Engram hashes and continuation history.
Full-model reference outputs and complete runtime allocation bounds are still prerequisites.
Keep DeepSeek and new nightly regression measurements in separate documents, not README.

The following state applies to the Qwen implementation.
The [operator packages](../python/oh_my_vllm/operators) keep each family's interface, plan, workspace, and CUDA code together.
The [structure design](operators/design.md) defines their dependencies and ownership.
Keep shared resource mechanisms in `runtime` and shared computation in `primitives`.
The native builder is `development.native_build`. It builds seven native modules.

The worker uses explicit layer weights, named outputs, and operator plan keys.
Use the [execution rules](execution.md) for Eager execution, CUDA Graph replay, and resource release.
Rust and the worker control persistent state selection.
Operators write only to the state locations supplied by their callers.

Decode-only [draft chains](../python/oh_my_vllm/worker/draft_chain.py) can keep stochastic sampling and private presence history on the GPU.
Only the first draft step gets its input token from host metadata.
Subsequent draft steps use the GPU token output from the last step.
Prefix-dependent masks and unsupported stopping conditions use the per-step path.
Keep each draft's proposal probabilities and RNG offsets the same as the per-step path.

GDN prefill stores chunk workspaces in sequence order, with a capacity calculated for each sequence.
Keep the packed offsets consistent with the current schedule and the plan capacity.
The [service cache budget](serving.md) uses static FA defaults and checks runtime memory and text workspace allowances.
Rust uses the capacities returned by the worker.

Small FFN and GDN operations use dependent launches at the boundaries in the execution rules.
FFN prefill uses these dependencies for 65 to 1024 rows.
The [prefix cache](serving.md#prefix-cache) can use a checkpoint from a block that is not full.
Keep source blocks available until continuation results have no validation errors.

Use the [operator validation framework](../development/operators) before kernel changes.
It supplies fixed numerical fixtures, direct PyTorch references, and theoretical-latency ratio limits.
Usual numerical tests load repository fixtures without executing PyTorch.
Keep the [precision policy](operators/precision.md) and [full-model limits](operators/evaluation.md).
Do not automatically change limits after implementation changes.

Operator performance sampling keeps timing events and its measurement graph across uncertain-result extensions.
The timing migration keeps the approved latency limits.

FA uses BF16 attention operands and cache state. GDN keeps recurrent state in FP32.
FP8 projections use checkpoint E4M3 weights and FP32 block scales.
The [model contract](operators/model.md#intermediate-precision) defines the intermediate precision.

HTTP accounting uses serial kernel groups and the current precision rules.
Use the [performance calculation](performance.md#lower-bound-calculation) when fusion or execution boundaries change.
Operator ratio limits keep the approved maximum latency across accounting changes.

The shared FP8 prefill projection kernel selects its tile group size from the input width.
The FFN and projection kernels keep TMEM addresses in registers during accumulation.
HTTP efficiency has no minimum delivery limit.
Keep the numerical limits and operator ratio limits.
Report the measured performance.

Run the applicable [test scope](testing.md) before delivery.
For implementation deliveries, use the nightly validation plan to select the 24 Qwen [HTTP configurations](performance.md).

The [host KV cache](kv-offload.md) supports interrupted requests and pressure-driven prefix storage.
Keep transfer leases until completion, also for cancelled requests.
Keep partial GDN read checkpoints separate from continuation write slots until validated commit.
Pinned storage grows at idle windows. Metadata uploads use GPU kernels instead of the bulk copy queue.

The [AgentX entry](agentx.md) measures MTP4 with four session trees.
It uses streaming usage and full response IDs to join service metrics.
Keep its approved CSV and figure in `docs/assets`.
Keep raw measurement artifacts external to the repository.
Put host settings in ignored `LOCAL.md`.
