# DeepInfer 算子和 serving 基线

审计日期: 2026-10-26. 范围: 只读检查 DeepInfer, FlashMLA, DeepGEMM, DeepEP 和 TileKernels 的公开源码. 本报告把上游事实和 `oh-my-vllm` 建议分开. 本报告不添加依赖, 也不声明运行时支持.

## 公开基线

### DeepInfer 事实

DeepInfer 是面向 DeepSeek-V4.1-Flash 的离线推理框架. README 列出优化的模型, sampler 和 kernel runtime. README 还列出 attention DP 或 balanced CP, 带 EPLB 的 MoE EP, DSpark, SWA bounded replay, asynchronous scheduling, batch-invariant inference, deterministic sampling 以及 structured output.

同一份 README 把 KV cache system, prefill/decode disaggregation 以及 data-parallel replica 之间的 load balancing 列为缺少的 service 部件. 因此公开版本提供模型 runtime 基线, 不是完整 HTTP service.

`deepinfer/config.py` 定义 data-parallel, context-parallel, expert-parallel 和 Engram-parallel size. 它从 context parallelism 推导 model parallelism, 并在 `parallel_state.py` 中建立显式 group. 配置还为 draft 和 scheduling metadata 计算额外 runtime position.

`csrc/engine/scheduler.hpp` 和 `deepinfer/engine/llm_engine.py` 提供 iteration scheduling, paged block allocation, prefill 和 completion engine mode, preemption, sequence output 以及 MTP metadata. Scheduler 和 engine 是同一 process group 中的 C++/Python 组件. 它们不是 Rust service 加 ZMQ worker.

`deepinfer/worker/cache_engine.py` 定义 paged SWA, compressed-KV 和 index-key storage. 它使用 model-specific shared-layer map, packed cache format 以及用于 replay 和 compression 的额外 state. `cuda_graph_runner.py` 为选定 batch size 捕获 graph variant.

### 算子事实

FlashMLA 提供 V4.1 sparse prefill 和 decode kernel, 支持 FP8 或 FP4 KV cache. 它还提供融合 norm, RoPE, attention, output RoPE 和 cast 的 kernel. 当前 README 面向 SM100 或 SM103, CUDA 13.1 或更高版本以及 PyTorch 2.10 或更高版本. 2026-09-30 release 修改 KV format, 并移除 Hopper 和旧模型支持.

README 报告 V4.1 sparse kernel 在 CUDA 13.3 的 B200 上, prefill 最高 1350 TFLOPS, sparse decode 最高 1024 TFLOPS. 这些是 kernel measurement, 不是 service throughput 或 latency result.
同一 README 在 V4 fused norm-RoPE-attention-RoPE-cast 部分报告 prefill 最高 1460 TFLOPS, decode 最高 950 TFLOPS. 不要把这些 V4 数值作为 DSV4.1 baseline.

DeepGEMM 提供 FP8, FP4 和 BF16 GEMM, grouped MoE GEMM, sparse indexer 的 MQA scoring, Hyper-Connection kernel, Mega MoE 以及支持 PDL 的路径. README 说明 input transpose 和 FP8 cast 由 caller 负责, 除非 caller 把它们融合到前置 kernel. 它的 grouped inference API 使用固定 N 和 K 维度, 对齐的 M segment, 以及用于 decode graph 的 masked 形式.

DeepEP 通过 `EPBuffer` 提供 expert-parallel dispatch 和 combine. Dispatch 返回带有 token layout, expert count, source metadata 和 combine state 的 handle. Combine 消费这个 handle. 库还包含 experimental bucket collective, pipeline 和 context parallel path, 以及 GPU 或 CPU Engram storage. 它使用 PyTorch process group, NCCL, NVLink, RDMA 和 runtime JIT compilation.

检查过的 source 没有定义 Engram weight 的 pinned host allocation. 它们展示 CPU 或 GPU Engram storage 和 shared-memory registration, 但没有把 table residency 绑定到 `cudaHostRegister`, 也没有定义 transfer lifetime. 因此 pinned Engram residency 仍是项目待决策项.

TileKernels 提供基于 TileLang 的 MoE routing, FP8/FP4 conversion, Engram gate, mHC 和 RoPE 实现. README 面向 SM90 或 SM100, CUDA 13.1 或更高版本, PyTorch 2.13 或更高版本以及 TileLang. 它提供 kernel 和 modeling API, 不提供 request scheduler 或 public service.

## 可复用思路

下面的思路可以指导 model-specific design, 但不应导入上游库:

- 使用显式 prefill 和 decode plan. 为每个 plan 给出固定 layout, capacity, cache format 和 graph condition.
- 把 sparse attention metadata 与 logical cache identity 分开. Candidate block, top-k index, compressed-KV position 和 cache slot 都是派生 execution data.
- 把 MoE dispatch 和 combine 视为一个 semantic operation. 通过 typed handle 或等价 state, 把 route preparation 连接到 combine.
- 在 operator contract 中固定 FP8/FP4 scale layout. 在 kernel tuning 前写明 block size, packing order, alignment 和 output precision.
- 当完整 operation 获益时, 融合小型 producer 和 consumer work. 候选包括 norm-plus-cast, norm-plus-RoPE, attention-plus-cast 和 mHC stage.
- 让 Engram lookup 异步执行. Fetch 必须在 model 读取结果前返回 readiness event 或等价 lifetime token.
- 把 DSpark proposal, confidence, verification, acceptance 和 state restoration 作为独立 semantic step. 只有 accepted token 更新 public request state.
- 当 batch composition 变化时使用 deterministic metadata 和 stable ordering. Batch-invariant behavior 是 model capability, 不是通用 scheduler promise.

## 与本仓库的冲突

仓库 [`docs/requirements.md`](../requirements.zh.md) 禁止新增 DeepGEMM 和 FlashInfer 等外部 operator library. 仓库允许 project CUDA, CUTLASS, native CUDA library 和 CuTe DSL. 因此:

- 不要把 DeepInfer, FlashMLA, DeepGEMM, DeepEP, TileKernels, TileLang 或它们的 JIT build system 加入 runtime dependency.
- 在现有 operator `plan` 和 `workspace` interface 后重写已接受的 algorithm. 在仓库 contract 中保留 native identity, source hash, preparation, graph lifetime 和 resource ownership.
- 不要把 DeepInfer C++ scheduler 复制到 service. Rust 拥有 admission, scheduling, logical KV state, deadline 和 validated state commit.
- 不要让 Python process group 替换 rank protocol. Rust client 拥有 rank startup, typed plan, msgpack payload, completion check 和 failure handling.
- 不要把 DeepInfer offline engine 当作 HTTP, stored response, tool, media, cancellation 或 worker replacement support 的证明. README 明确把 service 部件排除在 release 之外.
- 不要在没有绑定 DSV4 cache schema 的情况下采用 DeepInfer FP8 或 FP4 cache layout. 当前 DSV4 报告仍把 packed cache 和 reference BF16 cache 作为待决策问题.
- 不要因为 DeepInfer 使用 CP, EP 和 DP 名称就声明支持 TP, TEP 或 DEP. Adapter 必须为本仓库定义 rank ownership, request placement, collective plan, cache group 和 accepted-state publication.

## 建议的边界

### Rust serving 边界

Rust 应拥有 public API, admission limit, request queue, prefill 和 decode priority, logical cache block 以及 prefix identity. Rust 还应拥有 cancellation, deadline, worker lifecycle 和 accepted-state commit.
Step payload 应包含 typed token work, position, batch metadata, cache binding, parallel descriptor 和 speculation metadata. 它不应包含完整 request history 或 private kernel layout.

Rust scheduler 应在 dispatch 前选择完整 execution mode. 它应把 prefill, ordinary decode 和 speculative verification 视为由 adapter 提供的 model capability. 只有在所有 rank 和 Python worker 报告 validated completion 后, Rust 才发布一个 result.

### Python adapter 边界

Adapter 应拥有 tokenizer 和 media codec, DSV4 model equation, rank-local weight 以及 physical cache view. Adapter 还应拥有 sparse-attention metadata, MoE routing metadata, Engram transfer view, DSpark state 和 sampling input.
它把 typed Rust assignment 转换为 operator input, 并返回 named output 加 state delta.

Adapter 应为 MLA compression 和 sparse attention, routed MoE, Engram lookup, mHC, output projection 和 DSpark verification 提供完整 semantic call. 内部可以使用 fused native kernel.
它不应向 Rust 发送 kernel launch count, private buffer index 或上游 library object.

### Native operator 边界

每个 operator 应有 plan, prepared weight, workspace, batch view, state view 和 named output. Plan 应是纯 selection data. Workspace 应拥有 temporary storage, native context, graph resource 和 asynchronous transfer lifetime.

首个 DSV4 operator baseline 应使用以下 family:

1. 带有显式 scale 和 packing contract 的 FP8/FP4 conversion 和 projection.
2. MLA compression, index scoring, candidate-block selection 和 sparse attention, 并分别提供 prefill 和 decode plan.
3. MoE gate, dispatch, grouped expert computation 和 combine, 并使用 typed route handle.
4. Engram hash, host-row fetch, device staging, gate 和 projection, 并使用显式 event ownership.
5. mHC normalization, mixing 和 residual application.
6. 把 DSpark draft, confidence, verification, correction 和 acceptance 作为独立 semantic operation.

这些边界与公开 operator evidence 一致, 同时保留本地 service contract. 它们也允许 PDL 和 fusion, 而不改变 Rust ownership 或 ZMQ/msgpack boundary.

## 决策

使用 DeepInfer 作为公开 model-runtime 和 scheduling reference. 使用 FlashMLA, DeepGEMM, DeepEP 和 TileKernels 作为 algorithm, layout 和 kernel-shape reference. 不要导入它们的 serving 或 build system.

在性能工作前冻结 DSV4 cache format, rank strategy, Engram transfer lifetime 和 DSpark acceptance state. 然后通过仓库的 plan/workspace contract 实现各个已接受的 operator, 并使用现有 FP64 和 exact-operator fixture 验证.

## 来源

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
- [DeepInfer configuration](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/config.py)
- [DeepInfer engine](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/engine/llm_engine.py)
- [DeepInfer scheduler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/scheduler.hpp)
- [DeepInfer cache engine](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/cache_engine.py)
- [DeepInfer V4.1 attention](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/mla_attn.py)
- [FlashMLA README](https://github.com/deepseek-ai/FlashMLA/blob/0f11e178da1f67becb087a324b400854931ec3e3/README.md)
- [FlashMLA cache format](https://github.com/deepseek-ai/FlashMLA/blob/0f11e178da1f67becb087a324b400854931ec3e3/csrc/cuda_kernels/kv_cache_format.h)
- [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
- [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md)
- [DeepEP EP buffer](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py)
- [DeepInfer shared memory](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/shared_memory.py)
- [DeepEP Engram buffer](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py)
- [TileKernels README](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/README.md)
- [`oh-my-vllm` runtime contracts](../requirements.zh.md)
- [`oh-my-vllm` operator design](../operators/design.zh.md)
- [`oh-my-vllm` GPU execution rules](../execution.zh.md)
- [DSV4 model and checkpoint contract](dsv41-model-checkpoint-contract.zh.md)
