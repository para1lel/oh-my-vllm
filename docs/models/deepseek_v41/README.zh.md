# DeepSeek V4.1 适配器

DeepSeek-V4.1-Flash 使用四 rank Python worker group.
Rust 负责 service control, admission, scheduling, logical state 和 publication.
Python 负责 checkpoint loading, tensor, model computation 和 DSpark state.
worker boundary 使用现有的 ZMQ DEALER 和 msgpack protocol.

adapter 绑定 public configuration, inference configuration, tokenizer asset 和 checkpoint shard.
它支持发布的 48 shard safetensors layout, FP8 projection, FP4 routed expert, sparse attention, Engram table 和 Hyper-Connections.
prefill row 在进入 native operator 前切分.

当前 functional path 支持 text target decode 和 greedy DSpark5 verification.
短真实 checkpoint checks 在 GPU `4,5,6,7` 上使用 TEP4.
target path 使用 Eager execution.
共享 CUDA Graph implementation 尚未接入这个 multi-rank adapter.

当前公开 capability 是 text.
在 tokenizer, image processor, rank execution 和 HTTP checks 接通前, image preparation 不加入公开 capability.
当前 DEP4 command 选择 TEP4 executor 作为 functional fallback.
它没有提供 DEP4 request ownership 或 expert dispatch.

approved 38-case matrix 使用 [HTTP 流程](performance.zh.md).
三个 strategy case 使用 [AgentX 流程](agentx.zh.md).
这些流程定义 measurement, 不增加 performance gate.

## GPU 执行

使用 GPU 0 以外的四张显卡.
当前 validation 优先使用 GPU `4,5,6,7`.
通过 GPU wrapper 或 ignored `LOCAL.md` 设置四个 GPU UUID.

默认 worker context 为 functional startup 设置了上限.
执行 long-context check 前, 将 `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` 设为所需 physical context.
descriptor 保留发布的 1048576-token model limit.
physical cache setting 必须覆盖请求的 prompt 和 output.

## 状态

Python worker 当前在各 rank-local tensor 中保存 model cache state.
Rust logical cache transfer 为 protocol seam 返回 completion record.
DeepSeek-specific cache ownership, prefix reuse, host residency 和 restoration contract 仍在开发.

不要将短 TEP4 evidence 视为 TP4, DEP4, long-context capacity, image serving, CUDA Graph execution 或 SOTA performance 的证明.
