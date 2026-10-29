# DeepInfer 对比

日期: 2026-10-29

本文对比 DeepSeek adapter 与 DeepInfer. 本文也根据当前 vLLM 和 SGLang
实现检查 CUDA Graph 设计.

## 主要结论

当前最大的差距是执行形状. adapter 把每个 decode 请求分别发送给 target
model. DeepInfer 把变长 decode batch 放入一次 model step. 这个差距增加
Python 工作, kernel launch 工作和重复通信.

adapter 原来使用 384 行的默认 prefill chunk. DeepSeek native operator
支持最多 8192 行. 默认值现在是 8192. 环境变量
`OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` 可以降低这个值.

DSpark 验证现在把 accepted prefix 和全部 draft token 放入一次 target
调用. 它只投影需要的 logits. 它在一次 MTP seed 操作中保存 accepted target
hidden state. 这会去掉每个 draft token 的一次 target forward 调用.

这些改动改善了执行形状. 它们还没有把 adapter 变成 varlen engine. 下一步
主要工作是 per-request position metadata.

## 调度

DeepInfer 在 scheduler 中保存 MTP state. 它记录 draft confidence,
accepted token 和 schedule counter. `MTPVarlen` policy 根据实测 decode
curve 选择 total query budget. 它把额外 query row 分配给 cumulative
confidence 最高的请求.

scheduler 在 C++ 路径运行 prompt 和 decode 工作. 它可以通过
zero-overhead scheduling 让 CPU post-processing 与下一次 decode forward
重叠. CUDA 路径会在上一次 sample 后在 device 上修正 query length.

本仓库在 Rust 中调度 token ID 和 cache ownership. Python adapter 收到一次
同步 execute message 并返回一个结果. 具有相同 start 和 length 的 prefill
row 会分组. 其余 decode row 仍然逐个执行. DSpark acceptance 在 target
logits 返回后由 CPU 执行.

Rust scheduler 已经支持 draft row 和 cache transaction. 它还没有提供
DeepInfer 风格 varlen decode step 所需的 per-request query length 和
position array.

来源: [DeepInfer MTP scheduler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/mtp_varlen.hpp),
[DeepInfer worker overlap](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/worker.py),
[本仓库 scheduler](../../crates/scheduler/src/lib.rs).

## 计算和算子

DeepInfer 使用 FlashMLA 做 attention 和使用 DeepGEMM 做 FP8 和 FP4
matrix operation. 它使用 DeepEP 做 expert communication, 使用 TileLang
kernel 做 metadata, cache, MTP 和 sampling 工作. 它把 quantized KV data
保存在 attention kernel 使用的 cache format 中.

adapter 对 routing, SwiGLU, FP8 和 FP4 projection, hyper-connection,
sparse attention 和 transform 有 native path. 一些高成本路径仍然使用
PyTorch operation. 这些路径包括 low-rank projection, tensor parallel
embedding 和 head reduction, `einsum` output projection 以及 expert
output reduction.

sparse native path 在每次调用中都 pack window 和 compressed KV tensor.
DeepInfer 把 quantized KV data 保存在 cache 中, 并把 cache layout 传给
FlashMLA. adapter 也把 TEP4 作为 DEP4 fallback. TEP4 计算 local routed
expert 并使用 all-reduce. DeepInfer 使用 expert parallel token exchange
和 expert load balancing.

算子工作应按以下顺序进行:

1. 增加 per-request varlen metadata 和 batch decode.
2. 让 quantized KV 保持在 runtime cache format 中.
3. 用 prepared native workspace 替换重复的 PyTorch projection 和 reduction path.
4. 用 stable expert owner 和 token exchange 替换 TEP4 fallback.
5. 增加 GPU-side MTP acceptance 和 confidence scheduling.

来源: [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md),
[DeepInfer model runner](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/model_runner.py),
[DeepInfer attention backend](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/attention_backend.py).

## CUDA Graph 检查

共享 Qwen Graph path 遵守重要的 ownership 规则. 它使用独立 capture
stream, 并等待 caller 和 transfer stream. 它在 replay 前 upload graph,
并保持 buffer address 固定. 它在释放 workspace owner 前释放 graph. CUDA
cleanup 失败时, 它的 cleanup path 会保留资源.

DeepInfer 也使用相同的总体模式. 它 capture padded decode bucket, 保留可复用
input 和 output buffer, 使用 graph memory pool, 并在 main graph 后 capture
DFlash graph. 它会在 process group shutdown 前销毁 graph, 因为 collective
可能保留 graph dependency.

vLLM 明确管理 graph key 和 fallback selection. SGLang 在 replay 前检查
ragged verification, padding, speculative mode 和 attention parallel
alignment 的 Graph support. 这些检查避免 graph 使用不兼容的 metadata.

DeepSeek 应继续使用 eager, 直到 model call 接受每个请求的 static device
metadata. 当前 model 使用 scalar `start_pos` 和 scalar sequence length
assumption. 它也在 Python 中构造 cache index. 直接 capture 当前 loop 会
capture 一个 request shape, 或者 replay 错误 position. 正确的 DeepSeek
Graph design 需要:

- total query row 的 fixed capture bucket.
- query length, position, cache slot 和 block table 的 device buffer.
- 每个 rank 和 collective path 的 graph eligibility check.
- address 固定的独立 capture 和 replay buffer.
- 对不支持的 shape, cache 或 collective state 使用 eager fallback.
- 在销毁 process group 前明确销毁 graph.

来源: [vLLM CUDA Graph design](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md),
[vLLM GPU model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py),
[SGLang decode Graph runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py),
[DeepInfer Graph runner](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/cuda_graph_runner.py).

## 长上下文

DeepSeek AgentX command 接受 `1048576` 作为 context limit. service
descriptor 也公布一百万 token model limit. Physical GPU block 和 host
cache capacity 是独立的 runtime setting. 完整的一百万 token trace 需要
足够的 cache resource. 当前 AgentX test 检查 command 保留一百万 token
limit 和 concurrency 16.

## 决策

保留共享 Qwen Graph implementation. 在 varlen metadata 和 rank-safe graph
eligibility 出现前, 让 DeepSeek 保持 eager. 使用新的 batched DSpark
verification 和 8192 行 prefill default 作为第一批性能改动. 后续每次
varlen 或 operator 改动后都测完整 service.
