# DeepSeek V4.1 适配器

DeepSeek-V4.1-Flash 使用四 rank Python worker group.
Rust 负责 service control, admission, scheduling, logical state 和 publication.
Python 负责 checkpoint loading, tensor, model computation 和 DSpark state.
worker boundary 使用现有的 ZMQ DEALER 和 msgpack protocol.

adapter 绑定 public configuration, inference configuration, tokenizer asset 和 checkpoint metadata.
它在 load 期间读取 weight shard, 但不执行完整 startup hash pass.
它支持发布的 48 shard safetensors layout, FP8 projection, FP4 routed expert, sparse attention, Engram table 和 Hyper-Connections.
prefill row 在进入 native operator 前切分.
nonzero multi-token prefill 使用 fused chunk path.

当前 functional path 支持 text target decode 和 greedy DSpark5 verification.
短真实 checkpoint checks 在 GPU `4,5,6,7` 上使用 TEP4.
target path 使用 Eager execution.
共享 CUDA Graph implementation 尚未接入这个 multi-rank adapter.

当前公开 capability 是 text.
在 tokenizer, image processor, rank execution 和 HTTP checks 接通前, image preparation 不加入公开 capability.
当前 DEP4 command 选择 TEP4 executor 作为 functional fallback.
它没有提供 DEP4 request ownership 或 expert dispatch.

physical cache 在 Rust logical block table 后面保存 model page.
它支持 prefix reuse, partial block copy, host snapshot 和 restoration.
Accepted-state page 在精确的 endpoint 保存 target window, MTP window 和 compressor tail.
Restoration 在清空 active slot 前拒绝不匹配的 endpoint.
FA prefix copy 只包含 compressed KV, index key 和 Engram history.
由于 coordinated multi-rank Graph replay 尚未接入, DeepSeek 当前仍使用 Eager execution.

Verification 在覆盖 recurrent storage 前, 为每个可能接受的 endpoint 保存 private state.
Commit 只发布 accepted state 和 scheduled checkpoint.
Prefill 在释放 hidden row 前导出已知 checkpoint state.
Verification 最多额外保留五个 hidden row, 用于 accepted MTP seed state.
Window storage 随 checkpoint capacity 增长, 不随 FA token-page capacity 增长.

Native [projection preparation](../../../python/oh_my_vllm/models/deepseek_v41/kernel.py) 共享兼容的 encoded weight storage.
Packed scale 和 padded weight 保留独立的 storage.
Reference path 仍可使用 raw parameter.

CUDA sequence staging 为 compressed KV, index key 和 Engram history 使用按需映射的 storage.
每个 request slot 只映射其 active context range.
增长期间 logical tensor address 保持不变.
Restoration 在 state copy 前映射所需 range.
Slot release 在释放 physical storage 前等待 device work.

CPU reference execution 保留 dense tensor.
使用映射 storage 的 model 不支持 tensor device 或 dtype conversion.

Index score tensor 使用 query chunk, budget 为 256 MiB.
这个 budget 不包含 input, selected result 和 native library workspace.
FP16/BF16 index rank sum 使用 FP32, 求和后返回 score dtype.
Candidate filter 为每个 block 保留一个 boolean.
Consumer 只将当前 query chunk 展开到 token position.

approved 38-case matrix 使用 [HTTP 流程](performance.zh.md).
三个 strategy case 使用 [AgentX 流程](agentx.zh.md).
这些流程定义 measurement, 不增加 performance gate.

## GPU 执行

使用 GPU 0 以外的四张显卡.
当前 validation 优先使用 GPU `4,5,6,7`.
通过 GPU wrapper 或 ignored `LOCAL.md` 设置四个 GPU UUID.

启动 worker 前, 构建按需映射的 storage module:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build --module paged
```

默认 worker context 为 functional startup 设置了上限.
执行 long-context check 前, 将 `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` 设为所需 physical context.
descriptor 保留发布的 1048576-token model limit.
physical cache setting 必须覆盖请求的 prompt 和 output.

## 状态

Python worker 在各 rank-local tensor 和 physical page 中保存 model cache state.
Rust logical cache transfer 为 protocol seam 返回 completion record.
DeepSeek-specific DEP4 ownership 和 coordinated Graph replay 仍在开发.
nonzero multi-token prefill 使用 fused chunk path.
target 和 MTP seed chunk 会将 total native row 保持在 operator limit 以下.

不要将短 TEP4 evidence 视为 TP4, DEP4, long-context capacity, image serving, CUDA Graph execution 或 SOTA performance 的证明.
