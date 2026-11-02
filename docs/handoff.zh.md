# 工作状态

继续在 `nightly` 上工作.

## 当前服务路径

Qwen 保留现有的单 worker 服务路径.
共享 Rust service, scheduler, logical cache, DEALER transport 和 msgpack protocol 保持启用.
runtime 包使用 `requirements/runtime.txt`.
`requirements/development.txt` 包含该文件, 因此一次 development 安装即可.

DeepSeek-V4.1-Flash 已有 model adapter 和四 rank Python bridge.
adapter 使用原始 48 shard checkpoint 和 rank-local placement.
它支持 text target execution 和 DSpark5 greedy proposal verification.
bridge 支持 `tp4`, `tep4` 以及已明确记录的 DEP4 executor fallback.

adapter 使用 DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation, mHC, RMS, sparse attention 和 transformer support.
prefill 默认使用最多 8192 个 total row 的 working chunk.
设置 `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` 可以选择最多 32768 个 row.
完成的 chunk 会在下一次 native call 之前释放 hidden tail.

DSpark verification 会将 accepted prefix 和 draft row 放入一次 target call.
DSpark proposal query 共享完整的 draft-index block.
它只投影需要的 logit, 并在一次 MTP operation 中 seed accepted target hidden state.
rejected draft 保持 private state, 并输出 target correction.
Engram table 使用 file-backed host mapping.
Lookup 只将选中的 row 分阶段传到 GPU.

released DeepSeek shape 支持 native projection, routing, activation, mHC, RMS 和 sparse attention path.
native contract 支持长 row range.
较少行数的输入使用有界 operator workspace.
配置为大 chunk 的 prefill 使用最多 32768-row 的有界 launch.
默认值会减少 long-context launch count. 环境覆盖值可以降低 temporary storage.

Native SwiGLU 支持最多 1048576 个 routed row.
这使 six-route 8192-row prefill chunk 保持在 native activation path.

DeepSeek rank worker 为 prefill input copy 复用 pinned host 和 device token buffer.

DeepSeek 使用发布模型的 FP4 和 FP8 scale rule, 并对数值使用最近偶数舍入.
Dense 和 grouped projection 在 scale multiplication 和 FP32 accumulation 之前, 为每个 K32 group 重置 raw dot product.
FP4 weight 在 raw dot 前精确转换为 FP8.
FP4 转换使用寄存器查找表, 并保留每个符号位.
输入至少有 32 行时, FP8 路径会在两个各有 16 行的输入块之间复用每个权重块.
输出宽度至多为 1024 且输入宽度至少为 1024 时, 不使用此复用方式.

Native RMS 保持发布参考的 square-sum reduction 和 reciprocal multiplication 顺序.
Native routing 保持发布参考的 softplus 和 selected-score sum 顺序.

In-place quantization 先舍入再反量化.
Native quantizer 写入 encoded value 和 scale, 不使用随 context 增长的 temporary tensor.
Native attention 将 window KV 和 compressed KV 保存在两张 table 中.
它读取模型的 BF16 table, 不执行量化.
Reference path 在计算之前合并这些 table.

Shared expert 使用完整复制的 FP8 weight 和 scale.
其输出在 routed rank reduction 后添加.
Shared output 不执行跨 rank reduction.
Sinkhorn mixing 在第一次 column normalization 之前加上 epsilon.

Native mHC 保持发布参考的 product, sum 和取决于 row 数的 mean 顺序.
其 FP32 projection 使用 tensor runtime 的 cuBLAS image 和显式 32 MiB operator workspace.
Prepared HC operator workspace 为 Eager 或 Graph execution 保留 FP32 input 和 statistics storage.
Nonzero target chunk 保持 source-step projection 和 reduction layout.
其 target attention 和 MoE execution 保持批量执行.
Native Engram arithmetic 使用 logical request batch 执行每份 copy 的 reduction.

Sparse attention 按顺序处理 64-slot block, 在与 V 相乘之前将未归一化的 probability 舍入到 BF16.
其 denominator 使用发布参考的 cross-warp sum 顺序和 FMA.
这些 block 完成后, sink 才计入 denominator, 不改变它们的 running maximum.

独立的 development reference 使用官方 MP4 placement, 并记录 quantizer synchronization patch.
冻结的 text fixture 包含完整 vocabulary vector, DSpark5 proposal 和 verification vector, 以及全新 accepted-only 重放的 restoration endpoint.
候选实现验收期间, 固定的参考数据和数值误差限保持不变.

sparse attention status failure 会禁用 native path, 并使用 reference path.
Native sparse attention 是 supported input 的默认路径.
设置 `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=0`, 为 short query 选择 reference attention.
long query 使用 native sparse attention, 因为 reference path 的扩展性较差.

DeepSeek token page 在 request slot 之间保存 compressed KV, index key 和 Engram history.
Accepted-state page 在 accepted endpoint 保存 target window, MTP window 和 compressor tail.
cache operation 支持 logical block store, load, drop, partial prefix copy 和
page-backed host snapshot.

accepted DSpark target state 会 seed MTP window.
Private endpoint snapshot 在 verification 期间保存已覆盖的 target window 和 compressor tail.
state commit 先选择 accepted endpoint, 再恢复 active state.
nonzero multi-token prefill 使用 fused chunk path.
target 和 MTP seed chunk 会将 total native row 保持在 operator limit 以下.
aligned prefill batch 默认使用四个 request 组成的 subbatch, 使用 32768-row setting 时最多使用 16 个 request.

MoE dispatch 只对 sorted routed row 做一次 gather, 并为 local expert 重用 slice.
TP4 和 TEP4 的 FP4 routed expert 使用 native grouped projection
执行两个 up projection 和一个 down projection.
TP4 路由专家的权重和缩放因子对全部三个投影都按输出通道分片.
TP4 在使用完整 K32 累加的下投影计算前, 收集 BF16 激活乘积.
每个专家输出在本地 FP32 词元求和前舍入到 BF16.

TP4 在添加各计算进程复制的共享专家输出前, 收集 FP32 词元和.
TEP4 保留路由结果的跨进程归约.

DeepSeek 允许在同一个 Execute 中抢占并重新启动同一个请求.
已结束的请求不能在该 Execute 中同时被调度或抢占.

Grouped projection 在 device 上计算 row offset, 不上传 host descriptor.
Grouped path 保持 local token order, 对 non-FP4 weight 和 test expert
回退到 reference expert loop.
MoE 每个 routed expert batch 最多处理 1024 个 token row.
Static packed-expert pointer array 使用有界的 LRU cache.

MoE 准备本地 FP4 专家绑定, 并重用各投影元组.
权重或专家重新绑定, 子 `_apply` 和状态加载会清除这些绑定.
下一次 grouped 调用会再次准备这些绑定.

Native projection 共享兼容的 encoded weight storage.
Packed scale 和 padded weight 使用不同的 storage.
Prepared cache entry 保留其原始 weight 和 scale 的 owner.
DEP4 仍使用 TEP4 executor fallback, 还没有 rank-owned dispatch.

非 final prefill 会跳过 vocabulary projection.
DSpark 只保留 MTP window 所需的 target hidden tail.
ordinary execution 不会收集 target hidden state.

完整的 initial prompt 可以作为一个 aligned request batch 执行.
每次 target call 都会 reset shared query attention state.
index source 总是为当前 request slot 发布 cache view.
Compressed KV source 在当前 cache write 后发布选中的 slot.
Compressed KV 和 index copy 只包含 active context range.

CUDA sequence staging 为 active request slot 使用按需映射的 storage.
Slot restoration 在复制 state 前映射所需 context.
Slot release 在解除 storage 映射前等待 device work.
Index scoring 使用有界 query chunk 和紧凑的 candidate-block mask.

serving engine 会执行配置的 `max_num_seqs` resident limit.
它会保留一个额外的 request 做 preparation, 以便让 CPU admission 与 GPU work overlap.

DeepSeek execution 当前使用 Eager.
共享 Qwen CUDA Graph path 已有 stream handoff, bounded graph operator workspace, graph upload 和 cleanup.
DeepSeek multi-rank Graph capture 尚未接入.
对照当前 vLLM 和 SGLang graph pattern 后, 在定义 static multi-rank
operator workspace 和 capture bucket 之前, DeepSeek 保持 Eager.

DeepSeek startup 只读取 `config.json` 和 `model.safetensors.index.json` 生成 descriptor identity.
Ready 之前不会 hash weight shard.

rank-local checkpoint tile 使用 65536 行, 以减少小块 host-to-device copy.
rank-local parameter 使用 16-byte aligned contiguous storage.
GPU startup 通过 bounded staging buffer 从每个 safetensors shard copy 所需
tensor range, 并跳过未使用的 Engram payload range.
设置 `OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` 可以覆盖此 startup setting.
外部 startup trace 使用 `OH_MY_VLLM_STARTUP_TRACE`, 并使用 repository 外部的 location.

已批准的 DeepSeek workload 定义位于 [deepseek matrix](../development/serving/deepseek_matrix.py).
DeepSeek 完整 HTTP 和 AgentX workload 尚未验收.

## 下一步

- 在 approved long-context workload 中测量每个 state endpoint 的 window storage.
- 比较 persistent encoded KV storage 与直接 BF16 table read 的测量结果.
- 在 long-context capacity 调整前, 测量 raw 和 prepared projection storage.
- 性能验收前, 保留独立且固定的 DeepSeek 文本端点检查.
- 在长上下文容量调优前, 测量 TP4 激活和输出的收集操作及其临时存储.
- 在 long-context capacity 调整之前, 减少重复的 cuBLAS operator workspace storage.
- 在 image acceptance 之前增加独立的 image endpoint.
- 分别报告配置的 context limit 和观测到的 AgentX request length.
- 为并发 16 和 100K 至 200K token trace 调整 AgentX physical block 和 temporary operator workspace capacity.
- 在 DeepSeek CUDA Graph capture 之前增加 per-request varlen position metadata 和 batch decode.
- 启用变长解码批次前, 检查每个请求的投影布局.
- 增加 physical cache 和 sequence capacity 后, 运行更高容量的 AgentX case.
- 用稳定的 request ownership 和 expert dispatch 替换 DEP4 fallback.
- 将 native DeepSeek path 与 reference path 和公开 baseline 做性能比较.
- 将 large-row native temporary storage 移入 adapter-owned graph operator workspace.
- 将 grouped expert GEMM 扩展到 DEP4 ownership 和 communication path.
- 在添加 DeepSeek CUDA Graph capture 和 replay 之前定义 static multi-rank operator workspace.
- 运行批准的 DeepSeek AgentX case, 发布 CSV 和暗色背景散点图.
- 在 GPU 0 以外的显卡上运行 38 个 HTTP case 和 3 个 AgentX case.
- 交付前取得 independent code review.

原始报告和临时 process records 保留在 repository 外部.
Host settings 和 model paths 放在 `LOCAL.md` 中.
