# 工作状态

继续在 `nightly` 上工作.

## 当前服务路径

Qwen 保留单 worker 路径和共享 Rust service, scheduler, logical cache, DEALER transport 及 msgpack protocol.
`requirements/development.txt` 包含 `requirements/runtime.txt`.

Qwen decode 和 single-query verification 使用 native operator.
[采样](../python/oh_my_vllm/models/qwen38/sampling.py#L28)<!-- source: self.full_history_view = self.full_history.numpy() -->使用 NumPy view 打包 mask 和 metadata.
[上传](../python/oh_my_vllm/operators/runtime/uploads.py#L207)<!-- source: self._native_key = key -->保留 pinned snapshot, 并复用 native binding.
当 verification batch 至少有两个 request 时, 每组 metadata 上传一次.
其他 target 和 context input 使用分别上传的路径.

[Acceptance](../python/oh_my_vllm/models/qwen38/speculation.py#L232)<!-- source: self.metadata_inputs = Metadata(device, arrays, single_upload=True) --> 将 gather index, draft, slot 和 RNG 打包为一次上传.
准备后的 [metadata writer](../python/oh_my_vllm/worker/metadata.py#L116)<!-- source: self.uploads.update(self._write, sources) --> 在输入校验和事件完成后更新 pinned snapshot.

grammar 约束的 MTP4 chain 已禁用.
当至少有两个 request 时, 约束 draft 可以重叠 GPU 计算与 CPU 采样准备.
未使用的计算会恢复其 KV 行.
Batch 1 保留分别 draft 上传和原 mask 路径.
[草稿流水线检查](../development/operators/check_draft_pipeline.py)比较重叠执行和顺序执行.

Draft 复用重复检测证明和精确的已过滤 mask.

Qwen 使用 [HTTP reference v5](models/qwen38/performance.zh.md#下界计算) 和 [operator reference v4](models/qwen38/operator-performance.zh.md).
下一步: 重复长输出测量, 完成 24 项 HTTP 矩阵和 AgentX 验收.

DeepSeek-V4.1-Flash adapter 使用原始 48 shard checkpoint 和 rank-local placement.
其四 rank Python bridge 支持 text target 和 DSpark5 greedy verification.
它有 `tp4`, `tep4` 和按请求归属执行的 Eager `dep4` 路径.

adapter 提供模型 descriptor, cache groups, identity 检查和 native operator.
prefill 默认使用最多 8192 个 total row 的 working chunk.
设置 `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` 可以选择最多 32768 个 row.
完成的 chunk 会在下一次 native call 之前释放 hidden tail.

DSpark verification 会将 accepted prefix 和 draft row 放入一次 target call.
DSpark proposal query 共享完整的 draft-index block.
它只投影需要的 logit, 并在一次 MTP operation 中 seed accepted target hidden state.
rejected draft 保持 private state, 并输出 target correction.

两个 Engram buffer 使 lookup 与 GPU decode 重叠.
Event 保护 storage 复用和失败恢复.
Lookup 在 replay 前刷新 Graph buffer.

发布的 shape 使用 native operator 和有界 workspace.
Prefill 每次 launch 最多使用 32768 行.
设置较小的 chunk 可以减少 temporary storage.

Native SwiGLU 支持最多 1048576 个 routed row.
这使 six-route 8192-row prefill chunk 保持在 native activation path.


DeepSeek 使用发布模型的 FP4 和 FP8 scale rule, 并对数值使用最近偶数舍入.
Dense 和 grouped projection 在 scale multiplication 和 FP32 accumulation 之前, 为每个 K32 group 重置 raw dot product.
FP4 weight 在 raw dot 前精确转换为 FP8.
FP4 转换保留符号位.
支持的 [dense prefill 形状](models/deepseek_v41/gemm.zh.md#dense-prefill-分块) 使用 TMA 和 TMEM, 并保留精确的 FP16 转换与有序 K32 缩放更新.
其他 FP8 形状保留已有内核.

FP8 decode 可并行计算 K32 dot.
Scale 累加保留 K32 顺序.

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
单个 source row 的 FP32 projection 使用 32 个交错 split dot.

多 row 的 HC, router 和 compressor projection 从实测 table 为 2 到 32768 个 logical row 选择 chunk.
未实测的 shape 或 row 数会停止 call.
共享 tile 保持每个输出的 FMA 和 chunk 加法顺序.
单 chunk HC call 使用异步复制和两个共享 buffer.
源分块使用行切片和 32 MiB 工作区.
模型默认提供该 workspace.

显式的空 projection workspace 选择逐线程参考路径.
Prepared HC workspace 为 Eager 和 Graph execution 保留 FP32 input, statistics 和 32 MiB partial storage.
Ratio-1 compressor 的 BF16 projection 使用已发布的非 split-K cuBLAS 配置.
Startup 检查已接受的 BF16 projection pattern.

Nonzero target chunk 保持 source-step projection 和 reduction layout.
HC 和 FP32 的 source position 共用 [HC kernel 发射](../python/oh_my_vllm/operators/hyperconnection/cuda/hc_projection.cuh#L615)<!-- source: int steps = rows / logical_rows; --> 和 [FP32 projection 发射](../python/oh_my_vllm/operators/hyperconnection/cuda/hyperconnection.cu#L193)<!-- source: int step_rows = logical_rows ? static_cast<int>(logical_rows) : rows; -->.
Logical row 数继续选择每个 position 的 split, chunk 和 statistics layout.
其 target attention 和 MoE execution 保持批量执行.
Native Engram arithmetic 使用 logical request batch 执行每份 copy 的 reduction.

Sparse attention 按顺序处理 64-slot block, 在与 V 相乘之前将未归一化的 probability 舍入到 BF16.
其 denominator 使用发布参考的 cross-warp sum 顺序和 FMA.
BF16 table 每批执行四次读取, 每次读取八个值.
一个 warp 计算相同的 cross-warp sum tree, 其他 warp 读取其结果.

这些 block 完成后, sink 才计入 denominator, 不改变它们的 running maximum.
最多四行的 BF16 attention 使用调用者拥有的 scratch, 并行计算 64-slot partial.
各输出通道独立执行, 保留原来的 running maximum, denominator 和 numerator 更新顺序.

独立的 development reference 使用官方 MP4 placement, 并记录 quantizer synchronization patch.
冻结的 text fixture 包含完整 vocabulary vector, DSpark5 proposal 和 verification vector, 以及全新 accepted-only 重放的 restoration endpoint.
候选实现验收期间, 固定的参考数据和数值误差限保持不变.

sparse attention status failure 会禁用 native path.
native 调用在 device 上累积 status, step boundary 在 worker 提交 cache state 之前检查一次.
被拒绝的 step 会报错, 不会就地使用 reference path.
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
TP4 和 TEP4 使用原生 grouped FP4 上投影和下投影.
TP4 路由专家的权重和缩放因子对全部三个投影都按输出通道分片.
TP4 在使用完整 K32 累加的下投影计算前, 收集 BF16 激活乘积.
每个专家输出在本地 FP32 词元求和前舍入到 BF16.

词元行数至少为 128 时, kernel 在输出列之间重用每个 token 排序后的投影行索引.

TP4 在添加各计算进程复制的共享专家输出前, 收集 FP32 词元和.
TEP4 保留路由结果的跨进程归约.

DeepSeek 允许在同一个 Execute 中抢占并重新启动同一个请求.
已结束的请求不能在该 Execute 中同时被调度或抢占.

Grouped projection 使用 device offset, 压缩后的非空 expert, 以及保持 native K32 顺序的 persistent FP16 tile.
总 routed row 不超过 64 时, 使用并行 raw dot 和有序 scale 累加.

TP4 device count path 没有 routed-row 上限.
Grouped execution 保持 token order.
Non-FP4 权重和 test expert 使用 reference loop.

Routed 消息最多有 1024 行.
TP4 和 TEP4 prefill 使用较大的 [投影波次](models/deepseek_v41/gemm.zh.md#路由-prefill-波次).
Shared 和 routed gate/up 投影共享输入量化.
Shared expert 处理完整的 prefill chunk.

[固定容量内核](models/deepseek_v41/gemm.zh.md#固定容量投影与求和) 为 SwiGLU 将投影尾部清零, 并使用设备 status.
TEP4 Graph 和 DEP4 调用方的接入尚未完成.

TP4, TEP4 和 DEP4 为每个 weight set 保留 [device expert pointer](../python/oh_my_vllm/models/deepseek_v41/model.py#L1846)<!-- source: pointers = state.pointers --> 和 FP4 绑定.
权重或专家重新绑定, 子 `_apply` 和状态加载会清除它们.

Native projection 共享兼容的 encoded weight storage.
Packed scale 和 padded weight 使用不同的 storage.
打包缓存项保留源张量的引用和版本信息.
状态加载会清除共用底层存储的缓存项.
DEP4 使用完整稠密权重, owner 本地缓存, 完整 Engram 读取, 不同 token/目标 rank 的专家交换, 以及所有 rank 的完成确认.

非 final prefill 会跳过 vocabulary projection.
DSpark 只保留 MTP window 所需的 target hidden tail.

完整的 initial prompt 可以作为一个 aligned request batch 执行.
每次 target call 都会 reset shared query attention state.
Compressed KV source 在当前 cache write 后发布选中的 slot.
Compressed KV 和 index copy 只包含 active context range.

CUDA sequence staging 为 active request slot 使用按需映射的 storage.
Slot restoration 在复制 state 前映射所需 context.
Slot release 在解除 storage 映射前等待 device work.
Index scoring 使用有界 query chunk 和紧凑的 candidate-block mask.

serving engine 会执行配置的 `max_num_seqs` resident limit.
它会保留一个额外的 request 做 preparation, 以便让 CPU admission 与 GPU work overlap.

DeepSeek TP4 target decode 使用捕获的 CUDA Graph.
TEP4 和 DEP4 在 host 上读取 routed count, 使用 Eager execution.
每个 cache-slot set, context bucket 和 position residue 组合捕获一个 Graph.
Graph 包含 collective.
Engram host lookup, sampling 和 token read 使用 host path.

Serving 在每次 replay 后导出 target window 和 tail.
Graph 保留缓存的 tensor 输入, 并拒绝已改变的专家绑定.
未捕获的 group 和超过 Graph cache 上限的 step 使用 Eager.
Qwen Graph 保留有界 workspace, stream handoff, graph upload 和 cleanup.
清理失败后, operation 和输入 owner 与失败的 Graph 一起保留.
DeepSeek prefill 和 draft verification 使用 Eager execution.

实验性批量 target 保留每个请求的位置, 循环状态端点和每个请求的投影形状.
`OH_MY_VLLM_DEEPSEEK_BATCH_DECODE=1` 选择此路径.
默认关闭.

DEP4 在专家交换前, 对每层和每个 call 记录一次路由.
Hook 为嵌套的专家调用保留一次路由记录.
已接受 frontier 排除重复行.
所有 rank 完成 Execute 后, 一次 MAX 合并专家 presence, 一次打包读回提供计数.

TP4/TEP4 计数排除重算的历史行.
[Trace 契约](models/deepseek_v41/accounting.zh.md#运行时-trace-契约)规定混合阶段的错误处理.
Batch 检查点和 HTTP trace 校验未完成.

DeepSeek startup 只读取 `config.json` 和 `model.safetensors.index.json` 生成 descriptor identity.
Ready 之前不会 hash weight shard.

Rank-local checkpoint 加载使用 65536 行 tile, 16 字节对齐的连续参数和有界 staging.
它复制所需的 shard 范围, 并跳过未使用的 Engram payload.
`OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` 选择 tile 大小.
`OH_MY_VLLM_STARTUP_TRACE` 选择外部 startup trace.

DeepSeek 使用 [v3.1 参考](models/deepseek_v41/accounting.zh.md) 绑定 HTTP trace.
Harness 绑定 `GPU-` UUID, worker 身份, 计数和同一个可配置的 client/service 超时.
v2/v3 仍可复算.
完整 workload 尚未验收.
下一步: 使用最终 source 身份运行 [matrix](../development/serving/deepseek_matrix.py) 和 AgentX.

## 下一步

- 在[算子 microbench 设计](models/qwen38/operators.zh.md)中运行固定的 Qwen operator case.
- 使用 decode profile 选择下一项 host launch 和 collective 优化.
- 在 approved long-context workload 中测量每个 state endpoint 的 window storage.
- 比较 persistent encoded KV storage 与直接 BF16 table read 的测量结果.
- 在 long-context capacity 调整前, 测量 raw 和 prepared projection storage.
- 性能验收前, 保留独立且固定的 DeepSeek 文本端点检查.
- 在长上下文容量调优前, 测量 TP4 激活和输出的收集操作及其临时存储.
- 在 long-context capacity 调整之前, 减少重复的 cuBLAS operator workspace storage.
- 在 image acceptance 之前增加独立的 image endpoint.
- 分别报告配置的 context limit 和观测到的 AgentX request length.
- 为并发 16 和 100K 至 200K token trace 调整 AgentX physical block 和 temporary operator workspace capacity.
- 用四 rank 检查点输入验证批量 decode, 每个请求的投影布局, Graph 端点和跨请求 verification.
- 增加 physical cache 和 sequence capacity 后, 运行更高容量的 AgentX case.
- 完成 DEP4 checkpoint 数值, 状态, 容量和 HTTP 验收.
- 将 native DeepSeek path 与 reference path 和公开 baseline 做性能比较.
- 将 large-row native temporary storage 移入 adapter-owned graph operator workspace.
- 用 owner trace 记录校验 DEP4 HTTP manifest.
- 将 DeepSeek Engram lookup 移入 Graph, 或使其 host copy 异步.
- 依据实测 collective 成本评估 replicated `wo_b`.
- 用 Nsight Compute 分析 top non-collective decode kernel.
- 运行批准的 DeepSeek AgentX case, 发布 CSV 和暗色背景散点图.
- 在没有锁频的显卡上运行 38 个 HTTP case 和 3 个 AgentX case.
- 交付前取得 independent code review.

原始报告和临时 process records 保留在 repository 外部.
Host settings 和 model paths 放在 `LOCAL.md` 中.
