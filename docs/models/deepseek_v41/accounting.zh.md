# DeepSeek 参考计量

[HTTP 流程](performance.zh.md)选择参考并验证实测结果.
各计算器具有独立身份:

| 计算器 | 版本 | 源码 |
|---|---|---|
| 默认 HTTP 参考和 source trace | `deepseek-http-reference-v3.1` | [Trace 和 binding](../../../development/serving/deepseek_accounting.py), [资源数学](../../../development/serving/deepseek_accounting_math.py) |
| Packed 协议估算 | `deepseek-http-protocol-estimate-v5` | [Packed 计算](../../../development/serving/deepseek_accounting_packed.py) |
| 有条件部分资源下界 | `deepseek-http-resource-floor-v4` | [下界计算](../../../development/serving/deepseek_accounting_floor.py) |

## 计入规则

默认资源参考是对相同已接受输出及指定 TP4, TEP4 或 DEP4 策略的下界松弛.
集合通信延迟是经验估算, 不是经过证明的物理下界.
效率为 `reference seconds / full HTTP seconds`.

每次 decode 调用为每个请求计入已接受 draft 加一个 target token.
Proposal, 被拒行, padding, mask slot, 临时拷贝, HTTP, 排队和调度均为零计入.
复制工作给每个 rank 计一份.

## 速率与格式

| 资源 | 每个 rank 的速率 |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 每方向 900 GB/s |
| PCIe 5.0 x16 host link | 每方向 63.02 GB/s |

稠密投影使用 FP8.
路由专家保留 FP4 存储并使用 FP8 dot product.
FP4 每值使用半字节, 每个 K32 group 使用一个 scale.
FP8 每值使用一字节, 每个 32 乘 32 block 使用一个 E8M0 scale.
投影 group 计入输入, 输出和权重字节.

`wo_a`, sparse attention 和 index score 使用 BF16.
Router, Hyper-Connections, Engram gate, output head 和 ratio 大于一的 compressor 使用 FP32.
版本和速率摘要绑定格式, 速率, 放置, 复用和重叠规则.

[有序 K32 算术](quantization.zh.md)为每个输出及 K32 group 至少增加一个 FP32 FMA:

```text
scale_fp32_flops = 2 * rows * local_outputs * (local_K / 32)
                 = projection_fp8_flops / 32
```

此计入适用于每个 rank 在 prefill 和 decode 中的 FP8 稠密投影和 FP4 路由投影.
它允许精确的 scale 合并, 但不假定所有 scale pair 都允许合并.
BF16 和 FP32 投影没有 K32 scale 计入.

## 稀疏工作与放置

位置 `p` 有 `min(128,p+1)` 个 window pair.
压缩层增加 `min(512,floor((p+1)/r))` 个 pair.
Index source 2, 8, 14 和 20 对所有可达压缩位置评分.
Source 24, 28, 32 和 36 每个 query 最多评分 16384 个 candidate.

TP4 和 TEP4 在每个 index source 比较 head 分片与 query 分片.
Head 分片每个 rank 使用八个 head 和 packed FP32 score all-reduce.
Query 分片收集 BF16 query 和 head weight, 在各 query owner 计算全部 32 个 head, 并收集有效 top-512 INT32 位置.
其 score FLOPs 和必要 key 字节在全局资源池中除以四.
此分数化平衡不证明可实现的放置.

Layer 20 之后从 query 转到 head 分片时, 收集 candidate block ID.
后续 query 分片将这些 ID 保留在 query owner.
计算器按资源, 通信和重叠成本比较放置组合.
稠密和专家分片定义保持固定.

Engram 比较两种方式: 分片 packed host read 加 BF16 all-reduce, 或每张 GPU 完整读取 packed host row.
每个 packed row 为 264 字节.
没有 token 或 hash 身份时, 每个活跃层和调用只计入 24 个不同列的 row.
它允许请求和 token 之间完全复用.
DEP4 使用完整本地 host read 和本地 index 评分.

## 历史与权重

每次 target 调用, 每个共享物理 source 和使用它的 rank, 历史只读一次.
该 rank 上所有 query 和 head 复用该次读取.
Window 每层不同.
Compressed KV 和 index key 可以共享 source.

首次使用计入完整必要历史读取.
后续使用 `U` 个共享字节时, 计入 `max(0,U-128MiB)` 个 HBM 字节.
此 cache allowance 允许有利的保留, 但不允许无限 L2 存储.
放置比较计入必要子集的变化.

对请求范围 `[s,e)`, 历史 window 并集有 `min(s,127)` 行.
令 `n=floor(e/r)-floor(s/r)` 为新增压缩行数.
历史选中 KV 至少有 `max(0,min(512,floor(e/r))-n)` 行.
无上限 index source 读取 `floor(s/r)` 个历史 key.
Candidate source 至少读取 `max(0,min(16384,floor(e/r))-n)` 个历史 key.

默认计算为每个 KV row 计入 512 个 BF16 值, 为每个 index-key row 计入 128 个 BF16 值.
同一调用内最大化选择复用, 不为 QK 和 PV 计第二次读取.
当前投影计入新 cache write.
新 index-key 投影使用各请求的位置和压缩边界.

Prefill 跨 chunk 只计一次稠密权重.
每层每个 rank 的累计最大 touched-expert 数提供保守的权重并集界限.
一个 chunk 只计入该最大值的增量.
Output head 在首个有 logits 的 chunk 计入, 且只在每个请求最后一行 prefill 计算.
Decode 每次调用计入一次常驻稠密权重及 touched expert 权重.

## 通信与重叠

TP4 计入 embedding reduction, `wo_b` reduction, 路由 BF16 product gather, FP32 token-sum gather 和 output-head gather.
TEP4 将路由 gather 换成 routed FP32 output reduction.
DEP4 发送不同的 `(token,destination GPU)` BF16 row, 每个已 dispatch token 返回一个 FP32 本地和.
本地 route 不使用 wire byte.
每个 rank 的 wire pressure 为 `max(sent,received)`, 各方向有独立速率.
专家工作使用收到的 assignment, 包括没有自有请求的 rank.

对于串行 group, `S=sum_g max_rank(time(rank,g))`.
每个 group 使用最大资源时间.
FP8 和 BF16 时间在共享 tensor 资源上相加.
至少有两个计入的 target row 时:

```text
R_fixed = max_rank,resource(sum(fixed_work) / rate)
R_pool = max_resource((sum_rank(fixed_work) + sum(query_work)) / (4 * rate))
R = max(R_fixed, R_pool)
B2 = max(R, S / 2)
call_seconds = min(S, B2)
```

单个 target row 使用 `S`.
平衡双 microbatch 松弛允许共享权重 tile 和分数化资源平衡.
它不证明可实现的调度.
资源总量还计入 collective 估算秒数.

每次 collective 使用 `max(latency_model_seconds,wire_bytes/900e9)`, 不再计第二次传输.
空闲且未锁频的 NCCL profile 有 95 个已存中位数点.
原始 SHA256 为 `c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49`.
查表使用每个 rank 的输入字节, 累计最大值, 线性插值和端点截断.
All-gather wire byte 使用完整输出大小.
DEP4 使用 BF16 all-to-all 代理, profile 输入为 `ceil(wire_bytes*4/3)`.

Profile digest 绑定原始摘要和存储表.
中位数不能重建样本最小值或证明下界.
DEP4 通信是建模值, 不是实测值.
TEP4 回退不能提供 DEP4 证据.

## 运行时 trace 契约

顶层对象包含 `binding`, `manifest` 和 `calls`.
HTTP 响应完成后 manifest 才最终定稿.

```text
manifest = {mode: "ordinary" | "dspark5", requests: [
  {ordinal, id, response_id, input_tokens, output_tokens, owner_rank}
]}
call = {rank, call, phase: "prefill" | "decode", request_rows: [
  {id, start, count, accepted, proposed, logit_rows}
], expert_assignments: int32[L,4], dispatch_rows: int32[L,4],
   touched_experts: int32[L]}
```

`L=40`.
计数为整数.
位置和序号从零开始.
ID 为正数, `response_id` 以 `chatcmpl_` 开始, 以 `_<id>` 结束.
Manifest 长度为各请求的 HTTP usage token 数.
DEP4 `owner_rank` 保持固定.

每次 target 调用有四个 rank record, 包括没有自有请求的 rank.
每个 prefill chunk 是独立调用.
请求的 `start/count` 范围连接后无缺口或重叠.
Decode 有 `count=accepted+1` 和 `logit_rows=count`.
Prefill 的 draft 计数为零, 且仅在 prompt 完成时有一个 logit row.

TP4/TEP4 在所有 rank 重复 request row 和 assignment matrix.
DEP4 只记录自有 request row.
Assignment 使用全局 bucket `d=expert_id//96`.
每个 matrix row 的和为 `credited rows*6`.
TP4 计算全部四列, TEP4 计算自己的列, DEP4 计算跨 source rank 求和后的自己的列.

`dispatch_rows[layer,d]` 统计不同的 source token/destination pair, 包含本地 pair.
其他策略该值为零.
Combine 在本地求和后反向使用这些 pair.
`touched_experts[layer]` 只统计计入行使用的本地专家权重.
DEP4 在 Execute 后使用一次 packed MAX 和计数读取合并专家存在性.
空 rank 提供零 route, 并收到专家并集.

Binding 包含 `version`, `rate_profile`, `manifest_sha256`, source, native module digest, config/index digest, tokenizer 和四个 GPU UUID.
它还包含 strategy, effective strategy, owner plan, sampling, case ID 和 accepted/proposed 总量.
Manifest digest 使用无 NaN 值的排序紧凑 JSON 的 SHA256.

验证拒绝身份变化, 缺失 rank, 重复调用, 无效路由, 不完整层, 重复工作, 以及错误请求覆盖或输出计数.
一次调用只有一个 phase.
混合 phase 发出带 call ID 和原因的 `accounting_error`.
推理继续但不产生部分计数, HTTP assembler 拒绝完整 trace.

## Packed 估算

Packed 计算器使用[物理缓存格式](cache.zh.md#当前存储).
它区分共享物理历史和每个使用方的流量.
Index workspace 解码按每个物理 source 和调用计一次.
每个 index 使用方读取已解码 key, 每个 attention 使用方读取 packed KV.
权重计入不假定未经证明的跨调用保留.
Wire 时间只使用带宽, 不使用经验 NCCL profile.

TP4 activation 交付在 expert up 计算之后, down 计算之前.
DEP4 dispatch 在专家计算之前完成.
反向 FP32 combine 在专家计算之后进行.
每个方向独立计入.
Route metadata 和 count exchange 仅作诊断, 不增加参考计入.

估算保留资源和双 microbatch 松弛.
串行协议秒数和有条件部分下界是独立诊断项.
Cache 保留, padding, launch latency, overlap 和 MTP 工作不构成完整执行时间模型.
此估算不是普适物理最小值.

`source_trace_version` 和 `source_trace_rate_profile` 绑定默认 source trace.
`accounting_version`, `rate_profile` 和 `assumptions_sha256` 绑定计算器.
Runtime identity 还记录各 rank 的 batch-decode 和 Graph 设置, 包含 Graph 数量上限.
这些设置不标识每步 capture 或 Eager 回退.

## 有条件部分下界

下界固定 target 算术, 数值次序, 格式, 参数放置和复制投影.
其 MoE wire 计入假定固定 output shard, 完整 BF16 row 交付, 无额外权重副本, 无重算, 无未建模压缩或复用.

HBM 和 host strict 计入为零.
Embedding, `wo_b`, head, index, Engram 及 DEP4 dispatch/combine wire 的 strict 计入也为零.
Proposal, 被拒工作, padding, 拷贝, scheduler 成本, 经验延迟和 `S/2` 的 strict 计入为零.
非线性标量估算的 strict 计入也为零.
Weight union 和 `direct_protocol_*` 流量仅作诊断.
此部分下界既不证明完整执行时间, 也不证明常驻容量.
