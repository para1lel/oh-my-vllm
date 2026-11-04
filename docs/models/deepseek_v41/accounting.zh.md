# DeepSeek 参考计量

HTTP 参考使用 `development/serving/deepseek_accounting.py` 中的 `deepseek-http-reference-v3.1`.
文件 `deepseek_accounting_v2.py` 和 `deepseek_accounting_v3.py` 保留早期版本的复算能力.
使用 `reference_case(..., version="deepseek-http-reference-v3")` 计算 v3.

## 计入规则

资源参考是对相同已接受输出及指定 TP4, TEP4 或 DEP4 策略的下界松弛.
集合通信延迟部分是实测估算, 不是经过证明的物理下界.
效率为 `reference seconds / full HTTP seconds`.

DeepSeek 的目标是每个已批准的 8192 输出用例都至少达到 33%.

每次 decode 调用为每个请求计入已接受 draft 加一个 target token.
draft 提议, 被拒行, padding, mask 槽位, 临时拷贝, HTTP, 排队和调度均为零计入.
Qwen 计入其固定 MTP4 工作, 包括提议和被拒验证行.

因此两个目标使用不同的计入口径.
复制工作给每个 rank 计一份.

## 速率与格式

| 资源 | 每个 rank 的速率 |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 每方向 900 GB/s |
| PCIe 5.0 x16 主机链路 | 每方向 63.02 GB/s |

Dense 投影使用 FP8. Routed 专家保留 FP4 存储并使用 FP8 点积.
FP4 每个值占半字节, 每个 K32 组有一个 scale.
FP8 每个值占一字节, 每个 32 乘 32 block 有一个 E8M0 scale.

`wo_a`, 稀疏注意力和 index 分数使用 BF16.
Router, Hyper-Connections, Engram gate, 输出头和 ratio 大于一的 compressor 使用 FP32.
投影组包含输入, 输出和权重字节.

版本和速率摘要绑定格式, 速率, 放置, 复用及重叠规则.

### 必要的 K32 scale 累加

发布的 [FP8 kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py#L269) 和 [FP4 kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py#L553) 在下一个组之前清零各 K32 raw dot.
它们将 `raw * sa * sb` 加到有序 FP32 累加器中.
跨组的 tensor-core 累加有不同的舍入.

每个输出和 K32 组至少需要一次 FP32 FMA, 即两个 FP32 FLOPs.
对于发布的 K32 对齐形状, v3.1 按 75 TFLOPS 增加此工作:

```text
scale_fp32_flops = 2 * rows * local_outputs * (local_K / 32)
                 = projection_fp8_flops / 32
```

这适用于各策略, 各 rank 的全部 FP8 dense 和 FP4 routed 投影, 包括 prefill 和 decode.
该最小值允许安全合并 scale, 但不声称合并对所有输入都安全.
不能安全合并时需要更多工作. BF16 和 FP32 投影不增加此信用.


## 稀疏工作与放置

位置 `p` 有 `min(128,p+1)` 个窗口对.
压缩层增加 `min(512,floor((p+1)/r))` 个对.
Index-source 层 2, 8, 14 和 20 对全部可达压缩位置打分.

第 24, 28, 32 和 36 层每个 query 最多打分 16384 个候选.
按 head 分区的 FP32 分数归约只包含这些有效位置, 包括 prefill 时的因果限制.

TP4 和 TEP4 在每个 index-source 层比较 head 分区与 query 分区.
Head 分区每个 rank 使用八个 index head, 并对打包分数做 all-reduce.
Query 分区从投影分片收集 BF16 query 向量和 head 权重, 然后由各 query owner 计算全部 32 个 head.

它为注意力收集有效的 top-512 INT32 位置.
Query 归属可以将全部请求和位置划分到各 rank.
Query 价格使用分数资源下限: 总 score FLOPs 和必要历史 key 字节除以四.
这是对具体 query 放置的松弛, 不是可达到的放置.
Query tensor 和 HBM 工作使用全局资源池, 不强迫已有专家热点的 rank 承担平均工作量.

如果第 20 层使用 query 分区而后续某层使用 head 分区, 则通过 gather 传输选定的候选 block ID.
如果后续各层也使用 query 分区, 候选 ID 留在 query owner 上.
计算器在最终重叠公式下比较所有放置组合的计算, HBM 和集合通信工作.

Dense 与专家的 TP/EP 定义保持不变.

Engram 比较分片打包主机读取加 BF16 all-reduce, 与每个 GPU 完整读取打包主机行.
每个打包行占 264 字节.
两种方案都计入主机传输和 HBM 工作; 投影与 gate 工作为共同项.

长度不能确定重复 n-gram 或 hash 碰撞.
因此每个活跃 Engram 层和调用只计入 24 个互不相交的列行, 并允许全部 token 和请求之间完全复用.
这是保守下限, 不是观测到的唯一行数.

更精确的计入需要 token 或 hash 身份.
DEP4 使用完整的本地主机读取和本地 index 打分.

## 历史数据与权重

历史数据在每次 target 调用, 每个共享 source 和使用它的 rank 上读取一次.
该 rank 上的全部 query 和 head 复用这次读取.
Window KV 在每层不同. Compressed KV 和 index key 可以跨层使用共享 source.

首次使用计入完整的必要历史读取.
后续使用 `U` 个共享字节时, 至少从 HBM 读取 `max(0,U-128MiB)` 字节.
[Qwen 算子缓存规则](../qwen38/operator-performance.zh.md) 提供此 128 MiB 容量预算.
参考允许对选中子集最有利的保留方式, 但不允许无限 L2 存储.
Query 归属改变时, 必要子集可能增加.
放置比较包含此复用信用.

请求区间 `[s,e)` 的历史窗口并集有 `min(s,127)` 行.
令 `n=floor(e/r)-floor(s/r)` 为新增压缩行数.
选中历史 KV 至少有 `max(0,min(512,floor(e/r))-n)` 行.

无候选上限的 index source 读取 `floor(s/r)` 个历史 key.
候选 source 至少读取 `max(0,min(16384,floor(e/r))-n)` 个历史 key.
每个 KV 行包含 512 个 BF16 值; 每个 index-key 行包含 128 个 BF16 值.

这些规则允许同一调用内最大程度复用选择结果.
它们不为 QK 和 PV 增加第二次读取.
当前投影已包含新 cache 的写入.

新增 index-key 投影使用每个请求的位置和压缩边界.
各 index-key 权重在首次产生新 key 的 chunk 计入.

Prefill 在各 chunk 之间只计一次 dense 权重.
对每层和每个 rank, 各 chunk 中 touched-expert 计数的最大值是权重并集的保守下界.
每个 chunk 只计入 touched 计数累计最大值的增量.
总量仍等于最大计数, 不会在较早 chunk 计入之后才使用的专家权重.
此计入不由请求归属控制.

输出头权重在真正计算 logits 的首个 chunk 计入.
每个请求仅最后一个 prefill 行计算 logits.
Decode 每次调用计一次常驻 dense 权重和被访问专家权重.

## 通信与重叠

TP4 包括 embedding 归约, `wo_b` 归约, routed BF16 激活乘积 gather, FP32 token 求和 gather 和输出头 gather.
TEP4 将 routed gather 换为 routed FP32 输出归约.
DEP4 dispatch 发送不同 `(token,destination GPU)` 的 BF16 行.

目标端先对本地专家输出求和, 再为每个分发行返回一个 FP32 行.
本地路由不产生 wire 字节.
每个 rank 的 wire 压力为 `max(sent,received)`; 每个方向有独立链路速率.

专家工作逐层使用实际接收分配, 包括没有自有请求的 rank.

对于串行组, `S=sum_g max_rank(time(rank,g))`.
每组使用最大的资源时间; FP8 和 BF16 共享 tensor 资源, 时间相加.
至少有两个计入 target 行时, 调用价格为:

```text
R_fixed = max_rank,resource(sum(fixed_work) / rate)
R_pool = max_resource((sum_rank(fixed_work) + sum(query_work)) / (4 * rate))
R = max(R_fixed, R_pool)
B2 = max(R, S / 2)
call_seconds = min(S, B2)
```

资源总量还包含集合通信估算秒数之和.
单个 target 行使用 `S`.
`B2` 是均衡两微批的下界松弛, 不是可达到的执行时间.
它通过一半串行链保留微批内部依赖, 并保留全部资源总量.

拆分允许权重 tile 在被移出前供两个微批复用.
因此权重并集只计一次; 不要求读取两遍完整权重.
重新加载权重的日程有更多工作.

通信字节和估算延迟仍保留在资源总量中.

每次集合通信使用 `max(latency_model_seconds,wire_bytes/900e9)`.
传输时间已在实测延迟内, 因此不再次相加.
版本绑定的空闲未锁频 NCCL profile 保留 95 个已存中位数点.

原始 SHA256 为 `c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49`.
查表使用每 rank 输入字节, 累计最大值, 线性插值和端点 clamp.
All-gather wire 字节使用完整输出大小.

DEP4 在同等字节大小下使用 BF16 all-to-all proxy, profile 输入为 `ceil(wire_bytes*4/3)`.
FP32 combine 保留每值四字节.
Profile 摘要绑定原始摘要和已存表.

已存中位数无法恢复各大小的样本最小值; 两种统计量都不能证明下界.
DEP4 通信仍为建模结果, 不是实测结果.

TEP4 fallback 不能提供 DEP4 证据.

## 运行时 trace 契约

顶层对象包含 `binding`, `manifest` 和 `calls`.
Manifest 在 HTTP response 全部完成后定稿.

```text
manifest = {mode: "ordinary" | "dspark5", requests: [
  {ordinal, id, response_id, input_tokens, output_tokens, owner_rank}
]}
call = {rank, call, phase: "prefill" | "decode", request_rows: [
  {id, start, count, accepted, proposed, logit_rows}
], expert_assignments: int32[L,4], dispatch_rows: int32[L,4],
   touched_experts: int32[L]}
```

`L=40`. 所有计数均为整数; 位置和 ordinal 从零开始.
ID 为正数; `response_id` 以 `chatcmpl_` 开头, 以 `_<id>` 结尾.
Manifest 长度是每请求 HTTP usage 的 token 数.

`owner_rank` 固定不变, 用于 DEP4.
每次实际 target 调用恰有四条 rank 记录, 包括没有自有请求的 rank.
每个 prefill chunk 是独立调用; 请求的 `start/count` 区间必须连续拼接, 无间隔或重叠.

`count` 只包含计入行. Decode 要求 `count=accepted+1` 且 `logit_rows=count`.
Prefill 要求 draft 计数为零, 且仅在 prompt 完成时有一个 logit 行.
TP4/TEP4 在全部 rank 重复记录请求行和分配矩阵.

DEP4 只记录各 rank 自有的请求行.

三种策略均使用全局专家分桶 `d=expert_id//96`.
每个矩阵行之和为 `credited rows*6`, 不是此值的四份副本.
TP4 计算全部四列; TEP4 计算自己的列.

DEP4 计算全部来源 rank 记录中自身列的和.
`dispatch_rows[layer,d]` 统计不同来源 token/目标对, 包括本地对.
DEP4 之外该值为零. Combine 在本地求和后反向返回同样的对.

`touched_experts[layer]` 只统计计入行访问的本地专家权重.
设备可使用已接受行 mask 累加这三个数组, 包括在 CUDA Graph 内.
DEP4 将 expert-presence MAX 推迟到所有 rank 完成 Execute 后.
它将全部全局调用打包为一次 MAX 和一次计数读回.
空 rank 提供零来源路由, 并接收专家并集.
其他策略在 call 或请求结束后读回计数.

稀疏对数, index 对数和历史字节由位置及模型配置计算.

Binding 包含 `version`, `rate_profile`, `manifest_sha256`, source, native 模块摘要, checkpoint config/index 摘要, tokenizer 和四个 GPU UUID.
它还包含 strategy, effective strategy, owner plan, sampling, case ID, 以及 accepted/proposed 总计数.
Manifest 摘要使用键排序的紧凑 JSON 的 SHA256, 不允许 NaN.

校验拒绝身份改变, rank 缺失, 调用重复, 无效路由范围和层不完整.
逐请求状态检查完整 prefill 覆盖, 接受位置连续, 输出数准确, 以及无重复工作, 即使更换 call ID 也不例外.

一个 v3.1 call 只有一个阶段.
混合阶段的 target call 发出 `accounting_error`, 并提供 call ID 和原因.
推理继续, 不生成部分计数或额外的 call 记录.
此事件不是 v3.1 call.
只要任一 rank 记录此错误, HTTP assembler 就拒绝完整 trace.

## 已知松弛

Query 归属和两微批总量允许分数均衡.
它们不证明存在可达到的日程.
没有 token 身份时, 历史选择使用其最小可能并集.
Engram 允许在同一调用内完全复用 hash 行.

Prefill 权重计入使用最大计数, 而不是实测专家并集.
128 MiB 缓存预算可以有利于历史数据, 不为其他工作预留空间.
这些规则可能降低参考值.
集合通信 profile 是与版本绑定的经验估算.

## 理想参考值

19 个长输出用例使用理想接受, 均衡的合成路由和期望不同专家数.
它们是参考尺度, 不是实测 HTTP 效率结果.
每个 DSpark5 调用接受全部五个 draft, 最后截断的调用除外.

报告实测效率前, 运行时必须提供绑定 trace.

| 用例 | v3/lower_bound s | v3.1/lower_bound s |
|---|---:|---:|
| `dspark5:tp4:4:400000:8192` | 12.996 | 17.417 |
| `dspark5:tp4:4:800000:8192` | 21.819 | 29.933 |
| `dspark5:tp4:16:400000:8192` | 43.428 | 61.115 |
| `dspark5:tp4:16:800000:8192` | 78.722 | 111.179 |
| `dspark5:tp4:64:400000:8192` | 144.266 | 215.015 |
| `dspark5:tp4:64:800000:8192` | 285.440 | 415.269 |
| `dspark5:tep4:4:400000:8192` | 12.794 | 17.407 |
| `dspark5:tep4:4:800000:8192` | 21.427 | 29.923 |
| `dspark5:tep4:16:400000:8192` | 42.692 | 61.144 |
| `dspark5:tep4:16:800000:8192` | 77.221 | 111.207 |
| `dspark5:tep4:64:400000:8192` | 141.087 | 214.895 |
| `dspark5:tep4:64:800000:8192` | 279.203 | 415.148 |
| `dspark5:dep4:4:400000:8192` | 10.421 | 12.152 |
| `dspark5:dep4:4:800000:8192` | 16.284 | 18.287 |
| `dspark5:dep4:16:400000:8192` | 29.712 | 36.636 |
| `dspark5:dep4:16:800000:8192` | 53.166 | 61.179 |
| `dspark5:dep4:64:400000:8192` | 85.678 | 113.374 |
| `dspark5:dep4:64:800000:8192` | 179.494 | 211.546 |
| `ordinary:4:400000:8192` | 27.932 | 32.354 |

使用[测试范围](../../testing.zh.md)和[报告规则](../../testing.zh.md#记录).
