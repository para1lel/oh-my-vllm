# DeepSeek 参考计量

[HTTP 流程](performance.zh.md)选择计算器并验证实测结果.
效率为 `reference seconds / full HTTP seconds`.

| 计算器 | 版本 | 用途 |
|---|---|---|
| 默认参考和 source trace | `deepseek-http-reference-v3.1` | 历史估算和经身份校验的 target 计数 |
| Packed 协议估算 | `deepseek-http-protocol-estimate-v5` | 历史协议估算, 不是延迟下界 |
| 有条件部分下界 | `deepseek-http-resource-floor-v4` | 必要算术和有限的 MoE wire |
| 固定算法 roofline | `deepseek-http-roofline-v6` | 当前 HTTP 效率参考 |

[Source validator](../../../development/serving/deepseek_accounting.py) 对四种计算器执行身份校验.
[Roofline 计算器](../../../development/serving/deepseek_accounting_roofline.py)使用经过审查的[部分算术](../../../development/serving/deepseek_accounting_floor.py).
[资源函数](../../../development/serving/deepseek_accounting_math.py)规定维度, 格式和峰值.

V4 报告 `dep4_communication_modeled=False`, 因为它的 DEP4 协议字节仅为诊断值.
V5 和 v6 对 DEP4 报告 `dep4_communication_modeled=True`.
对 TP4 和 TEP4, 这些计算器报告 `dep4_communication_modeled=False`.
此标记表示所选计算器在所选策略下的能力, 与 trace 的通信量无关.
V4, v5 和 v6 报告 `dep4_communication_measured=False`, 因为它们的通信估计是数学估计.

## Roofline 范围

V6 是固定 target 算法, 数值规则和四个 rank 放置方式下的有条件资源下界.
它不是所有相同输出算法的最小值.
这些条件允许最大重叠和有利的数据复用.
它们禁止额外权重副本, 投影重算, wire 压缩及跨 token 的 wire 值复用.
TP4 保持专家的输出通道分片.
TEP4 保持专家 owner 和 tensor-parallel attention.
DEP4 保持请求 owner, 并在专家 owner 上存放完整专家.

每个请求计入 `input_tokens + output_tokens - 1` 个 target row 和 `output_tokens` 个 logit row.
Proposal, 被拒行, padding, 拷贝, scheduler 成本和 MTP 工作均为零计入.
复制矩阵在每个使用它的 rank 上计一份.
专家算术使用收到的 assignment, 包括没有自有请求的 rank.

| 资源 | 每个 rank 的峰值 |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 每方向 900 GB/s |
| PCIe 5.0 x16 | 每方向 63.02 GB/s |

[NVIDIA Blackwell 数据表](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf#page=8) 第 8 页的 HGX B200 列给出每个 GPU 的 HBM, FP32 和 tensor 峰值.
脚注 2 规定 dense tensor 峰值为表列 sparse 数值的一半.

FP4 专家存储使用 FP8 dot product.
FP8 和 BF16 共享 tensor 资源.
[有序 K32 算术](quantization.zh.md)至少需要额外 `FP8_FLOPs / 32` 个 FP32 FLOPs.
`wo_a`, attention product 和 index score 使用 BF16.
Router, HC projection, Engram gate, head 及 ratio 大于一的 compressor 使用 FP32.
非线性标量估算为零计入.

各 rank 的 tensor 秒数为 `FP8_FLOPs/4.5e15 + BF16_FLOPs/2.25e15`.
其他资源均用资源量除以对应峰值.
参考对所有 rank 和资源取最大值.
TP4 和 TEP4 的 index query 工作允许使用分数化的四个 rank 资源池.
池化的必要 wire 和 host 字节除以对应峰值的四倍.
公式没有经验延迟, 按实测调用划分的串行和, 或 `S/2` 项.

## HBM 和 host 的必要性

每个 rank 的保留额度为 `F = 1 GiB`, 其中 `1 GiB = 2^30 bytes`.
它包含寄存器, 数据缓存, 共享内存和 tensor memory.
它不包含 HBM, 位于 HBM 的 Torch 分配或主机内存.

[NVIDIA 指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)给出 126 MB L2, 每个 SM 的 256 KiB 寄存器和 256 KiB 联合 L1/共享内存.
[B200 布局](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/blackwell_chip)给出保守的 160 个 SM 物理布局.
[SM 说明](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm)给出每个 SM 的 256 KiB tensor memory.
把 L2 按 126 MiB 计算时, 这些数据存储共为 246 MiB.
1 GiB 额度还包含其余小型缓存和有利的保留.

令 `W_union` 为单个 rank 在整次 HTTP 中的必要权重并集.
Assignment 计数提供专家并集的界限, 不假设每次调用使用新专家.
令 `W_dense` 只包含每轮 target 都需要的矩阵, 包括 head.
间歇使用的 compressor 和 index-key 矩阵不进入 `W_dense`.

Ordinary 模式每轮 target 最多产生一个新行, DSpark5 最多产生六个.
单个请求至少需要 `N = 1 + ceil((output_tokens-1)/limit)` 轮, 包括 prompt 完成时的那一轮.
该请求的下一轮依赖前一轮的结果.
每轮都需要稠密矩阵; 从该时间区间之外保留下来的权重最多为 `F` 字节.
每个 rank 使用最长的自有请求链:

```text
weight_read_bytes = max(0, W_union-F, N*max(0,W_dense-F))
```

并集和依赖链可以描述同一批读取, 因此两者不能相加.
空的 DEP4 owner 仍计入其收到的专家权重并集.

[原生历史](cache.zh.md#当前存储)使用 288 字节 compressed KV, 68 字节 index key 和 1024 字节 BF16 window.
四个物理 source 各计一次; 使用方不产生新的存储.
对各请求, 令 `P` 为最终编码 KV 和 index 负载.
初始为空的状态必须发布这些历史, 且不允许重算.
最多 `F` 字节可以保持未写入, 因此每个请求的必要 cache write 为 `max(0,P-F)`.
权重读取和 cache write 描述不同传输, 可以相加.
没有必要性证明时, 重复历史读取, window write, workspace 解码和中间 activation 流量均为零计入.

Engram 初始位于 host 存储.
每个活跃层的 24 个不同列至少需要 24 个 packed row, 每行为 264 字节.
参考允许所有 token 和请求完全复用这些 row.
TP4/TEP4 将首次 host 传输池化; 各活跃 DEP4 owner 需要自己的本地 row.
没有 token 和 hash 身份时, 重复 lookup 为零计入.

## 通信和依赖

下界与完整协议诊断分开.
在固定的完整 BF16 MoE 交付方式下, TP4/TEP4 每层至少需要合计 `3*rows*H*2` 个 wire 字节.
TP4 还需要 `3*rows*K*I*2` 字节交付 activation product.
其中 `H=5120`, `K=6`, `I=2304`.
这些总量界限允许有利的 rank 平衡和最终 BF16 交付.
它们不把完整 FP32 reduction 当作必要信息计入.
Embedding, `wo_b`, head, index 和 Engram wire 的下界计入为零.
贪心选择不需要完整 vocabulary gather.

DEP4 使用直接 BF16 dispatch 和反向 FP32 本地和交付.
令 `D[s,d]` 统计不同的已计入 `(source token,destination rank)` pair, 包括本地 pair.
对 rank `r`, 排除 `s=d`:

```text
out_r = sum_d D[r,d]
in_r  = sum_s D[s,r]
sent_r     = H*(2*out_r + 4*in_r)
received_r = H*(2*in_r  + 4*out_r)
```

先在整次 HTTP 工作负载中对每个方向分别求和, 再取资源最大值.
不要把独立的 dispatch 和 combine 最大值相加作为串行下界.
本地 route, metadata, count exchange, padding 和 trace reduction 均为零计入.

协议诊断将各个有效 collective 放在对应的依赖位置:

| Strategy | 有序依赖 |
|---|---|
| TP4/TEP4 | Embedding lookup, BF16 reduction, 第一层 |
| TP4/TEP4 | Engram lookup, BF16 reduction, Engram projection, 该层 attention |
| TP4/TEP4 | Index scoring, FP32 reduction, sparse attention |
| TP4/TEP4 | `wo_b`, FP32 reduction, router |
| TP4 | Expert up/gate, BF16 product gather, down, FP32 token-sum gather |
| TEP4 | Expert up/gate, down, FP32 output reduction |
| DEP4 | Router, BF16 dispatch, expert up/gate, down, 反向 FP32 combine |
| TP4/TEP4 | Head projection, FP32 vocabulary gather, sampling |

不同行可以重叠这些阶段.
V6 计算资源下界, 未计入依赖延迟下界.
`dependency_seconds=0` 字段表示依赖延迟未计入.
Target 算法仍有依赖.
总资源约束在最大重叠下仍为必要条件.
V6 不计算完整的最小物理延迟.

对四个 rank, ring all-reduce 的 `N` 字节输入使各 rank 分别发送和接收 `1.5*N` 字节.
Ring all-gather 的 `N` 字节完整输出使各 rank 分别发送和接收 `0.75*N` 字节.
有四个 target row 时, TP4 product gather 在各 rank 每方向使用 82944 字节.
其 FP32 token-sum gather 使用 61440 字节; TEP4 output reduction 使用 122880 字节.
当 `D[0,1]=2` 且 `D[1,0]=1` 时, rank 0 的 dispatch 发送 20480 字节, 接收 10240 字节.
其反向 combine 发送 20480 字节, 接收 40960 字节.
这些协议字节是诊断值, 不额外增加下界计入.

## Trace 和身份

顶层 trace 包含 `binding`, `manifest` 和 `calls`.
HTTP 响应使 manifest 最终定稿:

```text
manifest = {mode: "ordinary" | "dspark5", requests: [
  {ordinal, id, response_id, input_tokens, output_tokens, owner_rank}
]}
call = {rank, call, phase: "prefill" | "decode", request_rows: [
  {id, start, count, accepted, proposed, logit_rows}
], expert_assignments: int32[40,4], dispatch_rows: int32[40,4],
   touched_experts: int32[40]}
```

每次调用需要四个 rank record, 包括空 owner.
请求范围连接后无缺口或重叠.
Decode 计入 `count=accepted+1`; prefill 仅在 prompt 完成时计入一个 logit row.
TP4/TEP4 重复请求和 assignment 计数; DEP4 只记录自有请求.
每层的每个已计入行有六个 assignment, 按 `expert_id//96` 分桶.
TP4 使用全部 bucket, TEP4 使用自己的 bucket, DEP4 使用跨 source rank 求和后的自己的 bucket.
DEP4 在 Execute 之后合并专家存在性, 并记录不同的 dispatch pair.
验证拒绝缺失 rank, 重复工作, 无效 route, 不完整层和错误的 HTTP 覆盖范围.
混合 phase 发出 `accounting_error`; assembler 拒绝该 trace.

Binding 包含 source, native module, checkpoint, tokenizer, 四个 GPU UUID, 放置方式, sampling 和 manifest digest.
Runtime identity 记录 batch 和 Graph 设置, 但不记录每步的 capture 或 Eager 回退.
V6 独立绑定其版本, 峰值, 模型维度, 保留额度和假设摘要.
验收检查从绑定的 trace 重新计算所选参考.
身份或已存数值变化会导致验证失败.

V6 资格检查要求测量 topology 中有四个 B200 型号名称.
缺少名称或使用其他型号均不能通过.
历史诊断不使用此硬件检查.

## 历史估算

V3.1 保留 BF16 历史流量和经验 NCCL 中位数延迟.
[Packed v5 计算器](../../../development/serving/deepseek_accounting_packed.py)保留每调用权重重载和双 microbatch 估算.
这些身份和计算保持不变, 用于旧记录.
V5 的协议阶段没有包含分片 Engram reduction.
其位于层末的 Engram 放置方式也与 runtime 的层入口依赖不同.
V6 诊断修正这些细节, 不增加下界计入.

将相同已接受工作拆成更多调用会增加 v5 秒数.
估算标签和 33% 比值都不能使 v5 成为延迟下界.
V4 对 HBM, host 和 DEP4 wire 为零计入, 仍是部分参考.
[验收流程](performance.zh.md#检查长输出行)要求当前效率验收使用 v6.
静态用例规模是估算, 不是 HTTP 证据, 也不是数值, 容量或 AgentX 验收.
