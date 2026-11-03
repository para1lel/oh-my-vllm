# DeepSeek 参考计量

DeepSeek HTTP 报告使用 `deepseek-http-reference-v2` 计量.
以前的 `deepseek-http-reference-v1` 计量通过 `wire` communication model 保留.
代码位于 `development/serving/deepseek_accounting.py`.

## 规则

参考时间是产生相同已接受输出所需时间的下界.
它按冻结的执行 trace 计算每个 rank 的必要资源工作.
它每组取四个 rank 的最大值.
它把串行组相加.

参考不计排队, HTTP 或调度.
它不计 padding 或 mask 槽位.
它不计每个 prefill chunk 的权重读取或临时拷贝.
它不计 draft 提议或被拒行.
复制计算只给每个 rank 计一份.
计算使用代价最小的合法放置.

## 速率

| 资源 | 速率 |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 每方向 900 GB/s |
| 空闲未锁频集合通信延迟 | 延迟模型与 NVLink 字节时间的较大值 |
| 主机链路 | 每方向 63.02 GB/s |

主机链路是 PCIe 5.0 x16.
计量身份保存速率配置摘要.
速率, 格式, 重叠规则或计入规则改变时, 使用新版本号.

## 格式

| 工作 | 计价 |
|---|---|
| Dense 投影 | FP8 |
| Routed 专家 | FP4 存储, FP8 点积 |
| `wo_a`, 稀疏注意力, indexer 打分 | BF16 |
| Router, Hyper-Connections, `ratio > 1` 的 compressor, Engram gate, 输出头 | FP32 |

adapter 在原始点积前把 FP4 权重转为 FP8.
FP4 存储每个值占半字节, 每个 K32 组一个 scale.
FP8 存储每个值占一字节, 每个 32 乘 32 block 一个 E8M0 scale.

## 稀疏工作

位置 `p` 的查询为一个 compress ratio 为 `r` 的层选择
`min(128, p+1) + min(512, floor((p+1)/r))` 个注意力对数.

8 个 index-source 层对压缩位置打分.
第 2, 8, 14, 20 层对每个可达位置打分.
第 24, 28, 32, 36 层对每个查询最多打分 16384 个候选.
TP4 和 TEP4 划分 index 头, 然后对 FP32 分数缓冲区做 all-reduce.
每个 index-source 层使用自己的有效上下文宽度.

Engram 主机传输使用打包 FP8 行及其 scale.
每行 264 字节.
TP4 和 TEP4 为每个 token 和每层加入 24 乘 256 BF16 Engram all-reduce.
DEP4 从主机映射读取完整行, 没有 Engram all-reduce.

## Prefill 与 decode

Prefill 只读一次每个权重.
分块 prefill trace 使用各块被访问专家计数的最大值.
该值是并集的下界.
第一个有行的块承担权重读取.
参考只对每个请求的最后一个 prefill 行计输出头.

Decode 使用 trace 中的 target 调用次数和已接受计数.
每次调用计入已接受 draft 加一个 target token.
draft 提议和被拒验证行不计.
每次调用只读一次常驻 dense 权重和被访问的专家权重.

## 通信

TP4 计入激活乘积 all-gather, FP32 token 求和 all-gather, `wo_b` all-reduce,
embedding all-reduce 和输出头 all-gather.

TEP4 计入每层 routed 输出 all-reduce 和 TP4 dense 集合通信.

DEP4 按请求划分.
每个 rank 每次调用读完整 dense 权重.
参考按 BF16 行计 dispatch, 按 FP32 行计 combine.
dispatch 和 combine 的链路字节来自 4 乘 4 路由矩阵.
矩阵保存一个层的计数.
参考把链路字节乘以层数.
零行 rank 不计工作.
TEP4 fallback trace 不是 DEP4 证据.

每个 rank 的链路字节取 `max(sent, received)`.
每个方向各有 900 GB/s.
集合通信与其依赖的计算串行.

`v2` 对每次集合通信使用 `max(latency_model_seconds, wire_bytes / 900e9)`.
实测延迟已包含数据传输, 因此公式不会重复加入这部分时间.
应用 NCCL 总时间只作诊断数据.

### Profile 与字节单位

Profile 包含四个空闲且未锁频 rank 的 95 个点.
每个 rank 的中位数来自九次图重放, 每次重放含 64 次调用.
每个存储点是这四个 rank 中位数的中位数.
原始 SHA256 是 `c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49`.
Profile 不含 FP32 all-to-all 点.

Profile 查询使用每个 rank 的输入字节数.

All-gather 的链路计算使用每个 rank 的完整输出字节数.
延迟模型先对实测点取累积最大值, 然后线性插值.
低于或高于实测范围时, 模型保持最近端点的值.
模型不外推延迟斜率, 也不创建实测点.

超出范围后, 链路时间仍随字节数增加.

Profile 摘要包含原始摘要及当前内存中的点表.
v2 速率摘要包含 profile 摘要及通信规则版本.
这些摘要能检测身份变化, 但不能证明测量质量或逐点匹配原始数据.
外部原始数据检查核验全部点尺寸与四 rank 中位数.

单个 TP4 decode 行的次数和字节单位如下:

| 操作 | 次数 | 输入 B/rank | 输出 B/rank | 每次链路 B/rank |
|---|---:|---:|---:|---:|
| Routed product, BF16 all-gather | 40 | 6912 | 27648 | 20736 |
| Routed output, FP32 all-gather | 40 | 5120 | 20480 | 15360 |
| `wo_b`, FP32 all-reduce | 40 | 20480 | 20480 | 30720 |
| Embedding, BF16 all-reduce | 1 | 10240 | 10240 | 15360 |
| Output head, FP32 all-gather | 1 | 129280 | 517120 | 387840 |
| Engram, BF16 all-reduce | 2 | 12288 | 12288 | 18432 |

Product 字节数已经包含六路 routed assignment.
旧 wire 公式也包含全部六路, 对此行每层给出 20736 字节.
旧版可复现检查与物理字节检查是不同检查.
38 个 HTTP 用例保留完全相同的旧 wire 结果.

### Index 分数与理想批处理

默认 index-source 层是 `(2, 8, 14, 20, 24, 28, 32, 36)`.
第 2, 8, 14 层使用 ratio 2; 其余五层使用 ratio 1.

有效上下文长度为 `E` 时, 每层宽度为 `floor(E / ratio)`.
FP32 all-reduce 输入为 `rows * width * 4` 字节.
`E=257` 且一行时, 有三次 512 字节调用和五次 1028 字节调用.
Prefill 与 decode 都包含这些归约.

TEP4 增加 40 次 routed-output FP32 all-reduce, 每行输入 20480 字节.
默认模型有 40 次 dense, 40 次 routed 和八次 index-score 归约.

理想下界合并整批, 每个 target 调用对每层集合通信计一次.
它对整批权重只读一次.
它不计 `indexing.score_chunks` 的 256 MiB 分数工作区分块.
这些分块可能在实际执行中增加 all-reduce.

Eager index 路径使用 `end_pos // ratio`, 与此模型一致.
已捕获 graph 使用 `compressed_context_width()` 先把上下文向上取到 bucket, 再除 ratio.
257 个 token 时, ratio 2 的 graph 宽度可能为 256, ratio 1 的宽度可能为 512.
对应八次调用的字节数更大; 下界不计这些填充.

[Decode 请求循环](../../../python/oh_my_vllm/models/deepseek_v41/inference.py#L1058)<!-- source: outputs_by_id[row["request_id"]] = self._execute_row(row) --> 逐个执行请求.
其实际集合通信次数约为理想次数乘 batch 大小.
批量 decode 可以消除这部分执行成本; 下界不计此成本.

### DEP4 模型

DEP4 没有已接受的 executor, 也没有实测参考结果.
其路由矩阵对每个 target 调用建模 40 次 BF16 dispatch 和 40 次 FP32 combine.
`dep4-bf16-all-to-all-proxy-v1` 规则对两种 dtype 都按相同字节数查询 BF16 all-to-all 延迟表.
All-to-all 在字节数相同时传输相同数据量, 因此该规则借用延迟数据而不改变传输量.

微基准输入包含本地四分之一, 路由链路字节不包含该部分.
规则用 `ceil(wire_bytes * 4 / 3)` 把路由链路字节换成 profile 输入字节.
FP32 路由链路字节仍按每个值四字节计算.

这些结果保留 `communication_model=lower_bound`, `communication_is_modeled=True` 和 `communication_is_measured=False`.
结果还保存规则版本和 profile 摘要.
Wire 计算则保存 `communication_model=wire` 和 `legacy-wire-v1`.
TEP4 fallback 不能提供 DEP4 结果.

## Trace 契约

运行时 trace 为每个 rank 和 target 调用保存一条记录.
每条记录保存这些计数:

| 计数 | 含义 |
|---|---|
| `rank` | 0 到 3 的 rank 编号 |
| `call` | 唯一调用编号 |
| `phase` | `prefill` 或 `decode` |
| `requests` | 此 rank 的活动请求数 |
| `rows` | 此 rank 的计入 target 行数 |
| `global_rows` | 四个 rank 的计入 target 行数 |
| `logit_rows` | 此 rank 的计入 logit 行数 |
| `context_end` | 此 rank 的最后一个计入位置 |
| `accepted` | 此 rank 的已接受 draft token 数 |
| `proposed` | 此 rank 的已提议 draft token 数 |
| `sparse_pairs` | 此 rank 的计入注意力对数 |
| `index_pairs` | 此 rank 的计入压缩位置数 |
| `engram_unique_rows` | 此 rank 的已选 Engram 行数 |
| `touched_experts` | 此 rank 读取的本地专家权重数 |
| `layer_ids` | 完整 backbone 层集合 |
| `route_matrix` | 4 乘 4 DEP4 每层路由计数 |

`sparse_pairs` 和 `index_pairs` 保存此 rank 的计入计数.
TP4 和 TEP4 把总对数除以四.
DEP4 按拥有的请求计数.

Engram 表使用行分片.
一个 rank 可以取得超过四分之一的唯一行.
它的上限是此 rank 的因果查找计数.

每个 TP4/TEP4 rank 持有 64 个注意力头中的 16 个, 以及 32 个 index 头中的 8 个.
Index-score 计算计数遵循头分片.
计入的计算对数占四分之一; 集合通信缓冲区是完整的分数和.
一个 TEP4 rank 可以收到超过四分之一的 routed 分配.
它的专家上限是其本地专家集合的因果最大值.

trace 身份保存源码提交和 native 模块摘要.
它保存 checkpoint config 与 index 摘要和 tokenizer.
它保存四个 GPU UUID, strategy, effective strategy, owner plan 和采样设置.
它保存用例 id 和速率配置摘要.
它保存已接受与已提议计数.

检查会拒绝缺少 rank 记录或调用重复.
它拒绝层集合不完整或计数超过因果上限.
它拒绝缺少 DEP4 路由矩阵或身份变化.
它拒绝 trace 计数与绑定计数不一致.

## 理想计划

`reference_case` 为一个已批准 HTTP 用例生成计划.
它计入全部五个 draft, 因此计划显示最好接受情况.
该计划是参考量级, 不是 efficiency 结果.
基准 CSV 用 `reference_acceptance=ideal` 标记此计划.
策略回退使用 `reference_acceptance=unavailable`.
它记录 `ideal_reference_seconds` 和 `ideal_efficiency_percent`.

CSV 行还保存 `accounting_version`, `communication_model`, `profile_digest` 和 `communication_rule_version`.
DEP4 行包含 `dep4_communication_modeled` 和 `dep4_communication_measured`.

## 8192 输出用例的参考值

这些值使用理想接受.
每个 DSpark5 调用接受全部五个 draft.
普通行每次调用使用一个 target 行.
这些值是参考量级, 不是 efficiency 结果.

第一张表只使用 `v1/wire`.

| 用例 | Strategy | v1/wire 秒数 |
|---|---|---:|
| `dspark5:tp4:4:400000:8192` | TP4 | 21.230 |
| `dspark5:tp4:4:800000:8192` | TP4 | 38.833 |
| `dspark5:tp4:16:400000:8192` | TP4 | 76.849 |
| `dspark5:tp4:16:800000:8192` | TP4 | 147.261 |
| `dspark5:tp4:64:400000:8192` | TP4 | 278.512 |
| `dspark5:tp4:64:800000:8192` | TP4 | 560.159 |
| `dspark5:tep4:4:400000:8192` | TEP4 | 20.831 |
| `dspark5:tep4:4:800000:8192` | TEP4 | 38.051 |
| `dspark5:tep4:16:400000:8192` | TEP4 | 75.317 |
| `dspark5:tep4:16:800000:8192` | TEP4 | 144.200 |
| `dspark5:tep4:64:400000:8192` | TEP4 | 272.149 |
| `dspark5:tep4:64:800000:8192` | TEP4 | 547.680 |
| `dspark5:dep4:4:400000:8192` | DEP4 | 14.427 |
| `dspark5:dep4:4:800000:8192` | DEP4 | 24.240 |
| `dspark5:dep4:16:400000:8192` | DEP4 | 45.904 |
| `dspark5:dep4:16:800000:8192` | DEP4 | 85.157 |
| `dspark5:dep4:64:400000:8192` | DEP4 | 152.000 |
| `dspark5:dep4:64:800000:8192` | DEP4 | 309.011 |
| `ordinary:4:400000:8192` | TP4 | 25.891 |

第二张表使用 `v2/lower_bound`.
DEP4 行使用所述建模规则, 没有实测 executor.

| 用例 | v2/lower_bound 秒数 |
|---|---:|
| `dspark5:tp4:4:400000:8192` | 53.039 |
| `dspark5:tp4:4:800000:8192` | 154.376 |
| `dspark5:tp4:16:400000:8192` | 194.223 |
| `dspark5:tp4:16:800000:8192` | 599.706 |
| `dspark5:tp4:64:400000:8192` | 737.554 |
| `dspark5:tp4:64:800000:8192` | 2359.488 |
| `dspark5:tep4:4:400000:8192` | 51.824 |
| `dspark5:tep4:4:800000:8192` | 152.778 |
| `dspark5:tep4:16:400000:8192` | 191.558 |
| `dspark5:tep4:16:800000:8192` | 595.512 |
| `dspark5:tep4:64:400000:8192` | 728.169 |
| `dspark5:tep4:64:800000:8192` | 2343.986 |
| `dspark5:dep4:4:400000:8192` | 16.765 |
| `dspark5:dep4:4:800000:8192` | 26.578 |
| `dspark5:dep4:16:400000:8192` | 48.415 |
| `dspark5:dep4:16:800000:8192` | 87.668 |
| `dspark5:dep4:64:400000:8192` | 153.989 |
| `dspark5:dep4:64:800000:8192` | 311.000 |
| `ordinary:4:400000:8192` | 74.490 |

## 缺少的运行时计数

DeepSeek 运行时 trace 没有这些计量计数.
以下计数缺少:

- rank 1 到 3 的执行 trace 记录
- 每个调用的用例 id
- 已接受与已提议 draft 计数
- 每个 rank 的 routed 分配计数和本地被访问专家计数
- 每个 rank 的已选注意力对数
- 每个 rank 的已打分 indexer 对数
- 每个 rank 的已选 Engram 行数
- DEP4 的 4 乘 4 dispatch 与 combine 路由矩阵
- trace 中的 native 模块摘要和四个 GPU UUID

当运行时提供这些计数后, 使用 `reference_trace`.
在该 trace 存在前, 不要报告实测 efficiency.

使用 [测试范围](../../testing.md) 和 [报告规则](../../testing.md#reporting).
