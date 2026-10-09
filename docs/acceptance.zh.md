# 验收证据索引

## DSpark 成对测量与开放验收

可选 DSpark 模式保持 checkpoint 权重固定, 并复用目标 embedding 及词表 head.
累积 confidence 阈值为 `0.2`. 设为 `0.0` 可关闭置信度截断.
输出, 上下文和 grammar 限制仍然适用.

[阈值选择记录](../bench/evidence/2026-10-08-dspark-confidence-selection.json) 的输入与正式合成负载不同.
包含两个自然文本及工具历史请求, 每请求输入 32768, 输出 512.
每个设置完整预热两次, 正式测量两次.
阈值 `0.0`, `0.05`, `0.1` 和 `0.2` 的 TPS 中位数为 177.287, 183.097, 184.198 和 184.323.
对应接受率为 28.487%, 44.957%, 53.406% 和 62.500%. 这些中位数适用于该小样本采集.

### 当前运行时代码的成对比较

[成对记录](../bench/evidence/2026-10-09-dspark-mtp4-paired.json) 使用运行时代码 `e83674c`, 在一张 B200 上执行.
输入 32768, 输出 4096, batch 为 1, 2 和 4, 使用 greedy 采样和冷前缀.
两种模式复用同一进程, 目标权重和物理目标缓存.
共三轮, 每轮每模式完整预热两次.
每轮有五对测量. 模式顺序逐对交替.

三个 batch 均输出规定的 token 数, 重算抢占为零.
全部 90 个正式区间的捕获和编译次数为零.

下表每行使用每模式十五个正式 batch.
TPS 是这十五次测量的中位数.
TPS 包含注册, prefill, decode, 传输, 采样和清理. 计时在模型加载及预热之后开始.
TTFT 先取每个 batch 内各请求的最大值, 再取这些值的中位数.
接受率用接受草稿总数除以实际验证草稿总数.

| Batch | 模式 | TPS | TTFT (s) | 接受率 | 每步验证数 | 每步接受数 |
|---|---|---|---|---|---|---|
| 1 | MTP4 | 319.134 | 1.631171 | 90.647075% | 3.980877 | 3.608549 |
| 1 | DSpark | 150.959 | 1.598959 | 72.105644% | 1.313688 | 0.947243 |
| 2 | MTP4 | 493.461 | 3.297708 | 80.736434% | 7.413793 | 5.985632 |
| 2 | DSpark | 221.112 | 3.223185 | 63.357157% | 1.854226 | 1.174785 |
| 4 | MTP4 | 723.972 | 6.565200 | 75.358676% | 14.667266 | 11.053058 |
| 4 | DSpark | 355.888 | 6.428824 | 56.250624% | 3.394175 | 1.909245 |

步骤数包含所有非空 prefill/decode 调度批次. 分子汇总每个 batch 的全部请求.
服务日志的步骤数按单个请求计数. 这两种统计范围分别保留.
自适应截断会改变实际验证草稿分母. 摘要保留提议数, 验证数, 接受数和步骤数.

Release binary 的 SHA-256 为 `4202db0d33251b705e1bedf65ec23bee979ec75377ee42ae7d9f30d0cc8c1072`.
已加载 CUDA module 的 SHA-256 为 `09729aa0e90b5b95108a0f0ed66114a0371af0559a8f6152ae545311c19645d4`.
已加载 DSpark 配置的 SHA-256 为 `dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd`.
已加载 DSpark 权重的 SHA-256 为 `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`.

独立审查通过 368 项检查, 核对原始日志, 区间, 源码身份, 已加载文件和已完成子任务的清理记录.
其范围为配对子任务. 后续边界任务因另一项任务使用 GPU 而失败.
摘要在不同字段保留原始父采集失败和已完成配对子任务成功.
Envelope 的 `original.sha256` 标识用于导出的输入. `raw_artifact` 字段标识完整原始记录.

三个成对 TPS 置信下界均小于零.
此比较不设新增 DSpark TPS, TTFT, 波动或置信下界门槛.
原有十二项普通/MTP4 负载门槛和完整算子门槛继续有效.
该 [尝试记录](../bench/evidence/2026-10-09-dspark-attempt-history.json) 保留失败采集, 诊断, 原始哈希和验证限制.

### 服务和容量范围

[此前的 DSpark 服务记录](../bench/evidence/2026-10-09-dspark-service-de1591b.json) 适用于运行时代码 `de1591b`.
通过 Chat 和 Responses 的六组客户端测试: 约束, 生命周期, oh-my-pi 工具循环和长上下文前缀复用.
摘要保留源码读取范围和模型回答的事实错误.
旧源码 `46f7529` 的完整 GPU 采集通过 285 项测试及九项 262144-token 边界用例.

当前运行时代码的算子, 边界, 十二项框架及服务验收等待 GPU 测试时段.
采集器拒绝 GPU 争用. 原始失败尝试和已完成子任务记录保留各自的源码身份.

当前图预算回归通过 548 项 CPU 测试和 70 项子测试, 排除 323 项 GPU 测试.
四项 GPU 图测试通过. 完整输出回退检查又通过 1 项测试.
该检查使用小型测试模型. 完整模型容量与服务检查将在当前运行时代码上再次执行.

## 历史验收

本索引标明测量, 源码范围和限制.
跟踪的 [可移植证据](../bench/evidence/README.zh.md) 是用于查阅和离线分析的历史派生数据.
每份摘要记录原始 SHA-256 和删除字段.
完整原始记录保留在被忽略的本地存储或原始 Git 历史中.
正式比较需要原始记录及匹配条件.
新服务器必须测量自己的匹配基线.

## 2026-09-29 的完整性能验收

源码: `619c9d98809c00081c5afb549987cde1cbd71690`, 2026-09-29 采集时为干净源码.
基线源码: `e9f169d16b9408bb9ae44f75072b91a5521d733c`.
原始基线 SHA-256: `fa3729f1a2ce160235b45df75542774d628dac7af963f01d673353fb419df8fd`.
Release binary SHA-256: `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
加载 CUDA module SHA-256: `24bec3a7208ac09629d0a594940266dec2dfbb272c87e02e27c6a8196cd84318`.

[算子摘要](../bench/evidence/2026-09-29-ir-operators.json) 包含 147 个通过输出校验和冻结 TileLang 比较的用例.
每个用例 3 轮, 每轮 20 对测量, 每个样本 100 次 graph repetition.
模式顺序逐对交替.
最小单侧 95% 正收益下界为 `0.0000036639670530955112 ms`.
原始外部算子采集器 SHA-256 为 `c6cdce051ea8fb50ec2e0adecde2ff0d132bd78456f829d14a3a3f49db3a5ab1`.

[框架摘要](../bench/evidence/2026-09-29-ir-framework.json) 包含 12 个通过行.
每个接受行包含 2 次完整预热和 5 次重复, 无抢占, 观察到的稳态审计通过.
每行使用单张 B200 GPU.
下表来自接受的完整集合:

| 行 | TPS/baseline | TTFT/baseline | TPS spread | TTFT spread |
|---|---:|---:|---:|---:|
| mtp-32768-1 | 100.67% | 94.22% | 0.12% | 0.88% |
| mtp-32768-2 | 99.53% | 95.74% | 0.20% | 0.70% |
| mtp-32768-4 | 99.27% | 95.26% | 0.24% | 0.25% |
| ordinary-131072-1 | 115.28% | 91.01% | 0.04% | 0.19% |
| ordinary-131072-2 | 113.78% | 91.65% | 0.11% | 0.37% |
| ordinary-131072-4 | 110.34% | 93.13% | 0.12% | 0.50% |
| ordinary-32768-1 | 118.15% | 92.98% | 0.10% | 0.95% |
| ordinary-32768-2 | 118.62% | 95.91% | 0.07% | 0.68% |
| ordinary-32768-4 | 115.82% | 94.21% | 0.20% | 0.40% |
| prefix-32768-1, fourth attempt | 118.86% | 83.64% | 0.16% | 5.02% |
| prefix-32768-2 | 125.63% | 84.55% | 0.28% | 3.64% |
| prefix-32768-4 | 117.53% | 76.92% | 0.17% | 4.72% |

此前 3 个完整 prefix-batch1 尝试因 TTFT spread 17.378%, 20.976%, 20.177% 被拒收.
第 4 个完整 2+5 集合通过.
另有 6 次尝试因无关 GPU 进程而中止.
摘要保留拒收尝试及 raw/log hash.
孤立 prefix TTFT spike 的原因尚不明确.
审计无法排除静默内存内重新编译, 或短于进程轮询间隔的干扰.

该源码还通过 544 个 GPU 测试, 70 个 subtest 和 303 个 CPU 测试, 包括 6 个最大上下文用例.
后续 IR 正确性修改通过 549 个 GPU 测试, 70 个 subtest 和 308 个 CPU 测试.
这些修改后没有重跑完整 147-case 和 12-row 性能采集.
这些历史结果不能证明当前 HEAD 的性能.

## 上下文与服务证据

| 证据 | 源码与结果 |
|---|---|
| [6 行上下文](../bench/evidence/2026-09-28-audit-evd07-context-boundary.json) | `bd8e21e5607387b081e1d4494bc7b8fdf79ac8d4`: ordinary/MTP4, batch1/2/4, input258048, output4096, 无 OOM / 抢占. |
| [长上下文 HTTP](../bench/evidence/2026-09-28-audit-evd07-long-context-http.json) | `8dfc97b544d18e21f0856a2b2b4098bea90c8be5`: Chat/Responses 各两次, strict JSON 和 MTP. |
| [真实 MTP 服务与 OMP](../bench/evidence/2026-09-22-ttft-agentic.json) | 两种 API 都完成 MTP 和实际读源码的工具结果往返. |

6 行上下文各输出 `batch_size * 4096` token.
MTP proposed/accepted 总量为 5010/2842, 9075/5917, 16219/12318.
Worker 最大 reserved memory 为 132441440256 bytes.
这些数字只适用于标明的边界源码.

长上下文 HTTP 每个 prompt 包含 131099 token.
每次返回 `{"n":123,"label":"verified"}`, 提出 20 个草稿.
重复请求命中 130928 个缓存 token.
专用 server, listener, IPC 路径和 worker 已清理.
同形状 262144 重复和低 headroom probe 的范围窄于跨形状 eviction/recapture.
参见 [审计](audit.zh.md).

## 关键前序证据

| 决策或里程碑 | 保留摘要 |
|---|---|
| 初始 adapter 与 MTP 设计 | [2026-09-19](../bench/evidence/2026-09-19-acceptance.json) |
| 独立运行时 | [2026-09-21](../bench/evidence/2026-09-21-independent-acceptance.json) |
| 冻结 TileLang | [2026-09-22](../bench/evidence/2026-09-22-tilelang-acceptance.json) |
| 刷新后的 12 行分母 | [EngineCore 基线](../bench/evidence/2026-09-22-refreshed-enginecore.json) |
| CUDA 实现 | [算子](../bench/evidence/2026-09-22-cuda-operators.json), [框架](../bench/evidence/2026-09-22-cuda-framework.json) |
| 审计修复源码 `e3c42e0` | [算子](../bench/evidence/2026-09-29-audit-final-operators.json), [框架](../bench/evidence/2026-09-29-audit-final-framework.json) |

早期测量使用不同源码 / 配置, 部分只有 3 次重复.
它们只证明对应历史范围, 不作为当前性能比较分母.
全部原跟踪记录都有字节不变的本地归档, 哈希位于 [originals.json](../bench/evidence/originals.json).
[决策记录](README.zh.md#决策记录) 保留实现替代记录.
本轮检查和开放工作见 [当前工作](handoff.zh.md).
