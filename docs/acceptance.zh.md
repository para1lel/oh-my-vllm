# 验收证据索引

本索引标明测量, 源码范围和限制.
跟踪的 [可移植证据](../bench/evidence/README.zh.md) 是用于查阅和离线分析的历史派生数据.
每份摘要记录原始 SHA-256 和删除字段.
完整原始记录保留在被忽略的本地存储或原始 Git 历史中.
正式比较需要原始记录及匹配条件.
新服务器必须测量自己的匹配基线.

## 最近一次完整性能验收

源码: `619c9d98809c00081c5afb549987cde1cbd71690`, 2026-09-29 采集时为干净源码.
基线源码: `e9f169d16b9408bb9ae44f75072b91a5521d733c`.
原始基线 SHA-256: `fa3729f1a2ce160235b45df75542774d628dac7af963f01d673353fb419df8fd`.
Release binary SHA-256: `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
加载 CUDA module SHA-256: `24bec3a7208ac09629d0a594940266dec2dfbb272c87e02e27c6a8196cd84318`.

[算子摘要](../bench/evidence/2026-09-29-ir-operators.json) 包含 147 个通过输出校验和冻结 TileLang 比较的用例.
每个用例 3 轮, 每轮 20 个交替配对, 每个样本 100 次 graph repetition.
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
[决策记录](README.zh.md#决策记录) 解释有用的替代关系.
本轮检查和开放工作见 [当前工作](handoff.zh.md).
