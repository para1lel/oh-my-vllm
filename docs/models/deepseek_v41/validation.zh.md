# DeepSeek V4.1 验证计划

## 状态与范围

本文是[架构](README.zh.md)和[阶段计划](plan.zh.md)的已批准验收范围.
它不报告已完成测试.
纯文档变更执行适用的文档和空白 hooks.
它们不触发 GPU 测量或修改旧性能数值.
实现变更执行适用的 CPU, 数值, 状态, 内存及性能检查.

## HTTP 矩阵

输入和输出长度针对每个请求, 不是整个 batch.
所有配置使用四张 B200 GPU, 总上下文容量为 1048576.

| 模式 | 并行策略 | Batch 大小 | 输入 tokens | 输出 tokens | 组数 |
|---|---|---|---|---|---:|
| DSpark, 最多五个草稿 | TP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 12 |
| DSpark, 最多五个草稿 | TEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 12 |
| DSpark, 最多五个草稿 | DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 12 |
| 普通解码 | TP4 | 4 | 400000 | 64, 8192 | 2 |

已批准总数为 38 组 HTTP 用例.
每种并行策略执行全部 12 组 DSpark 用例.
普通基线使用 TP4.
不得在批准的两组之外增加普通解码性能用例.
小型正确性检查与性能用例分开.

使用未缓存前缀, 并在模型专属提示编码后核对精确 token 数量.
沿用当前 tools 工作负载语义, 配合 DeepSeek codec 和带版本的采样 profile.
测量前冻结 temperature, top-k, top-p, penalties, seeds, reasoning mode 和 tool schema.
普通解码与 DSpark 对比采用相同请求分布.
保持采样和 grammar 启用. 不替换为无约束贪心生成.
计时前加载权重.
计时前完成初始化.
完整计时包含 HTTP 提交, 准备, 排队, prefill, decode, 采样, 通信和响应解析.
单独报告初始化和 timeout 设置.
检查配置的请求上限允许指定 batch 准入.
记录总体与逐请求时间, 不只记录首 token 时间.

## AgentX 矩阵

仅使用最多五个草稿的 DSpark.

| 会话树并行度 | 策略 | 组数 |
|---:|---|---:|
| 4 | TP4 | 1 |
| 16 | TEP4 | 1 |
| 64 | DEP4 | 1 |

共三组 AgentX 用例.
AgentX 并行度统计会话树, 不一定等于同时执行的 HTTP 请求数.
记录配置及观测到的 HTTP 并行度, 服务上限为 64 个活跃请求.
树分支提交更多请求时记录排队.
不得为避免排队而降低要求的会话树并行度.
先使用现有固定版本的 harness 和数据集.
记录数据集身份, tokenizer, seed, trajectory 规则, warmup 和 drain 设置.
每组至少 profiling 900 秒, 使用现有 120 秒 drain 默认值.
分别记录 overflow, failed, cancelled, incomplete 和 successful 请求数量.
使用完整 response IDs 关联 streaming usage 和服务 traces.

当前数据集为 256k traces. 它不能证明 1M 容量.
为该目标使用独立容量检查.

## 功能清单

实现[共享服务契约](../../requirements.zh.md#服务功能)中的全部适用功能.
使用 DeepSeek 编解码器, 状态恢复和 DSpark 规则.
定义命名 effort 级别到 DeepSeek 数值 effort 的映射.
保留公共 API 语义和限制.

## 数值与状态检查

保持冻结的 Qwen fixtures 和门槛.
DeepSeek 独立参考基于原始量化权重值, 不使用推理输出作为参考.
验收前冻结 checkpoint, 参考实现, fixtures, 完整词表向量和精度门槛.
短全模型用例使用逐层流式 FP64 参考计算.
恢复, 媒体和选定长上下文用例使用已保存的完整端点向量.
常规验收加载 fixtures, 不生成新期望输出.
分别检查目标 prefill, 单步 decode, 草稿输出, 验证和恢复后的续算.
对同一语义请求, 按批准的数值门槛比较 TP4, TEP4 和 DEP4.

覆盖压缩块边界, 共享 KV 所有权, 部分前缀分支和每个草稿拒绝位置.
检查取消后才完成的传输及 rank 通信期间的故障.
对修改的原生内存和同步路径使用 Compute Sanitizer.
尽量复用现有服务, 缓存, scheduler, constraints, execution 和 transfer tests.
新增测试前说明其检测的故障, 以及现有检查为何不能发现它.

## 容量检查

总上下文上限为 1048576, 输入与输出合计.
在上限处检查 1048512-token 提示与 64 个输出 tokens.
检查超过上限时明确拒绝, 不静默截断.
阶段 5 在较小用例通过后, 每种策略使用普通解码检查 64 个这样的活跃请求.
阶段 6 每种策略使用 DSpark 重复该容量检查.

这是容量检查, 不是额外的 HTTP 吞吐矩阵.
在仓库外记录逐 rank 分配和峰值使用, pinned host bytes, 排队及实际重叠的请求活动.
有界 chunks 共享执行时, 全部 64 个请求保持已准入.
不得以 64 个排队请求且仅四个准入请求作为容量验收.
预算包含 graph pools, expert buffers, 未完成 compressor 状态, rollback 存储和 host transfer leases.

## 结果与交付

遵循共享的[记录策略](../../testing.zh.md#记录).
首次完成已批准的 HTTP 或 AgentX 测量后, 创建本目录的 `results.md` 和 `results.zh.md`.
记录 38 组 HTTP 和三组 AgentX 的最新完整结果.

DeepSeek 的参考计算必须计入原生格式, 通信, 重复计算和主机传输.
不复用 Qwen 的 `http-reference-v4` 计算.
在性能验收前审查计算方式及其版本.
