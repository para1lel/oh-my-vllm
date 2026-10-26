# 测试范围

添加测试前遵循[贡献流程](../CONTRIBUTING.zh.md).
为当前契约和失败路径保留少量行为检查.
优先使用已有用例. 删除过时迁移检查和重复断言.
代理为维护者执行适用测试和 [hooks](development.zh.md#git-hooks).

## 共享检查

Rust 检查服务行为, 调度, 逻辑状态和 worker 生命周期.
HTTP 服务测试通过进程边界使用显式 Python 占位实现.
这些测试的耗时不是推理性能.
CPU 检查涵盖模型选择, readiness, 约束, 证据身份, 计算口径和 manifest.

[执行检查](../tests/test_execution.py)覆盖自适应选择, 准备复用, 绑定变化和 Graph 资源所有权.
它们检查录制失败, replay stream, 中断和关闭顺序.
[Worker 检查](../tests/test_offload_worker.py)覆盖 offload worker 和执行边界.
传输检查覆盖取消和存储生命周期.
原生内存或同步行为改变时使用 Compute Sanitizer.

仅文档变更使用文档和空白 hooks.
这类变更不触发 GPU 测量, 也不改变已记录的性能值.
代码变更使用相关 CPU, 数值, 状态, 内存和性能检查.
没有新问题时, 不重复无关且已经完成的检查.

## 模型验收

各模型维护自己的数学参考, fixtures, 限值和 workload 定义.
候选实现验收期间, 冻结的预期数组和数值限值保持不变.
通常验收加载 fixtures, 不生成预期输出.
将完整端点向量与独立参考计算比较.
不将推理输出作为自身参考.

| 适配器 | 数值和状态范围 | 性能范围 |
|---|---|---|
| Qwen | [全模型端点](models/qwen38/evaluation.zh.md), [算子用例](models/qwen38/operators.zh.md), 适用的 MTP, 媒体和恢复用例 | [24 组 HTTP 用例](models/qwen38/performance.zh.md), [AgentX](models/qwen38/agentx.zh.md) |

共享运行时变更执行适用的适配器回归.
Qwen 运行时交付包含其 24 组 HTTP 用例.
模型局部变更不触发另一模型的完整矩阵, 除非共享路径改变.
通用描述符或状态检查不能证明数值正确性, 容量或性能.

已有的[草稿链检查](../development/operators/check_draft_chain.py)将 Graph 采样, RNG offset 和历史与顺序采样比较.
GPU 命令通过 wrapper 执行:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.check_draft_chain
```

## 记录

保留当前说明和各已批准 workload 的最新完整结果.
使用各模型的结果文档及其完整中文翻译.
新矩阵完成后才替换表格.
不为每次变更新建报告, 不在 README 中放结果表.
从 README 链接已完成报告. 原始记录和无关测试结果保留在仓库外.

HTTP 行标明模式, 策略, batch, 输入/输出长度和实测请求数.
报告完整 HTTP 秒数, 理论参考秒数, efficiency 百分比和草稿接受情况.
Efficiency 是理论参考秒数除以完整 HTTP 秒数, 以百分比表示.
这些表格不包含 Input/s, Output/s, TTFT 或旧 prefill 对比.
各表格同时定义计算方式, 接受计数和单位.
失败和各 rank 内存单独报告.

结果保留源码, 可执行文件, 原生模块, checkpoint, tokenizer, 拓扑, 配置, 采样和命令身份.
使用可移植标识符或产物摘要, 不使用主机路径.
各模型必须有经过审查的参考计算, 计入其原生格式, 通信, 重复计算和主机传输.
未经此审查, 不复用另一模型的 efficiency 计算.
理论资源估算不证明物理最小延迟.

AgentX 为各已批准并行度和策略保留一个章节.
包含输出速率分布, 前缀复用, 排队, 请求状态计数, profiling 和 drain 时长, 数据集身份及观测 HTTP 并行度.
不将未完成请求计为成功样本.
原始 trace 和 profiler 记录保留在仓库外. 已批准的 CSV 和图放在模型结果旁.

遵循[文档规则](AGENTS.zh.md)和[交付流程](../CONTRIBUTING.zh.md#交付).
检查后停止自有 worker 和临时服务.
