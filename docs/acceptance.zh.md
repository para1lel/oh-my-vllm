# 验收证据索引

每项结果只适用于记录的源码, 配置和工作负载.
[要求](requirements.zh.md)定义验收标准, [测试](testing.zh.md)提供采集命令.
未改动的原始记录和独立审查提供验证依据, 便携摘要标明范围和哈希.

## 当前交付状态

选定最终运行时之后, 重跑数值, 回归, 容量, 服务和十五组工作负载验收.
工作负载矩阵包含 27 项延迟门槛, 两次预热, 五次测量和 10% 波动门槛.
前缀命中预填充保留延迟比值例外, [交接](handoff.zh.md)记录当前工作和已获授权的数值策略审查.

历史干净提交 `96ad82c` 使用原先的混合精度 reference, 通过全部 539 项正式数值检查, 未检测到干扰.
[报告](evidence/attention-merge/final-numerical.json)记录原始结果, 源码, 资源归属和独立审查.
它不能提供独立 FP64 数值验收.
此前 [注意力诊断记录](evidence/attention-merge/verification.json)保留诊断状态.

ordinary 模式 32768 输入/4096 输出批次一诊断观测到预填充/解码比值 4.305/2.181.
延迟门槛失败, 波动检查通过. 这些记录不能代替完整矩阵或最终源码服务验证.

## 源码范围记录

| 记录 | 范围 |
| --- | --- |
| [FP64 reference](evidence/fp64-reference/verification.json) | 独立公式, CPU 检查和小规模实际检查点 reference 产物, 不提供候选实现的数值或性能验收 |
| [GDN FP64 诊断](evidence/gdn-fp64/verification.json) | 六组大规模单算子输入, 保留输出失败记录, 以及四个已迁移的回归文件, 不提供生产验收 |
| [FP64 回归](evidence/fp64-regressions/verification.json) | 另外五个已迁移文件, 31 项 CPU 和 42 项 GPU 检查, 以及精确存储边界用例, 不提供完整测试集验收 |
| [FP64 融合投影](evidence/fp64-fused-projections/verification.json) | 另外两个已迁移文件, 28 项 CPU 和 54 项 GPU 检查, 保留的失败与精确特殊存储检查, 不提供完整测试集验收 |
| [3324a7f](evidence/refactor-3324a7f/verification.json) | 数值, 回归, 六项容量案例, 以及 MTP/DSpark 服务和恢复 |
| [子组扫描](evidence/refactor-s8-scan/verification.json) | 数值和重放诊断 |
| [f05a35d](evidence/refactor-f05a35d/verification.json) | 干净源码 539 项数值重跑及失败的整模型阶段门槛 |
| [窄 FP8](evidence/narrow-fp8/verification.json) | 绑定源码的数值和操作诊断 |
| [持久状态](evidence/persistent-state/verification.json) | 数值, 状态, 容量和整模型诊断 |
| [Split-K](evidence/split-k/verification.json) | 数值和操作诊断 |
| [原生调度](evidence/native-scheduling/verification.json) | 设备区间和被拒绝的调度记录 |
| [被拒绝的调度](evidence/rejected-schedules/verification.json) | 被拒绝候选的身份和未改动结果 |
| [精度记录](evidence/precision-lessons/verification.json) | 被拒绝的数值实验 |
| [仿射记录](evidence/affine-diagnostics/verification.json) | 矩阵链路诊断和失败的抵消案例 |
| [硬件记录](evidence/optimization-observations/verification.json) | 绑定源码的计数器, 输入和失败实验 |

重构前的交付记录保留在 Git 历史和外部原始存储中.
当前指南只保留契约, 达到规定的性能目标后再写入经过验证的优化方法.

## 历史服务长跑范围

## 源码 3324a7f

源码 `3324a7f` 每种模式有四个 OMP 会话, 运行超过一小时. 这些工作负载不同于要求的整本书翻译采集.
此前 MTP 采集器的 SSE 截止竞态审计失败, DSpark 采集器记录的结果保留 `passed=false`.
独立审查不改变采集器结果.

外部驱动器报告无法证明的被收养进程清理, 这些历史身份已无法重建.
记录中的服务, 工作器和沙箱身份均已退出, 临时推理监听器和 GPU 已释放.

工作器 RSS 曾增加, 这些记录不能证明内存平台, 完整归因或当前源码稳定性.

## 采集身份和限制

绑定源码, 可执行文件, 运行时版本, 已加载模块, 头文件, 编译器, 检查点, 硬件, 容量和进程出生身份.
原始日志, 完整失败集合和中断尝试保存在外部存储中, 保留所有样本.
编译活动, 捕获, 缓存驱逐, 干扰或源码改动会使测量重复失效.
剖析与正式计时分开, 便携摘要不能代替完整原始记录.

服务采集须报告持续可用性, 资源采样, 配置的截止时间, 中止, 请求排空尾段, 以及自有资源清理.
工作负载进展与性能和服务验收分开记录.
