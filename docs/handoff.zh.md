# 工作状态

使用 `nightly` 进行开发.
工作前读取被忽略的 `LOCAL.md` 中的接续索引.
其中提供主机设置, 外部产物位置, 主控规则和会话结束状态.

## 交付目标

目标是通过完整 Rust HTTP 服务和 AgentX 运行 DeepSeek-V4.1.
批准的 HTTP 矩阵有 38 个用例.
其中 19 个长输出用例的效率必须各自达到 `deepseek-http-roofline-v6` 的至少 33%.
批准的完整矩阵和 33% 目标尚未完成.
算子测量和旧功能记录不能证明这份源码通过验收.

硬截止时间是 `Asia/Shanghai` 的 2026-11-06 14:00.
在当天 11:40 开始最终检查和交付准备.
主控制订计划, 发出指令并验收工作.
所有实现工作使用 `opus` 和 `max`.
本机索引提供代理更换流程和已授权的 Git 顺序.

## 当前实现

主控已按记录的源码和二进制身份签收以下算子候选:

- SHA256 前缀为 `52b2ac1a` 的 `weighted_heads` 尾宽路径.
- 窄输出 dense FP8 projection, `N=512`, `K=5120`.
- 使用 `128x128` tile 的最初 grouped FP4 赢家.
- Dense FP8 `Compact` projection, `N=2304`, `K=5120`.
- DeepSeek Indexer 中完整的 DeepSelect BF16 `count512` top-k 实现.
- Sparse FP4 成对反量化, 含 Graph 测试对两个全局标志的修正.

另一名代理已审查 sparse 测试修正. 主控已签收该修正.
该修正未改变生产源码和二进制身份.
尚无代理基于冻结参考独立完成其中任何算子改动的全模型验收.

Grouped TMEM 候选已淘汰.
仓库已恢复最初 grouped 赢家.
后续 `weight-store-v1` 候选修改 bit3/bit4 线程置换.
它仅保存在外部, 已完成 CPU 构建, 未在 GPU 上运行, 也未由另一名代理验收.
本机接续索引提供其负责人交接文档, patch, 二进制身份和结束记录.

## 下一步

1. 读取 `LOCAL.md` 中主控最后的结束状态.
   按最终身份核对已签收的 sparse patch.
   按本机记录核对 grouped 赢家身份.
   为新工作分配 GPU 前, 确保资源已释放.

2. 下一次 grouped 短对照使用已保存的 `weight-store-v1` 入口.
   先比较独立计算的 literal 输出.
   再比较冻结 FP32 向量.

   然后按现有短测流程测量完整操作.
   不同 projection 二进制使用不同进程.
   不要为此次比较生成新的期望数组.

3. 不要重试已淘汰 TMEM, 旧 QK 融合或旧的自写 top-k CDF 实验.
   不要用增加采集或巡检工具代替实现工作.
   在长 HTTP 或 AgentX 测量前, 用短对照选出能提高性能的改动.

4. 不要修改数值顺序, checkpoint weights 和 scales, 错误状态, fixture 或验收阈值.
   修改容量前, 先测量 cache page, state endpoint, projection storage 和 temporary storage.
   适用的数值, 状态, Graph, batch, request owner 和容量检查见模型指南.

5. 提高性能的实现改动后, 运行完整[冻结参考流程](models/deepseek_v41/evaluation.zh.md).
   用例子集或算子对照不能代替独立完成的全模型检查.
   正式 HTTP 和 AgentX 测量前, 完成适用的验收检查.

6. 完成 [HTTP 矩阵](models/deepseek_v41/performance.zh.md)和 [AgentX 工作负载](models/deepseek_v41/agentx.zh.md).
   选择 `deepseek-http-roofline-v6`, 因为不另选计算器时 HTTP runner 使用旧计算器.
   对全部 19 个长输出行使用性能指南中的验收命令.
   不要用算子微基准代替正式 HTTP 行.

## 指南和交付

[v6 计算器](../development/serving/deepseek_accounting_roofline.py)提供批准的效率参考.
[计量指南](models/deepseek_v41/accounting.zh.md)定义必要通信, 原生格式和并行放置策略.
批准的结果文档使用[报告规则](testing.zh.md#记录).
仅在新矩阵完成后替换结果表.
将原始记录保存在仓库外部.

共享行为见[架构](architecture.zh.md), [运行时契约](requirements.zh.md)和[执行规则](execution.zh.md).
模型细节和验收范围见 [DeepSeek 指南](models/deepseek_v41/README.zh.md).
[Qwen 结果](models/qwen38/results.zh.md)提供其最后完成的 HTTP 和 AgentX 结果.
对适用的共享算子改动使用固定 [Qwen 算子用例](models/qwen38/operators.zh.md).

交付前遵守[贡献流程](../CONTRIBUTING.zh.md#交付).
请另一名代理审查文档和实现.
交付前修正审查错误.
停止你的 worker 和临时服务.
按主控最终记录确保会话已结束且 GPU 租约已释放.
