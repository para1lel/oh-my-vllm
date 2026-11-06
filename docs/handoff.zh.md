# 工作状态

使用 `nightly` 进行开发.
工作前读取被忽略的 `LOCAL.md` 中的接续索引.
其中提供主机设置, 外部产物位置, 主控规则和会话结束状态.
资源和 Git 交付状态以 `LOCAL.md` 索引中的最新外部记录为准.

## Harness 接续

仓库检查使用[开发循环](development.zh.md#开发循环).
代码限制为每个文件 500 个物理行, 每个函数 80 个物理行.
这些限制也适用于测试, 检查器和技能 helper.
[验证技能](../.claude/skills/verify-oh-my-vllm/SKILL.zh.md)驱动 Rust 应用, 并保存外部证据.
其中的 CPU worker fixture 检查服务和传输. 它不证明 GPU 推理, 数值正确性或性能.

Harness 工作将 Rust, Python, CUDA 和工具代码分为较小的模块.
新的验收使用当前源码和二进制身份.
前一份交接中的算子身份仅适用于其记录的源码和二进制字节.
这些身份不代表拆分后的源码, 也不证明其通过验收.
源码冻结后, 检查已移动的源码链接及原生构建身份.

## 交付目标

目标是通过完整 Rust HTTP 服务和 AgentX 运行 DeepSeek-V4.1.
批准的 HTTP 矩阵有 38 个用例.
其中 19 个长输出用例的效率必须各自达到 `deepseek-http-roofline-v6` 的至少 33%.
批准的完整 HTTP 矩阵, 三组 AgentX 工作负载和 33% 目标尚未完成.
旧 TP4 AgentX 记录不能证明全部会话树已完成.
源码变化后, 旧验收记录不能证明变更后的源码通过验收.

主控制订计划, 发出指令并完成验收检查和 Git 交付.
所有实现工作使用 `opus` 和 `max`.
本机索引提供代理更换流程.

## 当前实现

本次交付不要改动生产源码.
不要扩大试验范围.

生产源码包含 MTP proposal Graph, M16 内核和 M32 预取改动.
其中还包含 HC dot 和 router prefill 改动.
[执行指南](models/deepseek_v41/README.zh.md#mtp-proposal-graph)给出 MTP Graph 的条件和默认设置.
[量化指南](models/deepseek_v41/quantization.zh.md)给出内核参数和数值契约.
验收记录仅适用于记录中的源码和二进制身份.

`weight-store-v1` 和 grouped TMEM 候选已淘汰.
将 rawdot 调度用于更多小批量场景的候选已淘汰.
BF16 Compressor Graph 候选未整合到生产源码.
尚无获批 HTTP 测量表明该候选提高性能.
不要将恢复这些候选作为默认下一步.

## 下一步

1. 读取 `LOCAL.md` 中主控最新状态和组合源码身份.
   按该记录核对生产源码和二进制身份.
   为新工作分配 GPU 前, 确保资源已释放.

2. 读取 `LOCAL.md` 中组合源码身份的验收范围.
   仅完成这些记录尚未覆盖的检查.
   适用的数值, 状态, Graph, batch, request owner 和容量检查见模型指南.
   用例子集或算子对照不能代替独立完成的全模型检查.
   正式 HTTP 和 AgentX 测量前, 完成适用的验收检查.

3. 不要修改数值顺序, checkpoint weights 和 scales, 错误状态, fixture 或验收阈值.
   修改容量前, 先测量 cache page, state endpoint, projection storage 和 temporary storage.
   如果源码再次变化, 对新源码身份完成适用检查.

4. 完成 [HTTP 矩阵](models/deepseek_v41/performance.zh.md)和 [AgentX 工作负载](models/deepseek_v41/agentx.zh.md).
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

主控先提交并推送 `nightly`, 然后 squash 到 `main`.
主控在 `main` 上提交 squash 结果并推送 `main`.
主控再将 `nightly` 设为与 `main` 相同的修订.
主控再次推送 `nightly`.
仅推送 `main` 和 `nightly`.

停止你的 worker 和临时服务.
按 `LOCAL.md` 索引中的最新外部主控记录核对会话结束和 GPU 租约释放状态.
