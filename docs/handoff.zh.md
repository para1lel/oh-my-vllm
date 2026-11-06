# 工作状态

使用 `nightly` 进行开发.
工作前读取被忽略的 `LOCAL.md` 中的接续索引.
索引提供外部合同, 主控状态, 资源所有权和 Git 交付记录.
当前截止时间为 2026-11-07 09:00 CST.
维护者已授权当前缓存和服务改动.

## 交付目标

| 模型 | HTTP 和 AgentX 范围 | 长输出要求 |
|---|---|---|
| Qwen | 完整 24 例 HTTP 矩阵及批准的 AgentX 工作负载 | 12 个长输出用例各自效率至少 40% |
| DeepSeek-V4.1 | 完整 38 例 HTTP 矩阵及三组批准的 AgentX 工作负载 | 19 个长输出用例各自按 `deepseek-http-roofline-v6` 达到至少 33% |

主控负责规划, 资源分配, 验收和 Git 整合.
实现使用 `opus` 和 `max`.
文件所有权和 GPU 授权以当前主控记录为准.

## 当前实现

项目删除 KV 主机存储及其协议和配置.
Engram CPU 和文件后备权重继续可用.
必要的 token, metadata 和 draft 传输继续可用.

[Qwen 缓存](models/qwen38/cache.zh.md)将 8192-token GDN checkpoint 与 128-token FP8 KV page 分开.
可用持久容量遵循 `P = 64 * C`, null, 私有和 scratch 存储单列并计入字节预算.
Radix 策略保留独立前缀证据, 并按准入代次决定成功会话尾的替换顺序.
服务使用[第 2 版执行协议](adapters.zh.md#执行协议)和模型专属收尾规则.
DeepSeek 保持驻留缓存, 已接受状态恢复及四 rank owner 规则.

全部使用中算子 shape 的公开对比通过前, 真实 checkpoint 权重或 GPU 执行的模型验收继续关闭. CPU 单元测试已获允许, 也允许使用生成或合成权重的小模型测试.
新 Qwen 格式, 物理容量, GPU 行为, 完整 HTTP 矩阵和 AgentX 验收均未验证.
DeepSeek 的完整 HTTP, AgentX 和效率目标尚未完成.
旧测量仅适用于其记录中的源码, 二进制和计量身份.

## 下一步

1. 从 `LOCAL.md` 索引读取当前外部实施合同和主控状态.
   检查已分配文件和当前源码身份.
   请其他代理完成审查. 修正审查发现的问题.

2. 真实 checkpoint 或 GPU 模型验收前, 完成全部使用中算子 shape 的公开对比门槛.
   CPU 单元测试已获允许, 包括使用生成或合成权重的小模型测试.
   GPU 所有权使用[执行规则](execution.zh.md).
   授权记录不能替代实时资源租约.

3. 容量验收前检查完整 Qwen 预算.
   计入缓存 slab, endpoint bundle, hidden 值, scale, workspace, metadata, MTP 备份, Graph 和分配器存储.
   获准执行后测量物理峰值.
   静态容量示例不证明 GPU 可用容量.

4. 对冻结的实现身份完成适用的数值, 状态, 取消, 写时复制, Graph 和全模型检查.
   保持 checkpoint 权重, scale, 冻结 fixture, 数值顺序, 错误状态和验收阈值不变.
   使用[测试范围](testing.zh.md)和模型指南.
   后续源码变化后重新核对已变更身份.

5. 完成批准的 [Qwen HTTP](models/qwen38/performance.zh.md) 和 [AgentX](models/qwen38/agentx.zh.md) 测量.
   完成批准的 [DeepSeek HTTP](models/deepseek_v41/performance.zh.md) 和 [AgentX](models/deepseek_v41/agentx.zh.md) 测量.
   使用 DeepSeek 性能指南中的显式 v6 选择和验收命令.
   分别报告请求完成和会话树完成情况.

## 证据和交付

局部的 `qwen-fp8-components-v1` 资源计算不是完整 HTTP 验收计算器.
[已记录的 Qwen 结果](models/qwen38/results.zh.md)保留批准的 v5 表格及其记录范围.
新完整矩阵完成前, 不替换结果表.
原始记录和检查保存在仓库外部.

使用[开发循环](development.zh.md#开发循环), [报告规则](testing.zh.md#记录)和[贡献流程](../CONTRIBUTING.zh.md#交付).
[验证技能](../.claude/skills/verify-oh-my-vllm/SKILL.zh.md)记录 Rust 应用行为.
其中的 CPU fixture 不证明 GPU 推理, 数值正确性或容量.
交付前请其他代理完成实现和文档审查.

主控提交并推送 `nightly`, 再完成 squash 并推送 `main`.
主控再将 `nightly` 设为与 `main` 相同的修订, 并推送 `nightly`.
仅推送 `main` 和 `nightly`. 保持 hooks 启用.
停止自有 worker 和临时服务.
按外部主控记录核对进程身份, IPC 清理和资源释放.
