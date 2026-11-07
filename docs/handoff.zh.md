# 工作状态

使用 `nightly` 进行开发.
工作前读取被忽略的 `LOCAL.md` 中的接续索引.
索引提供外部合同, 主控状态, 资源所有权和 Git 交付记录.
维护者已授权当前缓存和服务改动.
本次交付保留当前实现进度, 供新会话继续工作.

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

当前 [HTTP 响应改动](../crates/zmq-worker/src/serving/engine/output.rs)将响应与记录的 Rust 完成结果绑定.
[Qwen 验收工具](../development/serving/qwen_fp8_acceptance.py)检查 24 例 HTTP 矩阵.
[AgentX 运行器](../development/serving/agentx.py)在失败后写出报告.
[MTP 适配器](../development/operators/adapters/mtp.py)通过[缓存辅助模块](../development/operators/adapters/mtp_cache.py)增加缓存证据.

当前算子改动增加 [FA 向量加载](../python/oh_my_vllm/operators/fa_fp8/cuda/attention.cuh).
[生产 projection 分派](../python/oh_my_vllm/operators/projection/cuda/projection.cu)对 `G384/M24/N576/K5120` 选择分组四行计算.
尚未对这个生产 projection 二进制运行 GPU 数值, 状态和 Graph 检查.
尚未在 Eager 和 Graph 模式下对完整 FFN 派生路径的 value 和 residual 输出运行检查.

计时入口仅有针对连接部分的有限静态审查.
最新计时入口审查未重新计算依赖哈希.
此审查不表示计时通过验收.

模型验收前, 全部使用中算子 shape 的公开对比必须给出符合要求的结果.
在此之前, 真实 checkpoint 权重或 GPU 执行的模型验收继续关闭.
CPU 单元测试已获允许.
CPU 小模型可使用生成或合成权重.

新 Qwen 格式, 物理容量, GPU 行为, 完整 HTTP 矩阵和 AgentX 验收均未验证.
两个模型的完整 HTTP, AgentX 和逐用例效率目标均未完成.
已有证据仅适用于其记录中的源码, 二进制和计量身份.

## 下一步

1. 从 `LOCAL.md` 索引读取当前外部实施合同和主控状态.
   检查已分配文件和当前源码身份.
   请其他代理完成审查.
   修正审查发现的问题.

2. 固定计时前, 对生产 projection 模块做 GPU 数值, 状态和 Graph 检查.
   在 Eager 和 Graph 模式下完成完整 FFN 派生路径的 value 和 residual 输出检查.
   按外部证据核对最新计时入口的依赖哈希.
   GPU 所有权使用[执行规则](execution.zh.md).
   授权记录不能替代实时资源租约.

3. 仅在各算子自身的数值检查通过后执行该算子的固定计时.
   分组算子计时不依赖完整 FFN 检查.
   真实 checkpoint 或 GPU 模型验收前, 完成全部使用中算子 shape 的公开对比门槛.

4. 容量验收前检查完整 Qwen 预算.
   计入缓存 slab, endpoint bundle, hidden 值, scale, workspace, metadata, MTP 备份, Graph 和分配器存储.
   获准执行后测量物理峰值.
   静态容量示例不证明 GPU 可用容量.

5. 对冻结的实现身份完成适用的数值, 状态, 取消, 写时复制, Graph 和全模型检查.
   保持 checkpoint 权重, scale, 冻结 fixture, 数值顺序, 错误状态和验收阈值不变.
   使用[测试范围](testing.zh.md)和模型指南.
   后续源码变化后重新核对已变更身份.

6. 完成批准的 [Qwen HTTP](models/qwen38/performance.zh.md) 和 [AgentX](models/qwen38/agentx.zh.md) 测量.
   完成批准的 [DeepSeek HTTP](models/deepseek_v41/performance.zh.md) 和 [AgentX](models/deepseek_v41/agentx.zh.md) 测量.
   使用 DeepSeek 性能指南中的显式 v6 选择和验收命令.
   分别报告请求完成和会话树完成情况.

## 证据和交付

[Qwen HTTP 指南](models/qwen38/performance.zh.md#新布局-http-计费)使用 `qwen-fp8-http-reference-v1` 作为完整冷 HTTP 计算器的版本.
局部的 `resource_floor` 计算保持 `qwen-fp8-components-v1`.
CLI 默认使用 `http-reference-v5`, 在新布局下返回 `blocked`.
对 24 种 HTTP 配置使用指南中的显式版本选择.
此计算器不涵盖 AgentX 前缀或会话树计费.

[已记录的 Qwen 结果](models/qwen38/results.zh.md)保留批准的 v5 表格及其记录范围.
新完整矩阵完成前, 不替换结果表.
原始记录和检查保存在仓库外部.
被忽略的 `LOCAL.md` 提供最新证据索引.

使用[开发循环](development.zh.md#开发循环), [报告规则](testing.zh.md#记录)和[贡献流程](../CONTRIBUTING.zh.md#交付).
[验证技能](../.claude/skills/verify-oh-my-vllm/SKILL.zh.md)记录 Rust 应用行为.
其中的 CPU fixture 不证明 GPU 推理, 数值正确性或容量.
交付前请其他代理完成实现和文档审查.

主控提交并推送 `nightly`, 再完成 squash 并推送 `main`.
主控再将 `nightly` 设为与 `main` 相同的修订, 并推送 `nightly`.
仅推送 `main` 和 `nightly`.
保持 hooks 启用.
停止自有 worker 和临时服务.
按外部主控记录核对进程身份, IPC 清理和资源释放.
