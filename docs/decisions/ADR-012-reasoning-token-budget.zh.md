# ADR-012: 思考 token 预算

状态: 生效. 日期: 2026-10-12.

## 问题

Reasoning effort 控制 Qwen template, 不设置数值 token 上限.
模型可能在给出答案前用思考耗尽总输出预算.
当前安装的 OMP OpenAI adapter 不发送思考 token 预算.

## 决策

两种 HTTP API 使用顶层 `thinking_token_budget`, 与 vLLM 同名.
`--thinking-token-budget` 设置服务默认值. 请求的显式值优先.
默认值为 `-1`, 只保留总输出限制. 非负值限制生成的思考内容.
Effort 保持独立. 不为 `xhigh` 虚构数值预算.

达到预算后, target 只允许 token `</think>`. 生成此 token 后继续答案或工具调用.
总输出与上下文限制继续生效. 结束 token 占用总输出预算.
原有 reasoning usage 包含此 token, 可能为 `budget + 1`.
Off 不产生思考块. 启用思考时, 零预算强制生成结束 token.
此契约针对已测试的 Qwen tokenizer, 其结束标记为独立的单个 token.

Python 根据已提交计数, 为每个 target verification row 及其前置草稿前缀预览预算.
Rollback 后, mask 预览不改变计数, 历史, grammar 状态或 KV 状态.
只有保留的输出 token 改变已提交计数.
到达边界前, 思考输出也使用 [ADR-016](ADR-016-canonical-serving-control-tokens.zh.md) 中统一控制 token 的 grammar.
到达边界时, greedy 和随机 target 采样均使用 mask, 包括 bonus row.

DSpark 提议可以使用不同分布. Target verification 使用受约束的分布.
错误提议 token 的 target 概率为零, 因此被拒绝. 拒绝修正提供强制结束 token.
浮点减法得到零 residual mass 时, 单点 target 使用确定性的回退.
提议概率校验继续生效.

## 来源比较

[vLLM 思考指南](https://docs.vllm.ai/en/stable/features/reasoning_outputs/#thinking-budget-control) 说明了逐请求预算与强制结束思考.
它默认不单独限制思考, 没有推荐适用于所有工作负载的固定数值预算.
[SGLang 思考 grammar](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/constrained/reasoner_grammar_backend.py) 通过 token mask 应用思考限制.
我们根据这些来源制定契约. 项目代码和依赖不使用这些引擎.

## OMP 适配与限制

当前安装的 OMP 为 `v18.4.3`.
已观察到的 Chat 与 Responses 请求省略 `thinking_token_budget`, 同时保留原有 `max` effort 到 `xhigh` 的映射.
CPU 请求测试显示, model `compat.extraBody` 会为 Chat 添加此字段, 但 Responses 省略它.
用服务默认值统一覆盖两种 API 与自动压缩, 无需分别修改客户端.
在原始验收记录中保留运行时预算配置.

已安装的 Responses adapter 在 `300000` 毫秒内没有收到认可的进展事件时停止.
SSE 注释和 `response.in_progress` 不重置此计时器.
Reasoning 预算不能减少编译或下载时间.

Cubin 包与 FlashInfer 的版本相同. 它在请求前安装带版本的产物.
默认情况下, FlashInfer 使用每个产物前进行校验.
冷启动验证仍使用原客户端及其超时.

比较环境将此包加入运行依赖 lock.
冻结 TileLang manifest 绑定该 lock. DSpark manifest 绑定新的 TileLang manifest 哈希.
参考内核哈希, case ID, 数值容差及 TileFoundry commit 保持不变.

此预算不检测思考结束后的重复, 不能保证答案质量或消除 worker 故障.
保留这些失败, 修改后重新测试受影响工作负载.

## 验证

CPU 用例检查零预算, 关闭思考, 不限思考, 总长度, 类型校验和混合请求.
它们检查草稿前缀, rollback, bonus row, grammar 切换, 以及使用不同提议概率的 DSpark 拒绝行为.
目标模型检查须覆盖普通, MTP4, DSpark, 两种 API, stream, JSON, 工具和服务默认值.
一小时持久会话工作负载须保留最高 effort, 并记录公共生产服务预算.
使用生产默认值, 显式检查不同预算的测试除外.
