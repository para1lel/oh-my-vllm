# ADR-016: 服务控制 token 的统一身份

状态: 实现处于验收阶段. 日期: 2026-10-13.

## 问题

`Generation.consume` 收到 tokenizer 的单个 `</think>` ID 后, Python 才结束思考状态.
Rust 在解码文本中首次出现 `</think>` 字符串时切换通道.
原 grammar 也使用字符串分隔符.
普通 ID 可以在不使用专用 ID 的情况下拼出相同字符串.
这种拼法可能结束 grammar 和 parser 的思考状态, 但 Python 继续累计思考 token.
之后到达有限预算时, grammar 已不允许结束 token, 因而可能报错.

原自动工具 grammar 到 `<tool_call>\n<function=` 才开始施加约束.
Rust 在 `<tool_call>` 就识别工具调用, 并接受更多空白形式.
因此, 错误的起始格式可能绕过 grammar 的 schema 和函数选择约束.
这些是构造出的协议失败. 它们不能证明已记录的正文循环的原因.

## 决策

对每个启用思考的请求验证单个, 独立的思考结束 ID, 包括无限预算请求.
该 ID 必须不同于 EOS.
`TokenFormat` 和 `Generation` 使用同一个 ID.
思考内容排除完整的 `<think>` 和 `</think>` 字符串.
普通 ID 的其他拼法不能只结束 grammar 和 parser 的思考状态.

纯文本思考输出也使用相同的思考外层结构, 后面接不受结构约束的文本.
保留结束 token 后原有的空白形式.
关闭思考且禁用工具的纯文本输出保留无掩码路径.
JSON 和启用工具的输出保留原有的正文分隔符和约束.

自动和必需工具调用从 `<tool_call>` 就进入约束.
使用 builtin 的函数选择和并行调用策略.
生成的规范调用从第一个起始标记起就必须满足声明的约束.
显式 `strict=false` 保留原有的不约束参数的 grammar.
Rust 的 XML 解析仍作用于这些参数.

Target mask 作用于普通, MTP4 和 DSpark 的验证行.
预览后回滚. 只有保留的 token 改变持久状态.
思考外层结构在真正的结束 token 前拒绝 EOS.
显式文本 stop 和总长度限制仍可在该 token 前结束请求.

## 重复调查

[诊断摘要](../../bench/evidence/2026-10-13-constrained-decoding-investigation.json) 区分已观察的失败, 构造的回归和 GPU 条件概率测量.
两次已记录的循环都发生在模型讨论思考结束分隔符, parser 切换到正文之后.
`</thinking>` 和 `<next_thinking>` 是普通文本, 并非思考控制 token.
现有正文 grammar 允许这些字符串.

在部分重建前缀处, target 将最高分赋给另一个真正的结束 token, 但正文 grammar 禁止它.
施加 mask 后, 普通标签片段具有较高条件概率.
在已测的循环后段起点, presence penalty `0.5` 没有使重复的下一个 token 的概率降到 `1` 以下.
这些测量只覆盖选定位置的下一 token 分布, 不覆盖完整循环中的每个 token.

原始采样 ID, logits 和随机状态均未保存.
诊断对保存的输出文本再次分词, 并插入一个规范结束 token.
它使用新建的 eager target 执行, 分别采用 BF16 和 FP32 GDN 状态.
它没有复现原来的 MTP, 调度器, graph 或前缀缓存执行.
结果支持将引用文本中的结束分隔符视为重复输出的可能原因.
它们不能证明唯一原因.

## 引擎比较与限制

[XGrammar structural tags](https://github.com/mlc-ai/xgrammar/blob/main/docs/structural_tag/tool_calling_and_reasoning.md) 约束输出结构, 并支持 token 分隔符.
结构合法的文本仍可能无法完成任务. 重复句子仍可能出现.
[vLLM 重复检测](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/#vllm.sampling_params.RepetitionDetectionParams) 是独立, 可选的请求终止机制.
其默认关闭. 检测会终止请求, 但 Agent 任务仍可能未完成.

按用户要求, 公共 presence penalty 保持为 `0.5`.
保持 checkpoint 采样, 思考预算默认值 `-1`, 总量限制, 数值容差和 DSpark 权重.
不全局阻止普通伪标签. JSON 字符串和工具值可以包含标记文本.
协议修复不声称解决语义重复.
仍需长程模型验收和独立内存验收.

## 验证

协议修改前已添加并复现失败的 tokenizer 和 grammar 回归.
测试覆盖分隔符的其他 ID 拼法, 错误工具起始格式, schema, 指定函数, 并行限制和推测回滚.
它们也覆盖禁用工具, 标签字面量, 纯文本原有空白形式, 显式非 strict 工具, stop 字符串和总长度.
正式验收必须使用新提交的源码及公共生产默认值.
