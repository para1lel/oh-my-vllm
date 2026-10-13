# 输出约束契约

## Grammar 与推测 token

JSON object, JSON Schema 和 strict tool grammar 在采样前生效.
约束答案可在思考之后生成.
一个请求不能同时使用 JSON 答案格式和启用工具.
Strict schema 需要 closed object, 全部 property 都 required.
不支持的 strict schema 明确报错.
JSON 答案及显式 strict 工具支持的 keyword 清单位于 `python/oh_my_vllm/worker/serving.py`.
这些 schema 不支持未知 string format, 或 pattern/format 与 length limit 的组合.

未显式设置 `strict=true` 时, 工具描述可以包含额外 schema keyword.
模型 prompt 保留完整原始描述和参数.
请求准备为现有 XML decoder 生成独立副本.
String 长度, pattern, format, default, examples 及其他省略 keyword 仍是模型应遵守的说明.
OMP 用原始工具定义校验生成参数, 并向模型返回失败信息.
已知 XML 类型, 必需字段, function choice 和并行调用规则保持现有行为.

[ADR-018](../decisions/ADR-018-client-owned-tool-descriptions.zh.md) 说明请求准备策略.

XGrammar 提供 Qwen XML tool grammar 和 JSON answer grammar.

每个思考请求使用经过验证的单个结束 token ID, 包括纯文本和无限思考预算.
拼出相同结束字符串的普通 ID 不能结束思考.
在该 token 前屏蔽 EOS. 显式文本 stop 和总长度仍可提前结束请求.
纯文本正文在思考结束后保留原有空白形式和标签字面量.

启用工具时, 约束与 Rust parser 一样从 `<tool_call>` 开始.
生成调用的后续部分必须满足函数选择, 并行调用和 schema 约束.
显式 `strict=false` 关闭工具参数的 schema 约束. XML 解析仍生效.
[控制 token 决策](../decisions/ADR-016-canonical-serving-control-tokens.zh.md) 给出合同和重复问题的限制.

每个 MTP 或 DSpark draft 和 bonus position 使用其 speculative prefix 对应的 mask.
模拟 grammar 推进后回滚; 只有保留输出推进持久状态.
被 mask 的草稿在第一个无效位置失败.
CPU 测试覆盖回滚, 拒绝和 reasoning-end 跨界.
真实 MTP4 证据覆盖两个 API, off/medium, JSON object, JSON Schema 和 strict tool.

DSpark proposal 在选择 candidate 前应用 grammar mask. Target 还应用数值思考预算.

随机输出的接受规则使用完整条件 target / proposal 概率, 应用配置的采样参数.
Candidate 接受概率为 `min(1,p(x)/q(x))`, 拒绝时使用归一化的 `max(p-q,0)`, bonus 使用 target 分布 `p`.
Greedy 输出使用 target argmax 验证.
Context 和状态提交参见 [架构](../architecture/speculation.zh.md#dspark).

## XML 工具限制

工具参数需要直接 object root, 不支持根 `$ref` 或 `anyOf`.
Property 类型需已知且无歧义.
不支持显式开放的 `additionalProperties`.
非 strict tool 省略它时, object 对已知 property 闭合.
允许 property 局部 `$ref` 和无歧义 `anyOf`.
XML 中 string / nonstring 混合 union 有歧义, 会被拒绝.

显式 strict XML string 不支持 pattern, length 或 format 约束.
其他工具描述可以包含这些 keyword, 由客户端校验.
String enum / const 不能包含参数结束 delimiter.
候选值不能只在外围空格, 换行或 tab 上不同.
Parser 从唯一 schema value 恢复 padding.
XML `anyOf` 不能带 sibling const / enum.
JSON answer schema 不受 XML 专属限制.

无约束 string 去除 template markup 的一个前导 / 尾随换行.
保留引号, 空格, 额外换行和 function/tool-close literal.
Enum / const string 恢复精确 schema value, 包括其自身换行.
Parser 只在参数值外识别结束 tag.

禁用工具时, 普通文本或 JSON 答案中的 literal tool tag 保持为文本.
启用工具时, 起始标记专用于调用. 字符串参数仍可包含标记字面量.

Grammar mask 约束结构. 正文重复和任务失败仍可能发生.
`</thinking>` 和 `<next_thinking>` 是普通文本, 并非思考分隔符.
持久会话验收保留客户端原有的重复检测和进展超时.
