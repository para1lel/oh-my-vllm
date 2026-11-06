# 工具和结构化输出

## 子功能

- 使用嵌套 `function` 形式的 Chat 工具调用.
- 使用平铺函数形式的 Responses 工具调用.
- 带结构化参数的必选 `lookup` 调用.
- 使用严格对象 schema 的 Chat JSON Schema 输出.

参见[请求规范化](../../../../crates/zmq-worker/src/serving/request.rs)及 [Qwen 输出解析](../../../../crates/zmq-worker/src/models/qwen38/output.rs).

## 如何进入 (用户视角)

完成 [Launch](../SKILL.zh.md#launch) 后, 向 Chat 或 Responses 发送 `tools` 和 `tool_choice`.
Chat 使用 `tools[].function`. Responses 将函数字段直接放在 `tools[]` 中.
使用 Chat 的 `response_format` 或 Responses 的 `text.format` 选择结构化输出格式.
helper 的 schema drive 使用 Chat.

## 通过 verify.py 驱动

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature tools --binary target/debug/oh-my-vllm-zmq-worker
```

[Tools drive](../drive.py) 采集 Chat 和 Responses 工具回复.
它检查 Chat 函数名, 并按 JSON 解析参数.
它检查 Responses 输出项类型.
随后, 它请求严格的 Chat schema, 将 `key` 限定为 `hello`.
它解析回复, 并将对象与固定 fixture 值对比.
读取 `features/tools.json` 及对应的 `http/` 记录.

## 注意事项

JSON Schema 不能与启用的工具同时使用.
fixture 按请求控制字段选择固定回复.
它不应用 GPU grammar mask, 也不执行模型采样.
工具结果不是外部函数执行.
生成行为使用 [checkpoint 路径](../SKILL.zh.md#checkpoint-服务路径)及模型局部的约束检查.
