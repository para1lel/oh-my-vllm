# Responses 历史和流式输出

## 子功能

- 通过 `POST /v1/responses` 创建.
- 通过 `GET /v1/responses/{id}` 读取.
- 使用 `previous_response_id` 继续文本对话.
- 通过 `DELETE /v1/responses/{id}` 删除, 随后读取返回 404.
- 通过 `store=false` 取得带类型的 SSE.

参见 [Responses 服务](../../../../crates/zmq-worker/src/serving/mod.rs)及[流式事件](../../../../crates/zmq-worker/src/serving/events.rs).

## 如何进入 (用户视角)

完成 [Launch](../SKILL.zh.md#launch) 后, 向创建路由发送公共模型 ID 和 `input`.
保存返回的 response ID.
将其用于读取, 删除或下个请求的 `previous_response_id`.
使用 `stream=true` 取得带类型的 SSE 事件.

## 通过 verify.py 驱动

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature responses --binary target/debug/oh-my-vllm-zmq-worker
```

[Responses drive](../drive.py) 创建父响应, 读取它, 再创建续接的子响应.
它删除父响应, 并记录读取父响应时返回的 404.
它在父响应删除后读取子响应, 然后删除子响应.
它还采集另一个禁用存储的流.
流必须以 `response.completed` 结束, 且 sequence number 连续.
读取证据目录中的 `http/` 请求/响应对和 `features/responses.json`.

## 注意事项

默认启用存储.
服务执行存储时限, 数量和字节上限.
已存储 ID 属于当前服务实例. 重新启动后使用新的父响应.

CPU fixture 为历史创建合成 prompt tokens.
这不检查 tokenizer 历史或 checkpoint 生成.
这些检查使用 [checkpoint 路径](../SKILL.zh.md#checkpoint-服务路径).
