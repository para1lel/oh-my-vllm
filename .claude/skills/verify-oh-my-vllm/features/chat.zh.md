# Chat 回复

## 子功能

- 通过 `POST /v1/chat/completions` 取得完整 JSON 回复.
- 通过 `stream=true` 取得 SSE chunk 及最后的 `[DONE]` 标记.
- 通过 `stream_options.include_usage=true` 取得输出 token 用量.

参见 [Chat 请求解析](../../../../crates/zmq-worker/src/serving/request.rs)及[输出事件](../../../../crates/zmq-worker/src/serving/events.rs).

## 如何进入 (用户视角)

完成 [Launch](../SKILL.zh.md#launch) 后, 从 `GET /v1/models` 读取模型 ID.
将该 ID 放在 `model` 中, 并提供 `messages` 数组和 `max_tokens`.
JSON 使用 `stream=false`, SSE 使用 `stream=true`.
helper 对这两个请求使用相同 prompt 和 seed.

## 通过 verify.py 驱动

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature chat --binary target/debug/oh-my-vllm-zmq-worker
```

[Chat drive](../drive.py) 通过新启动的 Rust 服务发送 JSON 和 SSE 请求.
它比较回复文本和输出 token 数.
它检查固定 fixture 文本, 完成标记, 结束原因及大于零的输出用量.
CPU fixture 回复为 `test answer`.
读取证据目录中的 `http/*-chat-json/response.body`, `http/*-chat-sse/response.body` 和 `features/chat.json`.

## 注意事项

重复 prompt 可能复用 prefix cache.
即使输出 token 数相同, JSON 和 SSE 的 prompt cache 计数也可能不同.
helper 保存每份 usage 对象的全部内容.
固定 fixture 回复检查解析和传输. 它不检查模型生成或采样.
推理使用 [checkpoint 路径](../SKILL.zh.md#checkpoint-服务路径)及所选模型的验收流程.
