# 服务指南

Rust 服务通过 ZMQ/msgpack 进程边界使用 Python 占位 worker.
占位实现不加载模型权重, 不使用 GPU.

## 启动

使用[开发环境](development.zh.md).
安装运行时包:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/runtime.txt
```

构建可执行文件:

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

启动服务:

```bash
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm.ipc serve
```

每个服务进程使用不同的 `--socket` 路径.
通过 `serve --listen` 设置地址. 默认地址为 `127.0.0.1:8000`.

服务没有身份认证. 主机地址放在被忽略的本地配置中.
使用 Ctrl-C 或 SIGTERM 停止服务及其 worker.

## HTTP API

| 方法与路径 | 功能 |
|---|---|
| `GET /v1/models` | 模型列表 |
| `POST /v1/chat/completions` | Chat JSON 或 SSE, 工具, reasoning 和用量 |
| `POST /v1/responses` | Responses JSON 或带类型的 SSE, 工具, reasoning 和历史 |
| `GET /v1/responses/{id}` | 查询已存储响应 |
| `DELETE /v1/responses/{id}` | 删除已存储响应 |
| `GET /diagnostics` | 服务, 资源, cache 和 worker 状态 |

以模型 ID `qwen3.8-27b-fp8` 发送文本输入.
服务为每个请求提供一个 completion.

未知字段和不支持的输入类型会产生验证错误.
通过 `previous_response_id` 续接已存储的 Responses 响应.

Responses 响应默认存储, 显式设置 `store=false` 时不存储.

请求示例:

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"reasoning_effort":"off"}'
```

## 离线命令

```bash
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker run --tokens 1 2 3 --max-tokens 8
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker bench --input-len 785 --output-len 8 --warmup 0 --repetitions 1
```

Run 和 Bench 保持固定输出长度.
HTTP 占位实现的输出长度在输出预算内随机选择.

通过 `--speculative-mode` 选择 `none` 或 `mtp`, 模拟对应 draft 传输.
这些模式不执行模型计算.

占位实现提供随机文本, 示例工具调用和简单 JSON 样例.
占位实现没有模型 tokenizer, grammar masks 或 GPU profiler.

占位实现的耗时不代表推理性能.
后续推理实现仍受[运行时契约](requirements.zh.md)约束.
