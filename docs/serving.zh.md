# 服务指南

Rust 服务通过 ZMQ/msgpack 进程边界使用 Python 推理 worker.
Worker 加载本地 checkpoint, 使用仓库内的 CUDA 模块.

## 启动

使用[开发环境](development.zh.md).
安装运行时包:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/runtime.txt -r requirements/kernels.txt
```

构建可执行文件:

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

按[算子环境配置](operators/development.zh.md#工具)构建原生模块.
将 `MODEL_ROOT` 设为 checkpoint 目录.
启动服务:

```bash
scripts/with-env.sh scripts/with-gpu.sh target/release/oh-my-vllm-zmq-worker --model "$MODEL_ROOT" --socket /tmp/oh-my-vllm.ipc serve
```

使用 `--max-num-batched-tokens` 设置每个调度步骤的 token 预算. 默认为 `32768`.
使用 `--max-num-seqs` 设置运行中请求的最大数量. 默认为 `4`.
将这些选项放在 `serve`, `run` 或 `bench` 前面.

调度器优先选择 decode 请求.
它将剩余 token 预算分配给 prefill 请求.
它按 cache 边界调整 prefill 数量, 并将未使用的 token 分配给其他 prefill 请求.

每个服务进程使用不同的 `--socket` 路径.
使用 `serve --listen` 设置地址. 默认地址为 `127.0.0.1:8000`.

服务没有身份认证. 主机地址保存在被忽略的本地配置中.
使用 Ctrl-C 或 SIGTERM 停止服务及其 worker.

## HTTP API

| 方法和路由 | 功能 |
|---|---|
| `GET /v1/models` | 模型列表 |
| `POST /v1/chat/completions` | Chat JSON 或 SSE, 工具, reasoning 和 usage |
| `POST /v1/responses` | Responses JSON 或 typed SSE, 工具, reasoning 和历史 |
| `GET /v1/responses/{id}` | 读取已存储的 response |
| `DELETE /v1/responses/{id}` | 删除已存储的 response |
| `GET /diagnostics` | 服务, 资源, cache 和 worker 状态 |

请求使用模型 ID `qwen3.8-27b-fp8`.
每条请求返回一个 completion.

User 消息可按输入顺序包含文本, 图片和视频.
Chat 内容部分使用 `text`, `image_url` 和 `video_url`.
Responses 内容部分使用 `input_text`, `input_image` 和 `input_video`.

为每个图片或视频提供 HTTP URL, HTTPS URL 或 base64 data URI.
Data URI 的媒体类型必须与图片或视频类型一致.
图片 detail 可为 `auto`, `low` 或 `high`.
每条请求历史最多包含 16 个媒体项.

Worker 下载并解码媒体. 每个编码后的媒体项上限为 64 MiB.
每张图片的解码像素上限为 67108864. 视频抽样帧的总大小上限为 512 MiB.
源高度和宽度分别不超过 65536 像素.
视频以每秒 2 帧抽样, 最多 768 帧.

每对时间帧在 prompt 中有一个时间戳.

媒体 token 使用三轴 RoPE 位置. 媒体请求具有独立的前缀 cache 标识.

服务对不接受的字段和输入类型返回校验错误.
使用 `previous_response_id` 续接已存储的 Responses 历史.

Responses 默认存储 response, `store=false` 时不存储.

请求示例:

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"reasoning_effort":"off"}'
```

## 离线命令

```bash
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker run --tokens 1 2 3 --max-tokens 8
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker bench --input-len 785 --output-len 8 --warmup 0 --repetitions 1
```

Run 和 Bench 使用固定输出长度.
在子命令前增加 `--model "$MODEL_ROOT"` 可执行模型计算.

在子命令前使用 `--speculative-mode mtp --num-speculative-tokens 4` 可启用 MTP4.
使用 `--speculative-mode none` 可启用普通解码.

## 输出控制

使用 `min_tokens` 可在最小输出长度之前屏蔽 EOS 和 stop 字符串.
最小长度不得大于输出上限.
使用 `temperature`, `top_k`, `top_p` 和 penalty 字段设置采样.
使用 `repetition_guard=false` 可关闭生成 token 的循环保护.

Chat 使用 `response_format` 设置 JSON 输出. Responses 使用 `text.format`.
JSON Schema 不能与已启用的工具一起使用.

Tool choice 可为 `auto`, `none`, `required` 或指定函数名.
使用 `parallel_tool_calls=false` 最多允许一次调用.
Worker 对 target 和 proposal 分布应用 grammar mask.
只有已接受的 token 才会修改请求的 grammar 状态.

[执行规则](execution.zh.md)为含 prefill 的 Execute 消息选择 Eager, 为仅含 decode 的消息选择 CUDA Graph 重放.
Worker 不使用 PDL.
推理性能测量使用[端到端流程](performance.zh.md).

模型临时缓冲区使用 [CUDA 内存池](../python/oh_my_vllm/operators/runtime/cuda/scratch_allocator.cuh#L19)<!-- source: inline cudaMemPool_t scratch_pool(int device) { -->.
Worker 在报告就绪前准备存储.
存储大小根据 token 预算确定, 且最多使用空闲 GPU 显存的一半.
内存池保留这些存储供后续计算使用. `gpu_reserved_bytes` 统计值包含这些存储.

## 占位实现

使用 `--model stub` 可在无 GPU 时执行服务测试.
占位实现提供随机文本, 工具调用样例和简单 JSON 样例.
它不加载 checkpoint, 不执行模型计算.
占位实现的计时不代表推理性能.
模型初始化或执行出错时, 不会切换到占位实现.
