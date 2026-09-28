# OpenAI 兼容本地服务

**状态（2026-09-23）：** 默认 CUDA Qwen Worker 通过真实 MTP4 JSON/工具约束、思考档位、生命周期及两种 oh-my-pi 核心任务。两种 API 的长 strict-JSON 请求均复用 130928 个缓存 token。证据及保留的回答局限见 [acceptance.zh.md](acceptance.zh.md)。

**历史审计发现，现已修复：** 2026-09-23 审计发现请求错误可能使所有活跃流失败（SRV-01）、父进程终止后 worker 孤儿化（SRV-02）、HTTP 错误分类错误（SRV-03），以及 `length` 结束时暂扣字节丢失（SRV-04）。当前源码与回归状态见 [audit-2026-09-23.zh.md](audit-2026-09-23.zh.md)；当前完整 GPU 与 12 行框架验收仍是独立的待完成检查。

## 运行

```bash
scripts/with-env.sh cargo build --release
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-speculative-tokens 4 serve
```

默认 `127.0.0.1:8000`，模型 ID `qwen3.8-27b-fp8`，本地工作流不鉴权。按需设置 `serve --listen` 和 `--served-model-name`。Rust 负责 HTTP（Axum）、协议适配、响应状态、在线接纳、取消、调度和 KV。Python 负责 tokenizer/模板、XGrammar 状态和 mask、增量 detokenization 及项目自有 GPU 模型执行。不引入 Python 调度器或工具执行器。KV 块大小保持 784。

## 兼容范围

- `GET /v1/models`。
- `POST /v1/chat/completions`：流式 SSE 和非流式 JSON、文本与函数工具、工具结果历史、usage 和 finish reason。
- `POST /v1/responses`：专用类型化 SSE 事件和 JSON output item、文本、函数工具、完整历史回放及 `previous_response_id` 续接。
- `GET /v1/responses/{id}` 和 `DELETE /v1/responses/{id}`。
- 工具选择 auto、none、required、指定函数；除非 `parallel_tool_calls=false`，允许多个调用。服务生成调用，由 OMP 执行工具。
- 采样：temperature、top_p、top_k、seed、frequency/presence/repetition penalty。默认取本地 generation_config.json（temperature 1、top_p .95、top_k 20）。离线 Run/Bench 保留贪心、固定长度、忽略 EOS 行为。
- 默认输出预算 8,192 token，包含 reasoning 和正文。CLI 默认上下文 65,536；prompt 加输出预算超过限制时报错，不自动截断。EOS 终止生成；普通文本支持最多四个非空 stop 字符串，包括跨 token/UTF-8 边界。stop 不能打断已启用工具或结构化输出。耗尽长度返回 `length` / `incomplete`，未完成工具调用绝不交付执行。
- 仅文本输入，每请求一个 completion。未知顶层字段及 logprobs、background mode、自动截断等不支持功能报错。这是文档规定的子集，不等于完整 OpenAI 平台。

协议参考：[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[Responses](https://developers.openai.com/api/reference/resources/responses/methods/create)、[Responses SSE](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。也测试已安装 OMP 18.2.6 provider 的载荷和事件消费逻辑。

## 思考与历史回放

Chat 接受 `reasoning_effort`，Responses 接受 `reasoning.effort`，默认 medium。原生 off/low/medium/xhigh 映射模型模板，high 明确作为 xhigh 别名，OpenAI none 作为 off 别名。这些是 prompt 控制，不是固定 reasoning 预算，也不承诺质量/时延单调。未知档位失败。思考与 content/JSON/工具参数分开：Chat 使用 reasoning_content，Responses 使用可读 reasoning summary event/item，内容是本地模型输出的 reasoning，不是另生成的摘要。

支持 OMP 校验过的 chat_template_kwargs preserve_thinking/enable_thinking/reasoning_effort 扩展，其他模板选项失败。直接 enable_thinking=false 也选择 off。历史工具参数以 JSON 字符串输入，套模板前转为对象。Responses 可读 reasoning 映射回 assistant 的 reasoning_content。Responses 顶层 instructions 只作用于当前响应，不经 previous_response_id 继承。

## 约束与 MTP

JSON object、JSON Schema 和工具参数 grammar 在**采样前**应用。reasoning 可以出现在约束答案前。同一请求不能同时使用响应 JSON 和启用的工具。strict=true 要求封闭对象且所有属性必填。不支持 schema 直接失败，不以生成后验证重试替代。

XGrammar 构造 Qwen 原生 XML 工具 grammar 和 JSON 答案 grammar。MTP 中，每个已调度 draft 位置及 bonus 位置都使用其推测前缀对应 mask；模拟 grammar 前进后回滚，只有被接受输出推进持久状态。无效 draft 在首个被 mask 的位置拒绝，不关闭 MTP，不切换到 vLLM 调度器。CPU 测试覆盖 mask 回滚、拒绝 draft 和 reasoning-end 跨越。真实 MTP4 运行进一步覆盖两 API 的 off/medium、JSON object、JSON Schema 和 strict tool；每个完成案例都实际提出并接受 draft。

支持的 schema keyword 明确列在 worker/serving.py。不支持的后端功能/组合拒绝，包括未知 string format，以及 pattern/format 与长度限制组合。工具 XML 有额外编码限制：

- 工具参数必须直接以 object 为根（无根 $ref/anyOf），属性类型明确且已知。不支持显式开放 additionalProperties；非 strict 工具省略 additionalProperties 时，也只允许已知属性。
- 支持局部属性 $ref 和无歧义 anyOf；拒绝混合字符串/非字符串 XML union，因为原始 `null`/`123` 无法区分 JSON 类型。
- 拒绝带 pattern、length 或 format 的 XML 字符串。string enum/const 不能包含参数结束分隔符，也不能只在周围空格/换行/tab 上不同；编码 padding 从唯一 schema 值恢复。XML anyOf 不能有同级 const/enum。JSON 答案 schema 不受这些 XML 专有限制。
- 无约束字符串去掉 Qwen XML 模板标记带来的一个前导/尾随换行，与所支持表示一致。引号、空格、额外换行及函数/工具结束字面量保留。enum/const 字符串恢复精确 schema 值，包括自身换行。解析器只在参数值之外识别结束标签。普通文本或 JSON 答案中的工具标签字面量不视为工具调用。

## 状态与资源限制

Responses 默认 store=true，OMP 通常发送 store=false。存储响应及解析后的对话快照在完成一小时后过期。限制 1,000 条记录和 256 MiB 序列化响应/历史载荷，可用 response-ttl-seconds、response-capacity、response-max-bytes 配置。对象/分配额外开销不计入序列化字节预算。达到容量时淘汰最旧记录；超大记录显式失败。未知/已删除/过期 ID 返回 404，重启使 ID 失效。并发续接复制快照；删除祖先不影响已开始请求或已保存子响应。

HTTP body 上限 8 MiB；接纳 channel 和活跃请求上限各 64；Rust 最多调度 32 并发序列。每个 stream 有 256 事件 buffer。客户端断开或过慢时只取消其请求，不取消引擎。默认请求超时 600 秒（可配置）；准备/步骤 RPC 的 deadline 为 120 秒并检测 worker 退出。请求局部的校验或生成错误只让该请求失败；CUDA/设备及 worker 致命错误使待处理请求失败，后续提交在重启前返回 unavailable。Ctrl-C 和 SIGTERM 关闭活跃 stream，并请求 Python 正常关闭。HTTP 连接排空和 worker 清理共用有界关闭期限（默认 35 秒，可用 `--shutdown-grace-seconds` 配置）；停读的 HTTP 客户端在到期时被强制断开。不单独抢占正在执行的 kernel。

## OMP 任务与验证

启用 MTP 启动服务，然后分别运行：

```bash
scripts/with-env.sh python scripts/agentic-acceptance.py --api chat --output-dir /tmp/oh-my-vllm-agentic-chat
scripts/with-env.sh python scripts/agentic-acceptance.py --api responses --output-dir /tmp/oh-my-vllm-agentic-responses
```

保持 MTP 开启，再执行 12 个真实约束案例：

```bash
scripts/with-env.sh python scripts/serving-acceptance.py --output-dir /tmp/oh-my-vllm-serving-constraints
```

`scripts/serving-lifecycle.py --server-log /tmp/serve.log --output /tmp/serving-lifecycle.json` 还检查五种思考档位、存储响应查询/续接/删除、并发普通/约束请求，以及断开后的成功请求。用 `RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug` 启动服务，将输出重定向到指定日志。脚本要求日志证明混合请求共享过同一调度 batch，且断开请求通过 Worker RPC 取消并释放；只在非空生成 SSE 内容之后断开。非默认端口时，三个脚本都指定 --base-url。检查日志中实际 MTP 活动，运行后立即停止服务和 worker。

agentic 脚本创建隔离 models.yml，允许 read/grep/glob，并要求：

> 阅读仓库 README、架构文档和必要源码，介绍项目目标、架构、运行方法和当前完成状态，引用支持答案的文件，不修改文件。写最终答案前，至少用 read 工具读取 crates/scheduler/src/lib.rs 和 python/oh_my_vllm/worker/model_runner.py；仅目录列表不算。

脚本保存客户端 JSONL、stderr 和精确命令/配置。验证工具执行、带工具结果的后续模型请求，以及基于文件的最终回答。JSONL 必须包含对两份指定实现文件的成功读取。退出码零本身不代表通过。超时时脚本终止自有进程组。本地虚拟 API key 不是凭据。

OMP 使用原生非 strict 工具 schema，因为其 strict 归一化会引入有歧义的 nullable XML 字符串。独立 strict API 测试覆盖支持的 schema。OMP minimal 映射 off：已安装客户端 off 控制会回退到最低支持档位。low→low、medium→medium、high/xhigh/max→xhigh。客户端 off/medium 均通过脚本化协议测试；两次真实 agentic API 任务在 medium 下通过，真实 API 检查覆盖全部五档。

不设新的严格服务吞吐门槛。看真实日志中的排队、准备、TTFT、端到端输出 tok/s、步骤、缓存命中和 MTP proposed/accepted。DEBUG 提供 RPC/worker 主机计时，不称其为 CUDA kernel 时间。调查持续排队、停顿、意外零 proposal/acceptance、过多抢占、显存增长或 worker 错误；预热后比较同类工作负载。既有 EngineCore >=95% 要求独立且不变。

tokenizer 使用 Transformers 和原生 Tokenizers DecodeStream 增量解码。词表属性只在准备时获取一次，避免每个输出 token 重复元数据工作。Python DEBUG 区分消息解码、主机执行和响应编码；这些时长包含同步，不是 CUDA kernel 计时。
