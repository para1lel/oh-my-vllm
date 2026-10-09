# HTTP 服务合同

Rust 负责 Axum HTTP, 接纳, 协议适配, 响应存储, 取消, 调度和逻辑 KV.
Python 负责 tokenization, template, XGrammar, detokenization 和 GPU 模型执行.
函数工具由客户端执行.

## 启动

按 [开发指南](development.zh.md) 设置环境和模型.
每个服务使用独立 IPC 路径:

```bash
scripts/with-env.sh cargo build --release
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-speculative-tokens 4 serve
```

默认监听 `127.0.0.1:8000`, 模型 ID 为 `qwen3.8-27b-fp8`.
本地服务无认证.
通过 `serve --listen` 和 `--served-model-name` 修改这些值.
实际服务器地址记入 `LOCAL.md`.

DSpark 设置 `OH_MY_VLLM_DRAFT_MODEL`, 显式选择模式:

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-dspark-serve.ipc --speculative-mode dspark serve
```

每个 worker 服务一种草稿模式, HTTP 请求不能切换模式.
原生 MTP4 可通过旧 4-token 选项或 `--speculative-mode mtp` 使用.
DSpark 至多 7 个草稿, 保持相同 target checkpoint, block784, API 和请求约束.

`--dspark-confidence-threshold` 初始设置为 `0.2`, 从首个 candidate 开始限制累计 confidence.
可返回 0 至 7 个 candidate, 返回 0 个时, 该步由目标执行单 token 解码.
使用 `0.0` 保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.

源码身份及测量范围见 [验收索引](acceptance.zh.md).

## API 范围

| 方法和路由 | 合同 |
|---|---|
| `GET /v1/models` | 模型发现. |
| `POST /v1/chat/completions` | JSON 或 SSE, 文本, 函数工具, 工具历史, usage 和 finish reason. |
| `POST /v1/responses` | JSON item 或 typed SSE, 文本, 工具, 完整历史和 `previous_response_id`. |
| `GET /v1/responses/{id}` | 查询已保存响应. |
| `DELETE /v1/responses/{id}` | 删除已保存响应. |

输入仅支持文本, 每请求一个 completion.
未知字段和不支持功能会报错, 包括 logprobs, background mode 和自动截断.
支持的 OMP provider payload 和 event consumer 有测试.
API 子集遵循 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 和 [Responses](https://developers.openai.com/api/reference/resources/responses/methods/create) 约定.

## 采样, 工具与停止

采样支持 temperature, top_p, top_k, seed, 以及 frequency/presence/repetition penalty.
Checkpoint 默认 temperature `1`, top_p `.95`, top_k `20`.
离线 Run/Bench 使用 greedy sampling, 固定输出长度并忽略 EOS.

Tool choice 支持 auto, none, required 和指定函数.
未设置 `parallel_tool_calls=false` 时允许多个调用.
服务生成 tool call; OMP 执行并返回结果.

默认输出预算为 8192 token, 包括思考和文本.
CLI 默认上下文为 65536 token.
Prompt 加输出预算超过上下文限制时即报错.
EOS 结束生成.
普通文本最多支持 4 个非空 stop string, 可跨 token 和 UTF-8 边界.
Stop 不能中断启用的工具或结构化输出.
长度耗尽返回 `length` 或 `incomplete`.
未完成的 tool call 不交付执行.

## 思考与重放

Chat 使用 `reasoning_effort`; Responses 使用 `reasoning.effort`.
默认 medium.
原生 off/low/medium/xhigh 控制 template; high 别名为 xhigh, none 别名为 off.
未知 effort 报错.
这些 prompt 控制不指定固定 token 预算或单调的质量 / 延迟关系.

Chat 将 `reasoning_content` 与文本和工具参数分开.
Responses 的 readable reasoning event / item 来自模型自身输出.
不另外生成摘要.
支持的 `chat_template_kwargs` 为 `preserve_thinking`, `enable_thinking` 和 `reasoning_effort`.
直接设置 `enable_thinking=false` 选择 off; 其他 template 选项报错.

历史工具参数以 JSON string 输入, 在 templating 前转为 object.
Responses reasoning 映射回 assistant `reasoning_content`.
Responses 顶层 instruction 只适用于当前响应.
续接不继承这些 instruction.

## Grammar 与推测 token

JSON object, JSON Schema 和 strict tool grammar 在采样前生效.
思考可先于约束答案.
一个请求不能同时使用 JSON 答案格式和启用工具.
Strict schema 需要 closed object, 全部 property 都 required.
不支持的 schema 明确报错.
支持的 keyword 清单位于 `python/oh_my_vllm/worker/serving.py`.
不支持未知 string format, 或 pattern / format 与 length bound 的组合.

XGrammar 提供 Qwen XML tool grammar 和 JSON answer grammar.
每个 MTP 或 DSpark draft 和 bonus position 使用其 speculative prefix 对应的 mask.
模拟 grammar 推进后回滚; 只有保留输出推进持久状态.
被 mask 的草稿在第一个无效位置失败.
CPU 测试覆盖回滚, 拒绝和 reasoning-end 跨界.
真实 MTP4 证据覆盖两个 API, off/medium, JSON object, JSON Schema 和 strict tool.

DSpark proposal 在选择 candidate 前使用相同 mask.
随机输出的接受规则使用完整条件 target / proposal 概率, 应用配置的采样参数.
Candidate 接受概率为 `min(1,p(x)/q(x))`, 拒绝时使用归一化的 `max(p-q,0)`, bonus 使用 target 分布 `p`.
Greedy 输出使用 target argmax 验证.
Context 和状态提交参见 [架构](architecture.zh.md#dspark).

## XML 工具限制

工具参数需要直接 object root, 不支持根 `$ref` 或 `anyOf`.
Property 类型需已知且无歧义.
不支持显式开放的 `additionalProperties`.
非 strict tool 省略它时, object 对已知 property 闭合.
允许 property 局部 `$ref` 和无歧义 `anyOf`.
XML 中 string / nonstring 混合 union 有歧义, 会被拒绝.

XML string 不支持 pattern, length 或 format 约束.
String enum / const 不能包含参数结束 delimiter.
候选值不能只在外围空格, 换行或 tab 上不同.
Parser 从唯一 schema value 恢复 padding.
XML `anyOf` 不能带 sibling const / enum.
JSON answer schema 不受 XML 专属限制.

无约束 string 去除 template markup 的一个前导 / 尾随换行.
保留引号, 空格, 额外换行和 function/tool-close literal.
Enum / const string 恢复精确 schema value, 包括其自身换行.
Parser 只在参数值外识别结束 tag.
普通文本或 JSON 答案中的 literal tool tag 保持为文本.

## 存储与资源限制

Responses 默认 `store=true`; OMP 通常发送 `store=false`.
保存的响应及已解析历史 snapshot 在完成 1 小时后过期.
默认上限为 1000 条和 256 MiB 序列化 response / history 字节.
Object / allocation 额外开销不计入该预算.
通过 `--response-ttl-seconds`, `--response-capacity` 和 `--response-max-bytes` 修改.
达到容量时淘汰最旧记录; 过大记录报错.
未知, 已删除或过期 ID 返回 404; 重启使 ID 失效.
并发续接复制 snapshot.
删除祖先不会影响已开始请求和已存子响应.

| 资源 | 默认限制 |
|---|---:|
| HTTP body | 8 MiB |
| 接纳 channel / 活跃请求 | 64 / 64 |
| 调度 sequence | 32 |
| Stream channel | 256 event |
| 待发送 stream 队列 | 1024 event 或 8 MiB 序列化字节 |
| 背压 grace | 30 秒 |
| 请求 timeout | 600 秒 |
| Prepare/step RPC deadline | 120 秒 |
| 共享 shutdown grace | 35 秒 |

Channel 满时暂停该请求, 启动单调背压 deadline.
待发送 event 保持顺序, 包括最终 event.
Pending 队列为空且 channel 至少有 128 个空 slot 后恢复生成.
稀疏读取不重置 deadline.
断连立即标记取消; 下一次引擎检查执行取消.
Deadline 检查在操作之间发生; RPC 或阻塞 send 可能延迟取消.
后台 CPU preparation 本身不阻塞该检查.
容量允许时暂停请求保留 KV; 优先级抢占仍可淘汰它.

请求局部 validation 或 generation 失败只影响该请求.
致命 CUDA / device / worker 失败会终止待处理请求, 后续提交在重启前返回 unavailable.
Ctrl-C 和 SIGTERM 关闭 stream, 请求 worker shutdown.
HTTP drain 和 worker cleanup 共享可配置的 `--shutdown-grace-seconds` deadline.
停滞 reader 在 deadline 断开; 内核不做单独抢占.

## 验证

使用 `scripts/serving-acceptance.py`, `scripts/serving-lifecycle.py` 和 `scripts/agentic-acceptance.py`.
每种草稿模式分别通过 `--api chat` 和 `--api responses`, 使用 target checkpoint 完成真实 OMP 任务.
使用 `--omp` 或 `PATH` 上的可执行程序.
客户端 JSONL 必须记录 `read` 成功读取 `crates/scheduler/src/lib.rs` 和 `python/oh_my_vllm/worker/model_runner.py`.
仅列出目录不能满足条件.
工具结果必须回传模型.
记录代理保留包含每个源码读取结果的实际转发模型请求.
客户端最后一条答案的 response ID 必须与记录的 HTTP 响应和实际模式服务日志匹配.
答案必须使用结果并引用文件.
代理任务期间保持仓库不变.

检查实际排队 / 准备时间, TTFT, 速率, 步数, 前缀复用和 verified / accepted draft.
`verified_draft_tokens` 统计已调度 candidate, 包括被拒绝 candidate, 后续 proposal 使用另一计数.
Completion 日志保留 `proposed_draft_tokens`, 作为已调度 candidate 数的旧 alias.
核对 `BENCH_CONFIG` 中的实际模式及验收脚本显式 `--speculative-mode` 选项.
原生 MTP4 和 DSpark 分别通过两个 API 重跑约束, 混合 batch / 取消和长上下文检查.
不增加 HTTP 吞吐门槛.
脚本 exit status 或 scripted worker 本身不能证明真实模型验收.
参见 [测试](testing.zh.md) 和 [验收](acceptance.zh.md).
