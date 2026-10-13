# HTTP API 用法

## 启动

按 [开发指南](../development.zh.md) 设置环境和模型.
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

源码身份及测量范围见 [验收索引](../acceptance.zh.md).

## API 范围

| 方法和路由 | 合同 |
|---|---|
| `GET /v1/models` | 模型发现. |
| `POST /v1/chat/completions` | JSON 或 SSE, 文本, 函数工具, 工具历史, usage 和 finish reason. |
| `POST /v1/responses` | JSON item 或 typed SSE, 文本, 工具, 完整历史和 `previous_response_id`. |
| `GET /v1/responses/{id}` | 查询已保存响应. |
| `DELETE /v1/responses/{id}` | 删除已保存响应. |
| `GET /diagnostics` | 服务健康状态, 资源 reservation, 缓存计数及最新 worker 内存 snapshot. |

输入仅支持文本, 每请求一个 completion.
未知字段和不支持功能会报错, 包括 logprobs, background mode 和自动截断.
支持的 OMP provider payload 和 event consumer 有测试.
API 子集遵循 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 和 [Responses](https://developers.openai.com/api/reference/resources/responses/methods/create) 约定.

## 采样, 工具与停止

采样支持 temperature, top_p, top_k, seed, 以及 frequency/presence/repetition penalty.
Checkpoint 默认 temperature `1`, top_p `.95`, top_k `20`.

公共 HTTP 服务默认增加 presence_penalty `0.5`. Frequency penalty 保持 `0`, repetition penalty 保持 `1`.
普通, MTP4, DSpark 和测试服务使用相同的默认值.
独立思考预算默认仍为 `-1`.
用 `serve --override-generation-config` 和 JSON object 修改支持的采样默认值.
请求显式值优先于服务值, 服务值优先于 checkpoint 值.

仅允许支持的采样键. 无效 CLI 值会在模型加载前失败.
请求中为 null 的采样字段使用默认值. 显式 `0` 关闭默认 presence penalty.

Diagnostics 报告服务默认值. 请求 metric 报告合并后的服务和请求显式采样字段.
Python 为省略的字段补充 checkpoint 默认值.

[采样决策](../decisions/ADR-015-shared-serving-sampling-defaults.zh.md) 记录模型指导, 选定值和验证限制.

公共 `repetition_guard` 默认值为 `true`. 请求和服务覆盖值接受布尔值.
显式 `false` 允许有意重复输出. 请求中的 null 值使用服务默认值.

保护限制已生成输出中的循环. 它保留 grammar 必需的续接, 草稿检查不改变已提交历史.
算法见 [循环检测](../../python/oh_my_vllm/worker/repetition.py)和 [采样集成](../../python/oh_my_vllm/worker/sampler.py).

保护不能保证任务完成, 也不能阻止这些限制之外的循环.
启用时也会影响有意重复的文本. 保持客户端的进展和重复检查有效.

离线 Run/Bench 显式禁用保护, 使用 greedy sampling, 固定输出长度并忽略 EOS.

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

两种 API 都接受顶层 `thinking_token_budget`.
非负整数限制生成的思考内容 token 数. `-1` 表示不单独限制.
请求省略此字段时, 服务选项 `--thinking-token-budget` 提供默认值.
它默认为 `-1`, 与 vLLM 默认行为相同.
请求值优先. Effort 不选择数值预算.
达到预算后, target mask 只允许 `</think>`, 然后继续生成答案或工具调用.
启用思考时, 零预算立即强制生成此标记. Off 不产生思考块.

结束标记占一个输出 token. 原有 reasoning usage 包含此 token, 因此可能为 `budget + 1`.
总输出与上下文限制继续生效. 它们可能在结束标记或答案之前停止生成.
该预算不限制答案 token, 也不检测正文重复.
Responses 输出与服务诊断记录有效预算.

当前安装的 OMP OpenAI adapter 省略此字段. 用服务选项统一覆盖两种 API, 包括自动压缩.
不能从 OMP `max` 或服务 `xhigh` 推导出数值预算.
[思考预算决策](../decisions/ADR-012-reasoning-token-budget.zh.md) 给出来源比较和推测解码契约.

Chat 将 `reasoning_content` 与文本和工具参数分开.
Responses 的 readable reasoning event / item 来自模型自身输出.
不另外生成摘要.
支持的 `chat_template_kwargs` 为 `preserve_thinking`, `enable_thinking` 和 `reasoning_effort`.
直接设置 `enable_thinking=false` 选择 off; 其他 template 选项报错.

历史工具参数以 JSON string 输入, 在 templating 前转为 object.
Responses reasoning 映射回 assistant `reasoning_content`.
Responses 顶层 instruction 只适用于当前响应.
续接不继承这些 instruction.
