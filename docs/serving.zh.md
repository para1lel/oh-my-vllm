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

[采样决策](decisions/ADR-015-shared-serving-sampling-defaults.zh.md) 记录模型指导, 选定值和验证限制.

公共 `repetition_guard` 默认值为 `true`. 请求和服务覆盖值接受布尔值.
显式 `false` 允许有意重复输出. 请求中的 null 值使用服务默认值.

保护只检查已生成 token, 包含各行草稿前缀, 窗口最多 `768` token.
它检测周期长度为 `1` 到 `256` token 的连续循环, 要求至少重复 `3` 次, 且覆盖 `64` 个重复 token.
在 grammar mask 之后, top-k/top-p 之前, 它将继续这些循环的所有下一 token 分数设为负无穷.
若没有剩余有限候选, 则保留原有掩码后分数. 因此, grammar 必需的续接仍可用.
它不检查提示词中的重复, 草稿检查也不改变已提交历史.

普通, MTP4 和 DSpark 的 target 分布使用相同规则. DSpark 提议概率在接受修正前包含此规则.

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
[思考预算决策](decisions/ADR-012-reasoning-token-budget.zh.md) 给出来源比较和推测解码契约.

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

[ADR-018](decisions/ADR-018-client-owned-tool-descriptions.zh.md) 说明请求准备策略.

XGrammar 提供 Qwen XML tool grammar 和 JSON answer grammar.

每个思考请求使用经过验证的单个结束 token ID, 包括纯文本和无限思考预算.
拼出相同结束字符串的普通 ID 不能结束思考.
在该 token 前屏蔽 EOS. 显式文本 stop 和总长度仍可提前结束请求.
纯文本正文在思考结束后保留原有空白形式和标签字面量.

启用工具时, 约束与 Rust parser 一样从 `<tool_call>` 开始.
生成调用的后续部分必须满足函数选择, 并行调用和 schema 约束.
显式 `strict=false` 关闭工具参数的 schema 约束. XML 解析仍生效.
[控制 token 决策](decisions/ADR-016-canonical-serving-control-tokens.zh.md) 给出合同和重复问题的限制.

每个 MTP 或 DSpark draft 和 bonus position 使用其 speculative prefix 对应的 mask.
模拟 grammar 推进后回滚; 只有保留输出推进持久状态.
被 mask 的草稿在第一个无效位置失败.
CPU 测试覆盖回滚, 拒绝和 reasoning-end 跨界.
真实 MTP4 证据覆盖两个 API, off/medium, JSON object, JSON Schema 和 strict tool.

DSpark proposal 在选择 candidate 前应用 grammar mask. Target 还应用数值思考预算.

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

## 存储与资源限制

Responses 默认 `store=true`; OMP 通常发送 `store=false`.
保存的响应及已解析历史 snapshot 在完成 1 小时后过期.
默认上限为 1000 条和 256 MiB 序列化 response / history 字节.
Object / allocation 额外开销不计入该预算.
通过 `--response-ttl-seconds`, `--response-capacity` 和 `--response-max-bytes` 修改.
达到容量时淘汰最旧记录; 过大记录报错.
未知, 已删除或过期 ID 返回 404. HTTP 服务重启使已存储 ID 失效.
Worker 替换保留已存储的历史, 但清空逻辑 KV.
并发续接复制 snapshot.
删除祖先不会影响已开始请求和已存子响应.

读取 HTTP body 或复制祖先历史前, 先保留请求容量.
查询已存响应时, 复制响应前使用相同的容量 reservation.
引擎, HTTP body 和已完成 stream 的 drain 共享 reservation, 全部 owner 释放后才归还容量.
因此, 读取缓慢的已完成 stream 仍占用 live request slot.

编码请求状态预算默认为 512 MiB. 通过 `--request-state-max-bytes` 修改.
该预算按序列化输入, 历史和输出大小的保守倍数计算.
JSON 树存储, allocator 开销及 Python 对象另外使用内存. RSS 与此预算独立测量.

| 资源 | 默认限制 |
|---|---:|
| HTTP body | 8 MiB |
| 排队 reservation / live reservation | 64 / 64 |
| 接纳 channel / 活跃请求 | 64 / 64 |
| 编码请求状态 reservation | 512 MiB |
| 调度 sequence | 32 |
| Stream channel | 256 event |
| 待发送 stream 队列 | 1024 event 或 8 MiB 序列化字节 |
| 背压 grace | 30 秒 |
| 请求 timeout | 3600 秒 |
| Prepare RPC deadline | 120 秒 |
| Worker step deadline | 600 秒 |
| 共享 shutdown grace | 35 秒 |
| Worker 替换尝试 | 5 |
| Worker 内存 snapshot 间隔 | 30 秒 |

Channel 满时暂停该请求, 启动单调背压 deadline.
待发送 event 保持顺序, 包括最终 event.
Pending 队列为空且 channel 至少有 128 个空 slot 后恢复生成.
稀疏读取不重置 deadline.
断连立即标记取消; 下一次引擎检查执行取消.
请求绝对 deadline 从读取 body 前开始, 包含排队, 准备和生成.
已过期的排队请求不能开始准备.
过载返回 503. 超出逻辑 KV 容量的请求返回 413.

用 `--worker-step-timeout-seconds` 设置单个 GPU step 的最长等待时间.
该值必须为正数, 包括首次编译和图捕获.
Step 超时会让活跃请求失败, 并启动 worker 恢复.
保持有限期限. 请求和 shutdown 期限分别生效.

Deadline 检查在操作之间发生; RPC 或阻塞 send 可能延迟取消.
后台 CPU preparation 本身不阻塞该检查.
容量允许时暂停请求保留 KV; 优先级抢占仍可淘汰它.

请求局部 validation 或 generation 失败只影响该请求.
未完成的 GPU 事务禁止后续继续使用该 worker 状态.
CUDA OOM, device 故障, worker 退出及无效执行回复让受影响请求失败, 并启动 worker 恢复.
HTTP listener 继续存活. 恢复期间新生成请求返回 503.

服务停止旧 worker 及其所属进程组, 再加载 replacement.
Scheduler 和旧 worker 的全部逻辑 cache identity 清空.
Replacement 必须报告相同容量. 新 epoch 允许重新准入.

默认恢复上限为 5 次加载尝试. 等待间隔为 1, 2, 4, 8 和 16 秒.
`--worker-restart-attempts` 设置更大时, 每次间隔最多 30 秒.
达到恢复上限后, HTTP 继续可用, 生成保持不可用.
Stream 明确报告失败. 服务不自动重放输出或工具调用.

Ctrl-C 和 SIGTERM 关闭 stream, 请求 worker shutdown.
HTTP drain 和 worker cleanup 共享可配置的 `--shutdown-grace-seconds` deadline.
停滞 reader 在 deadline 断开; 内核不做单独抢占.

无法捕获的 Rust SIGKILL 会触发 parent-death 机制, 向直接 GPU worker 发送 SIGKILL.
该信号之后的完整进程组清理需要外部 process supervisor. 正常关闭和 worker 恢复执行所属进程组清理.

`GET /diagnostics` 返回 Rust 请求, reservation, storage 及 KV 计数, 并附最新 Python 资源 snapshot.
通过 `--diagnostics-interval-seconds` 修改默认的 30 秒 snapshot 间隔.
Worker 报告各请求状态映射, graph cache, 编译 cache, 主机 RSS 及 GPU allocated / reserved 字节数.
逐请求 `SERVING_REQUEST_METRIC` 日志使用 UTC 身份及单调的排队 / 处理时长.
处理时间从 Python 准备前开始, 到生成完成结束. HTTP drain 使用不同生命周期.

## 验证

使用 `scripts/serving-acceptance.py`, `scripts/serving-lifecycle.py` 和 `scripts/agentic-acceptance.py`.
每种草稿模式分别通过 `--api chat` 和 `--api responses`, 使用 target checkpoint 和 OMP.
使用 `--omp` 或 `PATH` 上的可执行程序.
客户端 JSONL 和转发 HTTP 记录保留原始工具调用, 结果, 错误及后续请求.
记录代理保存原始 SSE 字节, 响应哈希, ID 及转发模型请求.
将 SSE 完成与 usage 同实际模式的服务 metric 核对.
源码读取结果和最终答案属于负载观察. 脚本单独报告原有任务审计结果.
代理任务期间保持仓库不变.

检查实际排队 / 准备时间, TTFT, 速率, 步数, 前缀复用和 verified / accepted draft.
`verified_draft_tokens` 统计已调度 candidate, 包括被拒绝 candidate, 后续 proposal 使用另一计数.
Completion 日志保留 `proposed_draft_tokens`, 作为已调度 candidate 数的旧 alias.
核对 `BENCH_CONFIG` 中的实际模式及验收脚本显式 `--speculative-mode` 选项.
原生 MTP4 和 DSpark 分别通过两个 API 重跑约束, 混合 batch / 取消和长上下文检查.
不增加 HTTP 吞吐门槛.
脚本 exit status 或 scripted worker 本身不能证明真实模型验收.
参见 [测试](testing.zh.md) 和 [验收](acceptance.zh.md).
