# 服务契约

## 服务需求

| ID | 合同 |
|---|---|
| REQ-SERVE-001 | 提供文本 Chat Completions 和 Responses, 流式响应, 工具, 历史, usage, 模型发现, 查询和删除. |
| REQ-SERVE-002 | 提供 off/low/medium/high/xhigh 思考; 默认 medium, high 映射为 xhigh, none 映射为 off. |
| REQ-SERVE-010 | 在普通, MTP4 和 DSpark 模式的 target 采样前应用显式思考 token 预算. Effort 与数值预算和总量限制保持独立. |
| REQ-SERVE-003 | MTP4 和 DSpark 分别通过两种 API 使用 oh-my-pi. 验证模型请求, 响应及后续输入中回传的工具结果. |
| REQ-SERVE-004 | 采样前施加 JSON / Schema / strict-tool mask, 包括每种模式的草稿和 bonus token; 拒绝不支持的 schema. |
| REQ-SERVE-005 | 检查排队, 准备, TTFT, 输出速率, 缓存和草稿计数, 不增加 HTTP 吞吐门槛. |
| REQ-SERVE-011 | 生产和测试服务在普通, MTP4, DSpark, 两种 API 及自动压缩中使用公共采样默认值. 显式值优先. |
| REQ-SERVE-009 | 所有服务模式应用有界的生成 token 循环保护. 保留 grammar 续接, 草稿前缀正确性和显式关闭选项. |

[服务合同](../serving.zh.md) 定义支持的参数, XML 限制, 存储限制和错误.
单独使用 scripted worker 不能证明真实模型验收.

思考 grammar 与预算计数必须使用同一个结束 token 身份.
启用工具时, 约束必须从输出 parser 识别的起始标记开始.

模型提示保留原始工具描述和参数 schema.
工具未显式设置 `strict=true` 时, 通过支持的 XML schema 的独立副本接受描述性关键词.
OMP 检查描述中的参数限制, 并把错误返回给模型. 显式严格 schema 和推理计算保持不变.

每种草稿模式都测试两种 API, 约束, 取消, 混合 batch 和长上下文前缀复用.
OMP 证据保留模型请求, 响应, 工具调用, 返回结果及后续请求.
任务进展, 工具错误, goal 状态和最终答案属于负载观察, 不决定框架验收.
通过 response ID 与服务日志核对实际模式及已验证草稿.

## REQ-SERVE-006: 资源上限与缓存正确性

请求准入, 队列, 历史副本, 输出缓冲及已完成 stream 的 drain 必须遵守明确的数量和字节上限.
复制历史或准备请求前先保留容量.
请求 deadline 从准入时开始, 包含排队时间.
暂时过载返回可重试错误. 超过支持容量的请求明确拒绝.

失败必须明确报告, 包括开始流式输出后的失败.

只有经过验证的 GPU 结果提交对应状态后, 才发布 prefix identity.
LRU 驱逐必须保护仍被引用的 block. 失败或取消后清除未提交的 identity.
逻辑 KV 驱逐复用固定 GPU arena, 不释放该 arena 的 HBM 存储.
Graph entry, 编译器 metadata, response record 及 shutdown 引用必须遵守各自的资源上限.

## REQ-SERVE-007: Worker 恢复

GPU OOM 或 worker 失败后, HTTP 进程继续存活.
Device state 无法安全复用时, 让受影响请求失败并重建 worker.
重新准入前清除该 worker 的全部逻辑 cache identity.
恢复期间报告暂时不可用. 恢复尝试与 shutdown 等待均遵守既定上限.
已经提供 stream 输出的请求不自动重放.

测试 OOM, 部分状态修改, 共享前缀及恢复后的再次服务.

## REQ-SERVE-008: 长时 agentic 服务

分别运行 MTP4 和 DSpark, 每种模式至少一小时. 每种模式由 4 个固定 oh-my-pi 会话提供请求负载.
这一小时验证推理服务可用性, 服务合同, 资源上限及清理.
连续追问和自动压缩期间保留进程, session ID 和会话历史.
使用 2 个 Chat 会话和 2 个 Responses 会话. 模型请求最多 4 个并行, 包括压缩请求.
报告实际请求数, 不设请求数下限.

服务就绪且客户端初始化后开始计时. 首次请求的编译和图捕获计入测试.
任务和压缩均使用最高 reasoning effort. OMP 的 `max` 映射为服务的 `xhigh`.
模型 context window 配置为 `262144`. 输出预算包含 reasoning token.
OMP 根据运行时 provider usage 和自身的会话估算决定自动压缩时机.
独立 token 检查验证输入和输出容量, 不决定压缩时机.
报告自然压缩的实际覆盖及同一会话中的后续请求. 摘要内容和任务结果属于负载观察.

每个会话在独立 sandbox 中新建数据库项目, 不复制推理源码.
使用 OMP 原生 goal 和自动续行. 预先准备实现, 故障, 并发与差分测试等连续 goal.
当前 goal 完成, 暂停或停止续行后提交下一项. 这些转换不导致验收失败.
一小时内保留同一进程, session ID, 历史和工作目录.
随后暂停续行, 通过原生 RPC abort 在途 Agent 工作, 并排空模型请求.
记录停止动作, 单独报告尾部耗时.
完成, 失败或取消后删除 sandbox 容器, 修改文件, 安装工具及缓存.
原始测试日志和源码哈希保存在独立证据目录.
采集期间检查资源上限, 最后排空后检查请求状态释放.
保留完整主机 RSS 和 GPU allocated / reserved 内存曲线. 结合上下文长度, 活动请求和缓存占用解释增长.
独立内存审查接受这些观察后, 才能将结果发布为通过的 soak test.
框架不得发生意外 OOM, worker 重启或模型执行失败.
客户端取消和已配置的请求 deadline 单独报告, 不表示请求成功完成.
取消, 慢客户端, 过载和故障测试单独执行.

记录其他 GPU 进程. 请求和资源检查通过时继续采集.
阶段时延测试要求 GPU 上没有其他进程.

README 每种模式发布一张散点图. 包含任务请求和实际压缩请求, 并标明用途.
横轴为框架开始准备请求的时刻.
纵轴为输出 token 数除以从开始处理到生成完成的墙钟时间.
输出包含 reasoning token. 排队时间, 取消请求和失败请求分别报告.
图例不按 API 区分颜色. 此测试不设性能门槛, 日常命令默认跳过.
推理路径修改后重测受影响的阶段负载.
