# 服务资源契约

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
参见 [测试](../testing.zh.md) 和 [验收](../acceptance.zh.md).
