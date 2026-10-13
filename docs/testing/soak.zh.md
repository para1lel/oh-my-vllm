# 长时服务检查

## 长时 agentic 服务

日常命令跳过这项重型测试. 每种模式使用专用服务, 最大模型长度为 `262144`.
使用新的外部目录. 采集全程保持源码和 release binary 不变.
一并构建服务和 CPU 请求 normalizer:

```bash
scripts/with-env.sh cargo build --release --bins
scripts/with-env.sh python scripts/agentic-soak.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/mtp-soak"
scripts/with-env.sh python scripts/agentic-soak.py --mode dspark --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/dspark-server.log" --output-dir "$EVIDENCE_DIR/dspark-soak"
```

设置 `OH_MY_VLLM_MODEL` 或传入 `--model`. 将 OMP 安装到 `PATH`, 或通过 `--omp` 指定.
每个命令连接对应模式的服务. 采集结束后停止该服务和 worker.
使用固定的 OMP `18.10.3` executable, 并启用 Docker 的 `rootless` 模式.
Collector 构建包含 OMP, Python, Bash 和证书的临时镜像.
将 `--sandbox-host` 或 `OH_MY_VLLM_SANDBOX_HOST` 设为访问主机 loopback 的容器网关.
主机特定选择保存在忽略的 `LOCAL.md`.
Collector 管理 4 个 sandbox 容器, 持久 OMP 进程和临时记录代理.
每个容器只挂载对应 agent 的新建外部目录. 不挂载推理源码和 Docker socket.
容器具有独立 PID namespace, 不访问 GPU 设备.

每个进程在连续原生 goal 中保留 session ID 和历史.
2 个会话使用 Chat, 2 个使用 Responses. 共享的 4-slot 上限也包含摘要请求.

每种推理模式必须在 4 个持久请求负载下运行至少 `3600` 秒. 不设请求数下限.
服务就绪后初始化客户端, 在它们发出首次请求前开始计时.
首次编译, 图捕获和后续新形状的开销均计入测量.

此采集器对应的服务使用 `--worker-step-timeout-seconds 600` 和 `--request-timeout-seconds 3600`.
从 diagnostics 记录这些上限. 这些期限允许冷编译完成, 不另行预热.

中途不排空, 不插入故障 probe. 一小时结束后暂停续行, 并发送原生 RPC `abort`.
排空模型请求, 记录停止动作.

任务和摘要使用 OMP `max`, 映射为服务的 `xhigh`.
此工作负载使用生产采样和思考预算默认值, 不另设启动覆盖值.
从 diagnostics 记录有效服务值, 从请求 metric 记录添加服务默认值后的采样字段.
OMP 在两种 OpenAI API 中省略这些字段. 服务默认值也应用于其摘要请求.
模型 context window 为 `262144`, 任务输出上限为 `32768`, 包含 reasoning token.
本地 `qwen3` tokenizer 元数据与 provider usage 让 OMP 了解会话大小.
启用 soft 自动压缩, threshold 为 `196608`, reserve 为 `65536`, 保留最近 `20000` 个 token.
OMP 自行选择压缩时机, 包括工具循环中途. 控制器不发送手动 compact 命令.
服务检查完整模板输入与请求输出之和是否在总容量内.
CPU guard 重复生产 normalization 与模板路径, 提供独立容量证据.
将其计数与 SSE usage 核对. SDK 和 journal 比较结果记入单独的负载审计.

OMP 的 settled `contextUsage` 使用 provider-input 锚点加上之后的本地变化.
压缩判定还考虑生成输出和已保存历史的估算.
两种记录均保留. 界面百分比不能代表完整的压缩判定.
每次观察到压缩时, 检查会话身份延续并报告后续请求. 摘要内容属于负载观察.
一小时内不强制发生自然压缩. 未覆盖时明确报告.
参见固定版本的 [OMP 压缩流程](https://github.com/can1357/oh-my-pi/blob/v18.10.3/docs/compaction.md).

每条 lane 在空 workspace 中获得嵌入式数据库任务, 使用 Python 标准库工具.
任务包含 SQL, transaction, index, 持久存储, 崩溃恢复, 独立测试与实测 benchmark.
控制器使用原生 RPC `goal create`, 为 `rpc` 启用自动续行.
不设 token budget. Goal 完成, 暂停或停止续行后选择预先准备的下一项.
后续 goal 检查故障, 并发, SQL 结果, 存储变更, 性能及剩余要求.
同一进程, session ID, 历史和工作目录继续使用. Goal 转换保留原始原因与时间戳.
Goal 命令 ID 只用于主机侧关联, 不代表原生 `prompt_result`.
将源码读取, 工具错误, 任务进展和 goal 转换记录为负载观察, 不用于决定框架验收.
保留增量 journal, 原生会话及完整工具输出日志, 以及完整 RPC/wire 记录.

一小时后暂停续行, 并 abort 在途 Agent 工作. 尾部耗时与采集窗口分开报告.
独立 sandbox 测试命令运行前后均对生成源码计算哈希. 记录失败, 不宣称数据库已完成.
随后删除自有容器, 修改文件, 安装工具, 缓存及临时镜像.
在 sandbox 外部的独立证据目录保留原始日志, 哈希 manifest, 摘要和资源图. 失败和取消使用同样的清理流程.
可选 `--workload audit --diagnostic` 路径保留旧源码审计 fixture, 不能提供正式验收.

独立于原生任务结果核对原始请求覆盖, 已完成 SSE 结束记录, usage 和服务 metric.
取消请求保留原始协议及传输观察. 不根据共用错误消息推断取消原因.

下列 SDK 与 journal 检查提供单独的原生负载诊断, 其结果不决定框架验收.
仅在同 ID 的协议, SDK 和服务完成证明齐全时, 将结束后的 `ConnectionResetError` 或 `BrokenPipeError` 保留为诊断信息.
SDK 与 SSE stop reason 必须相符. `length` 结束的输出数必须等于请求预算.
并行摘要请求必须唯一匹配 usage 和生命周期. 证据有歧义时验证失败.

OMP `18.10.3` 不将内部摘要 usage 写入会话 journal.
Collector 为每个客户端建立一个自有 loopback OTLP endpoint, 采集原始 SDK 结果 trace.
它保持 provider dispatch, 模型 API, 请求正文和自动压缩不变.
原生 trace 提供响应 ID, 会话 ID, token usage, finish reason 和摘要用途.

API 绑定来自带哈希的隔离配置及已记录的运行时模型状态. 原生 trace 不提供该字段.

每个 wire 请求必须唯一匹配一个 SDK 结果 trace. 记录缺失, 改变或有歧义时验证失败.
活动工作结束后, collector 等待已完成 SDK 导出, 再关闭 RPC 和 trace listener.
原始 protobuf 记录设有大小上限并保存哈希. 离线验证检查 collector 错误及关闭记录.

Exporter 可再次发送同一批次. Collector 保留这些原始记录.
具有相同 response ID 的两个 SDK 完成记录使验证失败.
只有 `message_end` 提供 RPC SDK 完成记录. 正常 `turn_end` 可以携带同一消息.

安装固定版本 OMP executable 后, 执行 CPU 采集检查:

```bash
scripts/with-env.sh python -m pytest tests/test_agentic_sdk_trace.py tests/test_agentic_sdk_integration.py tests/test_agentic_goal_integration.py tests/test_agentic_goals.py tests/test_host_allocator.py
```

集成 fixture 提供来自固定测试数据的 API 响应. 它检查完整请求 JSON 相等及全部三种自动压缩用途.
它不测试目标模型推理或目标模型的 tokenizer 计数.

采集期间检查资源的数量和字节上限. 最后排空后检查 Rust/Python 请求状态为空.
各逻辑池此时有 `blocks - 1` 个空闲块, 包含可驱逐复用的缓存块.
Block `0` 保持分配. HTTP 和 worker 必须保持原有进程出生身份.
结合主机 RSS, GPU allocated / reserved 内存, 图/编译缓存占用, 活动请求和输入长度分析.
会话大小变化可以影响有界缓存. 独立审查必须检查增长, 资源上限和清理.
采样上限不能证明内存平台, 完整分配归因或测量一小时之外的稳定性.

设置 `OH_MY_VLLM_HOST_ALLOCATOR_DIAGNOSTICS=1`, 在现有诊断间隔采集可选 glibc allocator 字段.
每条记录包含 `available` 值和调用耗时. 可用测量包含 glibc 版本和原始 `mallinfo2` 字段.
该调用使用 allocator 锁, 并检查空闲存储. 开销可能随 allocator 状态改变.

`uordblks + hblkhd` 表示 allocator 已用或保留的存储. `fordblks` 表示该 allocator 中的空闲存储.
线程缓存, 直接映射和其他 allocator 限制测量范围. 这些值不统计全部活跃对象, 也不替代 RSS.
该开关不改变 allocator 设置, 不启动额外资源释放. 不支持的主机明确报告测量不可用.

Collector 将 `operational_passed` 和 `awaiting_resource_review` 与正式验收区分记录.

GPU 进程查询前保留失败 worker 的 diagnostics.
GPU 查询失败时, 保留错误与重新读取的 worker 状态.
正常负载期间的 epoch 变化或 worker 重启均判为失败.
最终检查和证据哈希计算前必须结束资源采样. 收尾时到达的采样失败也必须使采集失败.

记录其他 GPU 进程和设备 HBM 占用. 请求和资源检查通过时, 轻度争用可继续.
15 项阶段负载仍要求 GPU 独占.
全部完成的任务和实际摘要请求按用途绘图, 不添加 Chat/Responses 颜色图例.
横轴为框架开始处理的时刻, 纵轴为输出 token 数除以处理墙钟时间.
输出数包含 reasoning token. 排队时间, 取消和失败分别报告.
不设性能门槛. `--diagnostic` 允许缩短的 smoke test, 不能提供正式验收.
