# 长时服务检查

## 长时 agentic 服务

日常命令跳过这项重型测试. 每种模式使用专用服务, 最大模型长度为 `262144`.
使用新的外部目录. 采集全程保持源码和 release binary 不变.
一并构建服务和 CPU 请求 normalizer:

```bash
scripts/with-env.sh cargo build --release --bins
scripts/with-env.sh python scripts/agentic-soak.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --corpus "$CORPUS_DIR" --output-dir "$EVIDENCE_DIR/mtp-soak"
scripts/with-env.sh python scripts/agentic-soak.py --mode dspark --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/dspark-server.log" --corpus "$CORPUS_DIR" --output-dir "$EVIDENCE_DIR/dspark-soak"
```

设置 `OH_MY_VLLM_MODEL` 或传入 `--model`. 将 OMP 安装到 `PATH`, 或通过 `--omp` 指定.
将 `CORPUS_DIR` 设为保存已校验书籍及其原始 manifest 的外部目录.
每个命令连接对应模式的服务. 采集结束后停止该服务和 worker.
使用[专用进程归属](process-ownership.zh.md), 为服务提供独立的进程范围.
使用固定的 OMP `18.10.3` executable, 并启用 Docker 的 `rootless` 模式.
Collector 构建包含 OMP, Python, Bash, 证书和 lane forwarder 的临时镜像.
主机特定选择保存在忽略的 `LOCAL.md`.
Collector 管理 4 个 sandbox 容器, 持久 OMP 进程和临时记录代理.
每个 lane 使用 `--network none` 运行, 不访问 GPU 设备, 不挂载推理源码, weight 和 Docker socket.
容器只挂载对应的新建 lane 目录和一个自有 socket 目录.
容器具有独立 PID namespace.

Collector 为每条 lane 管理一个用于模型调用的 AF_UNIX bridge 和一个用于 telemetry 的 bridge.
Guest loopback endpoint 将原始请求和响应 body 转发到这些 socket.
每个 bridge 上游必须是本地明文 HTTP collector origin, 因此不存在网络回退.
Body, 连接, listener 和关闭过程都有上限, 所有进程退出后移除 socket.
模型 socket 只承载生成路由; telemetry 使用自己的 socket.
fake upstream 检查在无外部接口时证明模型 SSE 和 protobuf export 的 framing.

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

每条 lane 在新建空 workspace 中翻译同一本完整英文书籍.
`--corpus` 指定保存 6 个卷及其原始 manifest 的外部目录.
Manifest 固定书籍合计值以及每卷的编号, 大小和完整 SHA256 digest.
校验在 lane 启动前检查固定的 manifest digest 与每个已暂存输入.

```json
{
  "title": "<book>",
  "author": "<author>",
  "source": "Project Gutenberg, <volumes>",
  "derivation": "<normalization note>",
  "book_file": "book.txt",
  "book_sha256": "<64 hex>",
  "book_bytes": 1,
  "book_english_words": 1,
  "downloaded_unix_s": 1.0,
  "volumes": [{"volume": 1, "gutenberg_id": 731, "sha256": "<64 hex>", "bytes": 1}]
}
```

已暂存的副本只读. Lane 工具 guard 允许 read, grep, write 和 goal 查询, 并拒绝 edit.
写入 corpus 与 specification 同样被拒绝, 因为固定版本 OMP 的 edit 调用携带 `{i, input}` hashline 段落及 REM 与 MV 操作.
译文与进度笔记保持可写.

控制器使用原生 RPC `goal create`, 为 `rpc` 启用自动续行.
不设 token budget. Goal 完成, 暂停或停止续行后选择预先准备的下一项.
下一项 goal 从 `notes/progress.md` 继续. 不设一小时内完成翻译的要求.
同一进程, session ID, 历史和工作目录继续使用. Goal 转换保留原始原因与时间戳.
Goal 命令 ID 只用于主机侧关联, 不代表原生 `prompt_result`.
将源码读取, 工具错误, 翻译进展和 goal 转换记录为负载观察, 不用于决定框架验收.
保留增量 journal, 原生会话及完整工具输出日志, 以及完整 RPC/wire 记录.

一小时后暂停续行, 并 abort 在途 Agent 工作. 尾部耗时与采集窗口分开报告.
客户端中止发生在服务端完成期间时, 使用[截止收尾证据规则](deadline-tail.zh.md).
采集前后均对已暂存 corpus 和 specification 计算哈希, 输入变化即失败.
记录翻译进展, 不宣称翻译质量或完成度.
随后删除自有容器, 修改文件, 安装工具, 缓存及临时镜像.
在 sandbox 外部的独立证据目录保留原始日志, 哈希 manifest, 摘要和资源图. 失败和取消使用同样的清理流程.
`--workload goal` 或 `--workload audit` 诊断 fixture 不能提供正式验收.

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
scripts/with-env.sh python -m pytest tests/tooling/test_agentic_sdk_trace.py tests/integration/test_agentic_sdk_integration.py tests/integration/test_agentic_goal_integration.py tests/integration/test_agentic_translation_integration.py tests/integration/test_agentic_deadline_cancellation.py tests/tooling/test_agentic_goals.py tests/tooling/test_agentic_lane_transport.py tests/tooling/test_agentic_corpus.py tests/tooling/test_agentic_translation.py tests/tooling/test_agentic_soak_inputs.py tests/unit/test_host_allocator.py
```

集成 fixture 提供来自固定测试数据的 API 响应. 它检查完整请求 JSON 相等及全部三种自动压缩用途.
它不测试目标模型推理或目标模型的 tokenizer 计数.

Sandbox fixture 带有 `omp` marker, 只使用固定 CPU API 数据.
运行前需有 rootless Docker 和固定版本 OMP executable.
它们证明密封容器没有网络接口, 并删除自有容器和自有镜像.

Goal lane 保留诊断用数据库 fixture.
Translation lane 暂存已接受 corpus, 并在容器内部走通挂载 socket.
它读取一卷, guard 拒绝一次 corpus 写入和两种 hashline edit 负载.
产出一份译文, 且原生 SDK export 保持完整.

截止 fixture 持续发送未结束的响应, 使原生 goal 到达截止时间时请求仍在途.
随后绑定已记录的 pause 与 abort 覆盖, 被中止的原生结果, 以及未改变的请求字节.
它还绑定规范的下游断开, 以及每个挂载 hop 自身失败的那一次尝试.
该 fixture 走通已接受的 collector 路径: 框架审计产生截止证明, runner 为每个原始请求推导一个尝试界, 传输审计再绑定各 hop.
Chat 与 Responses 分别作为独立案例运行, 服务端取消的结果与已完成收尾携带同样的证明.

传输检查通过两段 socket hop 驱动 fake 模型服务和 fake OTLP receiver.
它证明 SSE 投递顺序和字节一致, protobuf export framing, body 上限及 loopback-only listener.
上述 Docker lane 在密封容器内走通同样的 hop, 因此挂载 socket 路径已被覆盖.
两者都不证明目标模型推理或目标模型 endpoint.

Hop 审计要求每个已配置 hop 具备闭合记录.
Journal 计数, journal 错误, handler 释放, socket 移除和 guest endpoint 释放必须一致.
每次携带 body 摘要的尝试恰好覆盖一个原始请求或 export, 因此不同原始请求的相同 body 与重复 export 各自独立.
截止界只接受方向与错误一致的失败尝试, 且其完成时间必须落在已记录的 abort 动作与客户端断开之间.
未被任何 hop 消耗的界会被报告, 而从未转发过已记录请求的 hop 会失败.
仅模型和零请求 lane 仍然有效.

Corpus 与负载检查使用合成书籍. 它们证明 manifest 与 digest 拒绝, 暂存, objective 续行及只读 guard.
它们不证明保留书籍的实际内容或翻译质量. 独立审查必须绑定真实 corpus manifest.

十五行框架矩阵全部通过后才启动真实一小时 soak. 翻译完成度和质量均未被证明.

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

使用暗色背景, 以固定 0–1 渐变表示 `prefix_hit_tokens/input_tokens`.
任务与 compaction 请求仅通过不同 marker 区分.
横轴为框架开始处理的时刻, 纵轴为输出 token 数除以处理墙钟时间.
输出数包含 reasoning token. 排队时间, 取消和失败分别报告.
不设性能门槛. `--diagnostic` 允许缩短的 smoke test, 不能提供正式验收.
