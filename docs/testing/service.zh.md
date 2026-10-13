# 上下文与服务检查

## 上下文与服务验收

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/context_boundary.py --binary target/release/oh-my-vllm-zmq-worker --raw-dir "$EVIDENCE_DIR/boundary-logs" --output "$EVIDENCE_DIR/boundary.json"
```

6 行覆盖输入 258048, 输出 4096, ordinary/MTP4, batch1/2/4.
要求完整输出, 无 OOM / 抢占, 开始 / 结束源码匹配和 worker 清理.
记录 allocated / reserved GPU 显存峰值和草稿计数.
原始边界日志必须放在仓库外.

在一个终端使用 release binary 和独立 IPC 路径启动专用 MTP4 服务.
使用 Serving DEBUG 日志提供混批与取消证据:

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug target/release/oh-my-vllm-zmq-worker --socket /tmp/service-acceptance.ipc --max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128 --num-speculative-tokens 4 serve > "$EVIDENCE_DIR/server.log" 2>&1
```

服务就绪后, 从另一个终端执行以下命令.
将 OMP executable 安装到 `PATH`, 或通过 `--omp` 指定路径.
全部客户端使用相同的显式 URL. 长上下文工具的默认端口不同.

```bash
export SERVICE_URL="http://127.0.0.1:8000/v1"
scripts/with-env.sh python scripts/serving-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/constraints"
scripts/with-env.sh python scripts/serving-lifecycle.py --api chat --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle-chat.json"
scripts/with-env.sh python scripts/serving-lifecycle.py --api responses --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle-responses.json"
scripts/with-env.sh python scripts/long-context-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/long-context.json"
scripts/with-env.sh python scripts/agentic-acceptance.py --api chat --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/agentic-chat"
scripts/with-env.sh python scripts/agentic-acceptance.py --api responses --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/agentic-responses"
```

Lifecycle 工具必须在所提供日志中找到混合请求共享 batch, 以及 worker 取消 / 释放证据.
检查服务日志中的正 MTP proposal 和 accepted draft 计数.
这些测试完成后立即停止服务及其 worker.

长上下文门槛检查 strict JSON 内容, 输入超过 131072, 首次请求之后的前缀复用和逐请求正 MTP proposal.
按 [服务合同](../serving/resources.zh.md#验证) 分别通过两种 API 使用 OMP 和目标 checkpoint.
独立于任务结果检查原始模型 stream, usage 和服务 metric.

使用不同的 DSpark worker 再次执行服务门槛:

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug target/release/oh-my-vllm-zmq-worker --socket /tmp/dspark-service-acceptance.ipc --max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128 --speculative-mode dspark serve > "$EVIDENCE_DIR/dspark-server.log" 2>&1
```

每个验收脚本指定 `--speculative-mode dspark` 及对应 `--server-log`.
每种模式和 API 使用独立证据路径.
Lifecycle 检查分别使用 `--api chat` 和 `--api responses`.
约束及长上下文脚本已经测试两种 API.
每个完成 response ID 都必须与服务日志中的活跃模式和实际验证 / 接受计数匹配.

OMP 脚本通过临时记录代理转发真实客户端 HTTP, 请求 body 保持不变.
保留客户端 JSONL, 转发请求, 原始 SSE 字节, response ID 和工具结果哈希.

保留原有源码读取和答案检查, 作为单独的负载审计.
任务错误和未完成答案不决定框架验收. 没有模型请求时不能通过.
这些测试完成后停止 DSpark 服务及其 worker.

服务脚本在请求前独占输出文件或目录, 已存在的结果路径会报错.
适用合同通过前保留 `passed=false`.
每个已接收 HTTP 响应在内容或日志检查前保存.
失败时保留已收响应及明确错误记录.
Lifecycle stream 保留断连前已读取的 SSE frame.

原生 first-event 和 idle deadline 均为 900 秒. Collector turn deadline 为 7200 秒.
Oneoff 客户端任务时限为 30 分钟, 主机时限为 1860 秒. 代理读取 deadline 为 900 秒.

Recovery 请求和阶段 deadline 分别为 900 和 3600 秒.
用户免除了仅针对超时参数的额外测试. Collector 行为变化仍需有意义的回归检查.

## 真实 OOM 下的 worker 恢复

故障注入与正常 soak 流量使用不同记录. 日常命令也跳过这项真实模型 collector.
在空闲 B200 上显式运行:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/serving-recovery.py --mode mtp --output-dir "$EVIDENCE_DIR/recovery-mtp"
scripts/with-gpu.sh scripts/with-env.sh python scripts/serving-recovery.py --mode dspark --output-dir "$EVIDENCE_DIR/recovery-dspark"
```

测试在外部建立 worker launcher, 使用 `tests/fixtures/worker_faults.py`.
生产路径不导入故障 injector. 每个 marker 在 worker 重建前后只允许一次显式故障.
在采样, 接受状态提交, 草稿生成和 graph capture 后测试真实 CUDA allocator OOM.
HTTP 继续存活, stream 明确报告失败, 旧前缀失效, 单次 worker 替换后验证 Chat/Responses 正常服务.
确认恢复后的 cold 请求不会命中旧 KV identity, 然后验证新前缀复用.
Collector 管理并停止服务, worker 进程组及临时 IPC endpoint.
