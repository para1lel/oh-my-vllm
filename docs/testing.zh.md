# 测试与验收流程

使用 [开发指南](development.zh.md) 中的环境.
Tokenizer 或真实模型集成测试需设置 `OH_MY_VLLM_MODEL`.
有效门槛见 [需求](requirements.zh.md), 历史结果见 [验收](acceptance.zh.md).

## CPU 与 GPU 测试套件

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/check_docs.py
scripts/test.sh cpu
scripts/test.sh full
```

`scripts/test.sh` 构建测试 binary, 提供 `OH_MY_VLLM_TEST_BINARY`.
CPU 模式选择 `not gpu`, GPU visibility 为空.
Full 模式在等待空闲且绑定 UUID 的 B200 前要求模型位置.
模型未配置时, CPU tokenizer / HTTP 集成测试明确跳过.
纯校验测试无需 checkpoint, 仍然执行.
选定 pytest 参数放在 `cpu` 或 `full` 之后.

测试覆盖调度事务, 缓存所有权, 取消, RPC framing, 服务, 实际路径 FP64 参考和编译模型单元.
GPU 覆盖包括全部 6 个最大上下文用例和 metadata 变化后的 graph replay.
完整模型 drift 测试在严格算子容差之外, 使用各自记录的模型级界限.
协议回归使用 scripted worker; GPU / agentic 验收使用真实模型.
记录 warning 和跳过测试及原因.

## 精度

用独立计算的 FP64 参考比较实际舍入输入.
覆盖 prefill, decode, 边界写入, strided view, grouped MTP verification 和 speculative rollback.
BF16 算子输出使用 `atol=rtol=0.03`.
Recurrent state 的 NRMSE 至多 1%, 最大绝对误差至多为参考峰值 2%.
检查每个写入 slot 和未改动的受保护状态.
融合链包括全部舍入, scale, copy 和 reduction 行为.
参见 `tests/probe_worker.py` 和相关内核测试.

## 离线执行与数值探针

执行以下命令前构建 release binary. 按 [开发指南](development.zh.md) 设置环境和模型.

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

使用 `--num-speculative-tokens 4` 启用 MTP4.
使用 `--context-repeats 100 --prefix-hit` 跨越 784-token 边界并要求前缀命中.
使用 `--binary` 选择其他构建.
离线固定输出忽略 EOS.

`run --prompt-file PATH` 读取空白分隔的 token ID, 每行一个请求, 最多 32 个请求.
`--arrival-interval N` 按 N 个调度 step 延迟每次接纳.
`--prefix-hit` 先填充 prompt, 随后要求真实缓存命中.
全局 `--scheduler-blocks` 在 worker 报告容量内限制 Rust pool.
`bench` 子命令还提供 `--warmup` 和 `--repetitions`.

以下双请求探针每个请求使用 2048 个输入和 1024 个输出 token:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

它们要求正抢占计数, 且每个输出的开头和结尾都包含正确城市.
MTP4 必须接受 draft. 前缀模式必须有正初始命中.
检查打印文本, 作为语义 smoke test.
较小 pool 本身不能证明发生抢占.

诊断实际路径数值时, 使用可执行探针作为 worker interpreter:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" python scripts/smoke-text.py --socket /tmp/probe-text.ipc --max-tokens 32
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" OH_MY_VLLM_PROBE_MTP=1 python scripts/smoke-text.py --socket /tmp/probe-mtp.ipc --max-tokens 32 --num-speculative-tokens 4
scripts/with-gpu.sh scripts/with-env.sh python scripts/probe-independent-model.py --max-tokens 32
```

Worker probe 测量真实 GQA, GDN prefill / recurrent verification 和 MTP attention 调用, 保留生产内核.
它基于实际舍入输入, 通过 CPU FP64 计算独立参考并比较结果.
Eager 模式关闭 capture, 让诊断复制观察真实请求.
直接模型探针提供不经过 Rust 调度的短 eager 模型诊断.
性能验收时关闭这些探针和 eager 模式.

## 验收用 release binary

阶段时延采集器 读取仓库 `target/release/oh-my-vllm-zmq-worker`.
显式使用此目录, 防止其他构建位置或旧 binary 改变实测候选:

```bash
export CARGO_TARGET_DIR="$PWD/target"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

框架, 上下文或服务验收前使用此构建.
将 `EVIDENCE_DIR` 设为外部可写目录, 保存完整原始记录和日志.

## 正式算子门槛

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --output "$EVIDENCE_DIR/operators.json"
```

将 `EVIDENCE_DIR` 设置为仓库外的可写证据目录.
测试框架使用 `development/kernels/cases.py` 中的静态用例.

使用 `--operations NAME` 选择算子族.
为每个所选用例 ID 给出 `--case-id ID`.
少于 317 个用例 ID 的选择报告 `coverage.complete=false`, 并保持完整的逐用例协议.
矩阵验收前, 全部 317 个用例 ID 必须通过.

计时前校验完整输出和实际写入缓存.
每个用例需要 3 轮, 每轮 20 个交错配对, CUDA 每轮中位数更快.
节省时间的 hierarchical-bootstrap 单侧 95% 下界必须为正.
每个计时样本使用 100 次 graph repetition.
用例和轮次之间恢复持久状态.
Normalization, Q / K, recurrent 和 convolution 校验要求 1 次 fast, 0 次 generic host dispatch.
不以 graph replay 次数代替 host dispatch 次数.
来源记录和 profiling 分离见 [内核开发](kernels.zh.md).

## 框架性能

构建当前 release 二进制, 为每次尝试指定新的外部输出路径.
从 [要求](requirements.zh.md#req-perf-001-阶段时延) 中选择十五行之一.

```bash
scripts/with-gpu.sh scripts/with-env.sh python -m benchmarks.framework --mode ordinary --batch-size 1 --input-len 32768 --output "$EVIDENCE_DIR/ordinary-32768-1.json"
```

使用 `ordinary`, `mtp4`, `prefix` 和 `dspark` 模式, 输入 32768, batch 为 1/2/4.
普通模式另测输入 131072, batch 相同.
DSpark 通过 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model` 指定 checkpoint.
输出数量为 4096. 每次尝试执行两次完整预热和五次正式测量.
前缀行必须为每个请求命中 32144 个 token.
其他行在每次重复前重置前缀复用.

采集器记录每个请求的提交, 首 token, 末 token 和 step 边界.
从各请求自己的边界独立计算 prefill 和 decode.
每次重复分别取各阶段最长的请求区间.
清理在末 token 之后执行, 使用独立计时字段.

Worker 缓存有效形状, 状态访问, 草稿运算和编译运算身份.
在正式请求执行后写出 trace.
正式分析使用规范成本模型和官方峰值.
未识别的运算阻止验收.
GPU event 区间是用于诊断的 stream 区间, 可能包含主机提交间隙.

每个阶段的墙钟中位数最多为其理论下界中位数的三倍.
墙钟时延的相对极差最多为 10%.
保留完整失败集和中断尝试. 调查后重新执行整组测量.

每次尝试绑定源码, 二进制, checkpoint 字节, CUDA module, 硬件, 容量和运行时包.
采集期间保持源码不变.
编译 / cache 审计必须显示稳定执行.
现有日志无法排除全部静默内存编译.
原始日志和 trace 放在外部目录.
审阅使用可移植摘要. 正式验证使用完整原始记录.

未提交源码的实验使用 `--diagnostic`.
诊断记录不提供正式验收结论.
Profiling 与正式计时独立执行.

## DSpark 比较

将 `OH_MY_VLLM_DRAFT_MODEL` 设置为已校验的本地 checkpoint, 构建当前 release binary.
完整尝试保存在仓库外, 每次使用新的输出路径:

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/speculative.py --binary target/release/oh-my-vllm-zmq-worker --raw-dir "$EVIDENCE_DIR/dspark-attempts" --output "$EVIDENCE_DIR/dspark-comparison.json" --portable-output "$EVIDENCE_DIR/dspark-comparison-portable.json"
scripts/with-env.sh python benchmarks/speculative.py --input "$EVIDENCE_DIR/dspark-comparison.json" --output "$EVIDENCE_DIR/dspark-recheck.json"
```

采集器为每个 batch 启动一个比较 worker, 共享 target 权重和物理 target 缓存.
每种模式在预热前加载草稿模型, 每次尝试重置前缀复用.
使用 batch1/2/4, 输入 32768, 输出 4096, 合成 ID, greedy 采样, 忽略 EOS.
框架批吞吐包括注册, prefill, 传输和清理.

每个 batch 执行 3 轮.
每轮每种模式执行 2 次完整预热和 5 个交替顺序的测量配对.
报告每轮每种模式的吞吐中位数及波动.
记录 DSpark 中位数与原生 MTP4 的差值, 以及吞吐增益的 hierarchical paired-bootstrap 单侧 95% 置信下界.
报告 TTFT 及其波动.
该比较报告 DSpark 吞吐, 波动和置信界限.
不增加 DSpark 吞吐或 TTFT 门槛.

原始产物保留每次尝试, 日志哈希, 源码 / binary / Python 身份, package, hardware, 容量和 checkpoint 文件哈希.
加载的草稿配置和权重哈希必须与指定 checkpoint 字节一致.
记录包括 compile / capture 审计, GPU 显存峰值和实际验证 / 接受的草稿数.
配对审计检查全部 42 个 phase marker 和每个测量区间.
后续预热使用自己的 compilation / capture 区间.
日志和最后记录的 cache mtime 展示观察到的活动, 无法排除全部静默的内存内编译.

`--dspark-confidence-threshold` 选择固定或较短 proposal, 与实际 worker 配置一同记录.
初始设置为 `0.2`. 使用 `0.0` 保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.
通过已调度 / 接受草稿数及实测性能比较各设置.
接受率为测量配对内接受草稿总数除以已调度草稿总数.

使用 `bench --prompt-file PATH` 对与正式输入不同的文本或工具历史开展实验.
文件每行对应一个请求, token ID 以空白字符分隔.
行数必须等于 `--batch-size`, 每行长度必须等于 `--input-len`.
命令在加载模型或 GPU 前检查这些约束.
实验记录保留文本来源, token 哈希, 模板和采样设置.

正式比较使用的 `spec-bench` 保持固定合成输入.
它不接受 `--prompt-file`, 阈值实验与该比较使用不同输入.

实际验证草稿为已调度 candidate, 返回的下一步 proposal 使用独立诊断计数.
Portable 输出是派生摘要, 不能替代比较所需的原始证据.

源码身份及测量范围见 [验收索引](acceptance.zh.md).

## 上下文与服务验收

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/context_boundary.py --binary target/release/oh-my-vllm-zmq-worker --raw-dir "$EVIDENCE_DIR/boundary-logs" --output "$EVIDENCE_DIR/boundary.json"
```

6 行覆盖输入 258048, 输出 4096, ordinary/MTP4, batch1/2/4.
要求完整输出, 无 OOM / 抢占, 开始 / 结束源码匹配和 worker 清理.
记录 allocated / reserved GPU 显存峰值和草稿计数.
原始边界日志必须放在仓库外.

新增 3 个 DSpark 边界用例, 保留普通 / MTP4:

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/context_boundary.py --binary target/release/oh-my-vllm-zmq-worker --modes ordinary mtp4 dspark --raw-dir "$EVIDENCE_DIR/boundary-dspark-logs" --output "$EVIDENCE_DIR/boundary-dspark.json"
```

DSpark 需要 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model`.
这 9 行沿用完整输出, 无 OOM, 无抢占及显存记录合同.

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
按 [服务合同](serving.zh.md#验证) 分别完成两种 API 的真实 OMP 任务.
要求实际读文件和工具结果, 不只看脚本成功.

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
保留客户端 JSONL, 转发请求, response ID 和工具结果哈希.

两次源码读取必须无错误地完成, 结果在最后一条答案前进入后续模型请求.
最后一条答案必须引用源码文件并使用读取结果.
自动检查之外单独复核答案语义.
这些测试完成后停止 DSpark 服务及其 worker.

服务脚本在请求前独占输出文件或目录, 已存在的结果路径会报错.
适用合同通过前保留 `passed=false`.
每个已接收 HTTP 响应在内容或日志检查前保存.
失败时保留已收响应及明确错误记录.
Lifecycle stream 保留断连前已读取的 SSE frame.

## 教程与清理

使用 [code-journey](../code-journey/README.zh.md#验证) 中的检查.
教程修改后检查桌面 / 移动截图和全部 13 条路线.

所有 GPU 程序经过 `scripts/with-gpu.sh`.
完成或失败后停止全部所属进程及后代.
确认 GPU 进程退出, listener 释放和 IPC 清理.
只保留用户明确要求持续运行的服务; 本机信息记入 `LOCAL.md`.
