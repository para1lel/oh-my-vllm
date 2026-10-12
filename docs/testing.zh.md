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
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/test.sh cpu
scripts/test.sh full
```

`scripts/test.sh` 构建测试 binary, 提供 `OH_MY_VLLM_TEST_BINARY`.
CPU 模式选择 `not gpu`, GPU visibility 为空.
Full 模式在等待空闲且绑定 UUID 的 B200 前要求模型位置.
模型未配置时, CPU tokenizer / HTTP 集成测试明确跳过.
纯校验测试无需 checkpoint, 仍然执行.

Scripted HTTP fixture 的服务首次就绪 timeout 为 `30` 秒.
这段等待包含 Python import 和 ZMQ 连接. 请求, 恢复与关闭断言保留原有 deadline.

选定 pytest 参数放在 `cpu` 或 `full` 之后.
Full 模式使用两个新的 pytest 进程, 先执行 context-boundary 模块.
边界收集器的 GPU 所有权检查执行时, 没有此前测试的 CUDA context.
Partition plugin 保留所选 node ID 及其重复次数. 空分组不会掩盖失败.

各进程先使用原始收集范围和 pytest filter, 再按路径分组.
这样 `--ignore` 和 `--ignore-glob` 对边界模块继续生效.

使用 `full --evidence-dir "$EVIDENCE_DIR/python-tests"` 指定新的外部证据目录.
未指定时, launcher 打印新的临时目录, 退出后保留文件.
每个已启动的 pytest 进程保留日志和进程出生记录. Pytest 写入 JUnit 结果时保留文件.
缺少分组报告不能视为完整测试证据.
`tests/test_python_test_partition.py` 将真实收集结果与原始 pytest node multiset 比较, 不执行 GPU 计算.

已有 worker lifecycle 测试在 CPU 上检查模型, callable 和 graph-pool 引用释放.
六项真实模型边界用例检查进程在未改变的 worker shutdown deadline 前正常退出.
移除末尾 `torch.cuda.empty_cache()` 调用使用这些行为检查, 不增加重复源码语句的测试.

测试覆盖调度事务, 缓存所有权, 取消, RPC framing, 服务, 实际路径 FP64 参考和编译模型单元.
GPU 覆盖包括全部 6 个最大上下文用例和 metadata 变化后的 graph replay.

基准所有权检查通过 `nvidia-smi -i` 仅查询选定的 GPU UUID.
查询 timeout 保持 `10` 秒. 查询失败会停止采集.
进程接受条件仍要求精确的所属祖先关系和出生身份.
CPU 回归检查选定设备隔离, 外部进程, 出生身份复用和查询失败.

`tests/test_graph_capture.py` 在 CPU 上检查实际 allocator 设置与工作区生命周期屏障.
`tests/test_dspark_context_graph_gpu.py` 包含生产使用的 29-by-25600 BF16 上下文投影.
它在捕获 stream 的 pool 外预热, 重置无关 GEMM 图, 施加 allocator 压力, 并重放变化后的输入.
它用 `atol=0.03`, `rtol=0.03` 检查全部五层 cache 输出, 精确检查未修改行的 canary, 并验证工作区预算未变.
完整模型 drift 测试在严格算子容差之外, 使用各自记录的模型级界限.
协议回归使用 scripted worker; GPU / agentic 验收使用真实模型.
记录 warning 和跳过测试及原因.

## 思考预算验收

构建 release binary, 用 `serve --thinking-token-budget 32` 启动各模式服务.
使用 `none`, `mtp` 和 `dspark`, 选择空闲且锁定 UUID 的 B200, 并指定对应启动模式.
脚本通过两种 API 检查零, 小值, 默认, 不限与关闭思考预算.
它检查 stream, JSON, strict tool, 总长度, 有效请求 metric 与源码身份不变.
每次运行后停止所属 service 和 worker. 保留原始请求与响应记录.

```bash
scripts/with-env.sh python scripts/thinking-budget-acceptance.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/thinking-budget.json"
```

## 精度

用独立计算的 FP64 参考比较实际舍入输入.
覆盖 prefill, decode, 边界写入, strided view, grouped MTP verification 和 speculative rollback.
BF16 算子输出使用 `atol=rtol=0.03`.
Recurrent state 的 NRMSE 至多 1%, 最大绝对误差至多为参考峰值 2%.
检查每个写入 slot 和未改动的受保护状态.
融合链包括全部舍入, scale, copy 和 reduction 行为.
参见 `tests/probe_worker.py` 和相关内核测试.

`tests/test_ir_normalized_linear.py` 在 `1/2/4/6/12/24/32` 行检查小行数融合.
Graph 重放读取变化后的输入和权重 scales, 保持紧凑的输出形状, 并保持输入字节不变.
存储偏移用例在 `2/17` 行分别检查三个归一化输入.
之前的精确检查和 FP8 算子容差继续有效.

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

GPU 采集前, 在 CPU 上将当前 runtime 与已审核的语义契约核对:

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_performance_metadata.py::test_current_runtime_contract_matches_semantic_review -q
```

此检查核对源码摘要, provider, header, compiler flag 和版本 pin, 不初始化 CUDA.
仅在语义审查后修改 `performance/contract.json`. 保持源码与 provider 相等检查启用.

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
少于 517 个用例 ID 的选择报告 `coverage.complete=false`, 并保持完整的逐用例协议.
矩阵验收前, 全部 517 个用例 ID 必须通过.

原有 317 项保留配置和对比版本.
补充矩阵从 `development/kernels/long_prefill_cases.py` 和 `development/kernels/short_prefill_cases.py` 增加 200 项.
部分采集必须保留其源码范围, 不能单独证明完整矩阵通过.

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
MTP4 另测输入 131072, batch 相同. `prefix` 模式使用 MTP4.
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

除 prefix-hit prefill 外, 每个阶段的墙钟中位数最多为其理论下界中位数的三倍.

记录 prefix-hit prefill 的时延倍率, 不设三倍门槛.
其波动检查继续有效. Prefix-hit decode 保持三倍门槛.
墙钟时延的相对极差最多为 10%.
保留完整失败集和中断尝试. 调查失败后, 对该负载重新执行两次完整预热和五次测量.

每次尝试绑定源码, 二进制, checkpoint 字节, CUDA module, 硬件, 容量和运行时包.
采集期间保持源码不变.
编译 / cache 审计必须显示稳定执行.
现有日志无法排除全部静默内存编译.
原始日志和 trace 放在外部目录.
审阅使用可移植摘要. 正式验证使用完整原始记录.

未提交源码的实验使用 `--diagnostic`.
诊断记录不提供正式验收结论.
Profiling 与正式计时独立执行.

## DSpark 阈值与草稿统计

`--dspark-confidence-threshold` 选择固定或较短 proposal, 与实际 worker 配置一同记录.
初始设置为 `0.2`. 使用 `0.0` 保留至多 7 个固定数量 proposal.
输出, 上下文和 grammar 限制可减少 proposal 数量.
通过已调度 / 接受草稿数及实测性能比较各设置.
接受率为测量期间接受草稿总数除以已调度草稿总数.

使用 `bench --prompt-file PATH` 对与正式输入不同的文本或工具历史开展实验.
文件每行对应一个请求, token ID 以空白字符分隔.
行数必须等于 `--batch-size`, 每行长度必须等于 `--input-len`.
命令在加载模型或 GPU 前检查这些约束.
实验记录保留文本来源, token 哈希, 模板和采样设置.

将加载的草稿配置和权重哈希与指定 checkpoint 字节比较. 两者必须一致.
实际验证草稿为已调度 candidate. 返回的下一步 proposal 使用不同的诊断计数.
可移植摘要不能替代正式验证所需的原始证据.

源码身份及测量范围见 [验收索引](acceptance.zh.md).

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
Collector 管理 4 个持久 OMP 进程和临时记录代理.
每个进程在连续源码审计任务中保留 session ID 和历史.
2 个会话使用 Chat, 2 个使用 Responses. 共享的 4-slot 上限也包含摘要请求.

每条 lane 至少持续 `3600` 秒, 不设请求数下限.
服务就绪后初始化客户端, 在它们发出首次请求前开始计时.
首次编译, 图捕获和后续新形状的开销均计入测量.

此采集器对应的服务使用 `--worker-step-timeout-seconds 600` 和 `--request-timeout-seconds 1800`.
从 diagnostics 记录这些上限. 这些期限允许冷编译完成, 不另行预热.

中途不排空, 不插入故障 probe. 一小时结束后完成在途工作, 再排空请求.

任务和摘要使用 OMP `max`, 映射为服务的 `xhigh`.
此工作负载使用 `serve --thinking-token-budget 8192`, 从 diagnostics 和请求 metric 记录有效值.
此值是测试配置, 不是质量或延迟建议.
OMP 在两种 OpenAI API 中省略此字段. 服务默认值也适用于它的摘要请求.
模型 context window 为 `262144`, 任务输出上限为 `32768`, 包含 reasoning token.
本地 `qwen3` tokenizer 元数据与 provider usage 让 OMP 了解会话大小.
启用 soft 自动压缩, threshold 为 `196608`, reserve 为 `65536`, 保留最近 `20000` 个 token.
OMP 自行选择压缩时机, 包括工具循环中途. 控制器不发送手动 compact 命令.
服务检查完整模板输入与请求输出之和是否在总容量内.
CPU guard 重复生产 normalization 与模板路径, 提供独立容量证据.
将其计数与 SSE usage, SDK 输入加缓存用量, 以及 journal 的 provider-input snapshot 核对.

OMP 的 settled `contextUsage` 使用 provider-input 锚点加上之后的本地变化.
压缩判定还考虑生成输出和已保存历史的估算.
两种记录均保留. 界面百分比不能代表完整的压缩判定.
每次观察到压缩时, 检查会话身份延续, 摘要可用, 以及之后成功的模型工作.
一小时内不强制发生自然压缩. 未覆盖时明确报告.
参见固定版本的 [OMP 压缩流程](https://github.com/can1357/oh-my-pi/blob/v18.4.3/docs/compaction.md).

4 条 lane 分别审计调度/KV, HTTP/生命周期, GPU/缓存, 测试/文档.
它们读取有界源码区间, 搜索实际符号, 在外部 workspace 中保存进度笔记.
逐个已完成的审计工作项核对源码读取及转发的工具结果. 保留增量会话 journal 和完整 RPC/wire 记录.

逐个模型请求核对已完成的服务 metric, SSE 结束记录及 SDK usage.
仅在同 ID 的协议, SDK 和服务完成证明齐全时, 将结束后的 `ConnectionResetError` 或 `BrokenPipeError` 保留为诊断信息.
SDK 与 SSE stop reason 必须相符. `length` 结束的输出数必须等于请求预算.
并行摘要请求必须唯一匹配 usage 和生命周期. 证据有歧义时验证失败.

采集期间检查资源的数量和字节上限. 最后排空后检查 Rust/Python 请求状态为空.
各逻辑池此时有 `blocks - 1` 个空闲块, 包含可驱逐复用的缓存块.
Block `0` 保持分配. HTTP 和 worker 必须保持原有进程出生身份.
结合主机 RSS, GPU allocated / reserved 内存, 图/编译缓存占用, 活动请求和输入长度分析.
变化的会话大小可能改变有界缓存. 独立审查需解释增长并确认内存稳定性.
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

## 教程与清理

使用 [code-journey](../code-journey/README.zh.md#验证) 中的检查.
教程修改后检查桌面 / 移动截图和全部 13 条路线.

所有 GPU 程序经过 `scripts/with-gpu.sh`.
完成或失败后停止全部所属进程及后代.
确认 GPU 进程退出, listener 释放和 IPC 清理.
只保留用户明确要求持续运行的服务; 本机信息记入 `LOCAL.md`.
