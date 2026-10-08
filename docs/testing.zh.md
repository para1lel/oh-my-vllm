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

TTFT harness 读取仓库 `target/release/oh-my-vllm-zmq-worker`.
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
计时前校验完整输出和实际写入缓存.
每个用例需要 3 轮, 每轮 20 个交错配对, CUDA 每轮中位数更快.
节省时间的 hierarchical-bootstrap 单侧 95% 下界必须为正.
每个计时样本使用 100 次 graph repetition.
用例和轮次之间恢复持久状态.
Normalization, Q / K, recurrent 和 convolution 校验要求 1 次 fast, 0 次 generic host dispatch.
不以 graph replay 次数代替 host dispatch 次数.
来源记录和 profiling 分离见 [内核开发](kernels.zh.md).

## 框架性能

当前 12 行协议使用 `benchmarks/ttft.py`.
它接受单行完整原始基线, 包括 hardware, configuration, warmup, repetition 和 measurement audit.
历史刷新产物包含多个 row; 提取其中原始 baseline row, 不改变字段.
原主机使用保存的完整记录, 其他服务器重新采集匹配基线.
提取前, 将 `BASELINE_COLLECTION` 设为完整的刷新基线集合.
序列化规则和哈希断言保留每个归档行的字节身份.
不要提供可移植摘要.

```python
import hashlib
import json
import os
from pathlib import Path

collection = json.loads(Path(os.environ["BASELINE_COLLECTION"]).read_text())
assert "artifact_kind" not in collection, "use the full original collection"
output = Path(os.environ["EVIDENCE_DIR"])
output.mkdir(parents=True, exist_ok=True)
for row in collection["rows"]:
    raw = (json.dumps(row["artifact"], indent=2) + "\n").encode()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    (output / f"baseline-{row['label']}.json").write_bytes(raw)
    print(row["label"], row["artifact"]["hardware"]["cpu_affinity"])
```

比较前明确拒绝可移植摘要.

设置 `EVIDENCE_DIR`, 模型和匹配的 CPU affinity 后, 单行命令为:

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/ttft.py --baseline "$EVIDENCE_DIR/baseline-ordinary-32768-1.json" --output "$EVIDENCE_DIR/candidate-row.json"
```

其他行按 label 替换 `--baseline` 文件. Harness 从该记录选择 mode, input/output length 和 batch.
提取结果打印 label 和 CPU affinity; 采集时使用匹配的 CPU affinity.

覆盖 ordinary/MTP4/prefix-hit 的 input32768, output4096, batch1/2/4, 以及相同 batch 的 ordinary input131072.
Candidate 默认使用 4200 个历史容量单位和 128 个 GDN slot.
对应 1400 个 FA slot.
校验实际 worker 和 scheduler 容量, 不只看 CLI 设置.
基线容量需让相同工作负载驻留, 无抢占.

匹配 CPU affinity 和 GPU 型号, 显存, 驱动.
比较器还要求相同 model / configuration 和冻结基线源码身份.
使用 2 次完整预热和 5 次重复.
记录每个请求 TTFT 和固定输出数量.
每行应用 95% 吞吐, 110% TTFT 和 10% spread 门槛.
Spread 失败需调查并完整重跑.
保留全部拒收完整尝试和中断运行.

采集器记录源码, binary, Python 文件哈希, package 版本, 加载 CUDA 模块和完整日志.
Log / cache 审计需显示测量处于稳态, 没有观察到编译或 capture.
现有日志和缓存记录不能排除静默的内存内重新编译.
性能运行禁用会改变执行的诊断功能.
GPU-only prefill timing 不作为 EngineCore TTFT.

`benchmarks/compare_vllm.py` 是历史 9 行工具, 校验固定的原始 SHA.
早期 3 次重复产物不满足当前 5 次重复规则.
工具要求显式 `--baseline-json`, 从不启动 vLLM.
旧验收不作为当前性能比较分母.

## 隔离基线采集

基线采集器是唯一获授权依赖 vLLM 的工具.
使用独立 interpreter, checkout 和 cache.
不通过项目环境包装脚本调用它.
将 `BASELINE_PYTHON` 和 `BASELINE_CHECKOUT` 设置为隔离安装位置:

```bash
scripts/with-gpu.sh "$BASELINE_PYTHON" benchmarks/baseline/enginecore.py --checkout "$BASELINE_CHECKOUT" --model "$OH_MY_VLLM_MODEL" --mode ordinary --batch-size 1 --input-len 32768 --output-len 4096 --output "$EVIDENCE_DIR/baseline-row.json"
```

采集器要求冻结提交和未修改 checkout.
记录实际容量和完整测量身份.
新源码 revision 需要单独批准的基线策略.
基线 cache 与项目 cache 隔离.

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
scripts/with-env.sh python scripts/serving-acceptance.py --base-url "$SERVICE_URL" --output-dir "$EVIDENCE_DIR/constraints"
scripts/with-env.sh python scripts/serving-lifecycle.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle.json"
scripts/with-env.sh python scripts/long-context-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/long-context.json"
scripts/with-env.sh python scripts/agentic-acceptance.py --api chat --base-url "$SERVICE_URL" --output-dir "$EVIDENCE_DIR/agentic-chat"
scripts/with-env.sh python scripts/agentic-acceptance.py --api responses --base-url "$SERVICE_URL" --output-dir "$EVIDENCE_DIR/agentic-responses"
```

Lifecycle 工具必须在所提供日志中找到混合请求共享 batch, 以及 worker 取消 / 释放证据.
检查服务日志中的正 MTP proposal 和 accepted draft 计数.
这些测试完成后立即停止服务及其 worker.

长上下文门槛检查 strict JSON 内容, 输入超过 131072, 重复前缀复用和逐请求正 MTP proposal.
按 [服务合同](serving.zh.md#验证) 分别完成两种 API 的真实 OMP 任务.
要求实际读文件和工具结果, 不只看脚本成功.

## 教程与清理

使用 [code-journey](../code-journey/README.zh.md#验证) 中的检查.
教程修改后检查桌面 / 移动截图和全部 12 条路线.

所有 GPU 程序经过 `scripts/with-gpu.sh`.
完成或失败后停止全部所属进程及后代.
确认 GPU 进程退出, listener 释放和 IPC 清理.
只保留用户明确要求持续运行的服务; 本机信息记入 `LOCAL.md`.
