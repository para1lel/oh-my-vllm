# 测试指南 — oh-my-vllm

[English](testing.md)。agent 以英文版为准。

所有命令都经由 `scripts/with-env.sh` 运行，它设置 conda 环境、`PYTHONPATH`、`CARGO_TARGET_DIR`
以及独立的 kernel 缓存根目录。GPU 命令还要经由 `scripts/with-gpu.sh`，它会等待一块空闲的 B200
并固定其 UUID。每次运行使用唯一的 socket。运行结束时停止你启动的每一个 GPU 进程。

已知的测试缺口记录在 [audit-2026-09-23.md](audit-2026-09-23.zh.md)（EVD-03…EVD-11）中。
不要把其中列出的检查当作它未能覆盖的那项属性的证据。

## 静态检查与 CPU 检查

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/test.sh cpu
```

`scripts/test.sh` 先构建 Rust HTTP 二进制文件，再通过项目环境运行 pytest。`cpu`
隐藏 CUDA 并排除带 `gpu` 标记的用例；`full` 等待空闲 B200 并固定 UUID。pytest 是
统一的 Python 测试入口。旧的 `unittest discover` 命令会导入以下 pytest 风格的模块，
但不运行其中测试：

- `test_independent_kernels.py`
- `test_independent_decode_attention.py`
- `test_elementwise.py`
- `test_greedy_batch.py`
- `test_proposal_graph.py`
- `test_gqa_accuracy.py`

它们由 full 模式收集；GPU 用例在 `pytest.ini`/测试模块中明确标记。混合的
greedy-batch 和 validation-guard 模块仅标记其中的 GPU 用例，因此 `cpu` 模式
仍运行其 CPU 用例。

Rust 测试覆盖：

- 池、空闲链表、哈希及前缀计数
- Chunked prefill 与请求到达
- 重算抢占（recompute preemption）
- MTP 接受与拒绝
- 私有前缀命中状态
- 投机槽位迁移
- 分离的 FA/GDN 池
- 完整释放

CPU Python 测试覆盖：

- Batch 规划与 sampler
- 服务预处理（真实 tokenizer/XGrammar）
- 使用脚本化 worker 的 HTTP 服务器
- Bridge
- Benchmark 与测量工具
- 算子对比统计与冻结参考实现的完整性

脚本化 worker（`tests/fixtures/serving_worker.py`）产生的是伪造输出，绝不能作为模型或性能证据。

## 完整 GPU 测试套件

```bash
scripts/test.sh full
```

在 `ef07b2b` 之后、后来成为 `030f60f` 的工作树上运行的历史默认 CUDA 套件
通过 174 项测试及 31 个子测试，没有跳过
（`bench/baseline/2026-09-22-cuda-features.json`）；没有在干净的 `030f60f`
提交上单独重跑。`96e4ecc` 后的完整
B200 套件通过 211 项测试及 32 个子测试，六项 context 边界占位测试跳过。这些
跳过不能验证 REQ-CONTEXT-001（EVD-07）；`d33b844` 未重跑完整 GPU 套件。

算子测试将每个 kernel 与 CPU FP64 参考实现对比，覆盖：

- Block-scaled FP8 与 paged/grouped GQA
- Gated delta 递推、因果卷积、RMS 与 partial rotary
- 融合的 attention 预处理
- 非连续页
- 784/785 token 边界
- 超出 int32 的页地址
- 每个候选的 MTP 快照
- Graph 重放

设置 `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 可在冻结的对比后端上运行适用测试。
`test_gqa_accuracy.py` 专门将自有 CUDA 分页 decode 与 CPU FP64 比较，包含
784/785 的两页边界；TileLang 后端下跳过该项。graph 测试也有独立 CPU FP64
参考（EVD-04/05，`4ed62ab`）。

## 真实文本与实际路径 FP64 探针

```bash
scripts/with-env.sh cargo build --release -p oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

选项：

- `--num-speculative-tokens 4`：启用 MTP。
- `--context-repeats 100`：跨越 784 token 边界。
- `--prefix-hit` 配合 `--context-repeats 100`：先预置 prompt，并要求命中缓存。
- `--binary`：选择隔离的构建产物。

固定的输出上限会忽略 EOS，与吞吐 workload 一致。

若要为真实 kernel 调用添加 FP64 探针，在 `with-env.sh` 之后追加以下变量：

```bash
env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON=/data0/shared/dongwu.chen/oh-my-vllm/tests/probe_worker.py
```

对于 MTP，还需设置 `OH_MY_VLLM_PROBE_MTP=1` 并传入 `--num-speculative-tokens 4`。
`scripts/probe-independent-model.py` 运行同样的短 eager 模型探针。

- **探针插桩的对象：** 真实的 GQA、GDN prefill 与递推验证，以及 MTP paged decode。它从不替换生产 kernel。
- **参考：** 在实际舍入后的输入上计算的 CPU FP64 参考。
- **容差：** BF16 输出使用 atol=rtol=0.03。递推状态要求 NRMSE ≤ 1%，且最大绝对误差 ≤ 参考峰值的 2%。
- **覆盖要求：** 必须在关闭前出现所有要求的覆盖类别，关闭才会成功。
- **范围：** 这些是 eager、单请求的诊断，不构成完整模型 token 一致性的结论。由于禁用了
  graph capture，诊断用的 CPU 拷贝只观察真实请求。接近零的逐元素状态相对误差不稳定，不作为判据。

绝不要在性能运行中启用探针或 eager 模式。

## 功能组合与抢占

Rust `bench` 命令接受 `--arrival-interval`、`--prefix-hit`、`--warmup` 和 `--repetitions`。
全局选项 `--scheduler-blocks` 会缩小 Rust 池以强制抢占。需检查结果中 `preemptions > 0`，
因为仅凭池较小并不能证明确实发生了抢占。

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

每条命令以错开的到达时间运行两个不同的中文 prompt。每个请求有 2048 个输入 token 和 1024 个输出 token。

脚本检查：

- 观察到抢占。
- 每个输出的开头和结尾都给出正确的城市。
- 有被接受的草稿（MTP）。
- `initial_prefix_hit_tokens > 0`（前缀模式）。

它还会打印完整的生成文本供检查。这些是语义层面的冒烟检查。

`run --prompt-file PATH` 每行读取一个 token-ID 请求。

## 冻结基线性能验收（REQ-PERF-001/002）

**工具。** `benchmarks/ttft.py` 是验收工具。它只启动本项目的 worker，并在每次调用时强制执行：

- 至少 2 次 warmup 和 5 次重复。
- 每个请求 4096 个输出且零抢占。
- 精确的前缀命中计数。
- 一致的 CPU 亲和性、GPU 型号与驱动。
- 冻结的基线 SHA。
- 稳态编译/捕获审计。
- 吞吐比 ≥ 0.95、TTFT 比 ≤ 1.10，且两个引擎的两项指标均满足 `(max-min)/median` ≤ 10%。

候选侧的 scheduler 配置和 FA/GDN 容量从 worker 日志与已分配设备张量读取，再与请求值
比较（EVD-02）；这条路径仍待新的完整 12 行运行。验收使用 `benchmarks/ttft.py`。
`benchmarks/compare_vllm.py` 现在拒绝少于五次正式重复或超过 10% 的离散度，因此其
历史九行基线只有三次重复，无法通过当前门槛（EVD-01）。

**基线输入。** 冻结的基线行嵌入在 `bench/baseline/2026-09-22-refreshed-enginecore.json` 的
`rows[].artifact` 下。须逐字节一致地提取一行，格式为 `json.dumps(artifact, indent=2)` 加一个结尾换行。
其 SHA-256 必须等于 `rows[].sha256`。然后在 `artifact.hardware.cpu_affinity` 记录的 CPU 核上运行该行：

```bash
scripts/with-env.sh python - <<'EOF'
import hashlib, json
rows = json.load(open("bench/baseline/2026-09-22-refreshed-enginecore.json"))["rows"]
for row in rows:
    text = json.dumps(row["artifact"], indent=2) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == row["sha256"]
    open(f"/tmp/baseline-{row['label']}.json", "w").write(text)
    print(row["label"], ",".join(map(str, row["artifact"]["hardware"]["cpu_affinity"])))
EOF
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh taskset -c 8-15 \
  python benchmarks/ttft.py --baseline /tmp/baseline-ordinary-32768-1.json \
  --output /tmp/candidate-ordinary-32768-1.json
```

默认值为 `--num-gpu-blocks 4200`、`--mamba-blocks 128`、2 次 warmup 和 5 次重复。任何门槛未通过时，
脚本以非零状态退出。它在仓库根目录下运行 `target/release/oh-my-vllm-zmq-worker`，因此不支持
自定义 `CARGO_TARGET_DIR`。prefix 各组必须恰好报告 `batch_size * 32144` 个命中 token
（每个请求 `(32768-1)//784*784`）。

**某一行计为验收所需的规则：**

- 从干净的提交构建，运行期间不要编辑源码。
- 保留每一次尝试。若出现外部 GPU 进程，该次运行无效。调查任何离散度失败，然后重复完整的一组运行。
- 绝不挑选单次重复。
- 未经用户明确授权，绝不重新运行 vLLM 基线。

**结果存放位置。** 已接受的矩阵记录在 `bench/baseline` 中，参照 `2026-09-22-cuda-framework.json`。

## 最大上下文（REQ-CONTEXT-001）

运行六个边界行（普通与 MTP4，batch 1/2/4，258048 输入加 4096 输出）。全局选项放在 `bench`
子命令之前；普通模式使用 `--num-speculative-tokens 0`，MTP 使用 `4`：

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker \
  --socket /tmp/boundary-mtp-4.ipc --num-gpu-blocks 4200 --mamba-blocks 128 \
  --max-model-len 262144 --num-speculative-tokens 4 \
  bench --batch-size 4 --input-len 258048 --output-len 4096 --warmup 0 --repetitions 1
```

每次运行必须显示：

- 无 OOM。
- `preemptions == 0`。
- 记录了峰值显存：worker 在关闭时记录 `max_reserved_bytes`
  （`worker/model_runner.py:380-385`）。

目前尚无自动化测试覆盖这一点（EVD-07）。

`scripts/long-context-acceptance.py` 针对正在运行的 MTP4 服务，测试 131072 token 的 strict-JSON
请求和前缀复用。需从服务器日志确认 MTP 活动。

## 服务

```bash
scripts/with-env.sh cargo build
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_serving*.py'
```

这些测试覆盖：

- 针对 reasoning/XML 解析器、请求映射与响应状态的 Rust 单元测试。
- `test_serving_worker.py`：使用真实 tokenizer 和 XGrammar，测试严格约束、投机前缀回滚、
  跨越 reasoning 结束位置、EOS、stop 以及 UTF-8。
- `test_serving_http.py`：使用脚本化 CPU worker 的真实 Rust 服务器，覆盖两种 API、usage、
  工具历史、已存储的响应、截断、取消、并发和错误处理。

针对 MTP4 服务的真实 GPU 验收使用以下脚本。其命令和所需的日志检查见 [serving.md](serving.zh.md)。

- `scripts/serving-acceptance.py`：12 个约束用例。
- `scripts/serving-lifecycle.py`：思考档位、已存储的响应、混合 batch 和断开连接。
- `scripts/agentic-acceptance.py`：两种 API 上的 oh-my-pi。

脚本化输出绝不是 GPU 证据。

## 算子对比（REQ-KERNEL-002）

`benchmarks/kernels.py` 运行 `development/kernels/cases.py` 中定义的正式静态用例。它使用成对、
交错的计时，将 CUDA 与冻结的 TileLang 进行对比。命令和判定规则见
[cuda-development.md](cuda-development.zh.md)。

`test_kernel_comparison.py` 测试判定统计。`test_kernel_reference.py` 测试冻结参考实现的哈希和后端选择。

该对比只测量时间。后端之间的输出一致性来自 FP64 测试套件（EVD-09）。

## 严格的 Rust 格式

```bash
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh python -m unittest discover -s tests -p test_rust_style.py
scripts/with-env.sh pre-commit run cargo-fmt --all-files
scripts/with-env.sh pre-commit run rust-line-width --all-files
```

`rustfmt.toml` 设置：

- Stable Rust 2024 格式
- 100 字符宽度
- Unix 换行符
- 多行形式的 if/else 与 let/else 表达式

另有一个硬宽度 hook，以相同限制（展开 tab）检查宏、注释和字符串字面量，rustfmt 可能会让这些内容保持超长。
将 `json!` 主体拆分为字段，并对长字面量使用 `concat!`。
