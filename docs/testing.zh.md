# 测试指南 — oh-my-vllm

当前 EngineCore 对比要求吞吐 >=95%、TTFT <=110%，基线和候选实现的两种指标均满足 (max-min)/median <=10%。用户于 2026-09-22 将稳定性上限从 5% 调整为 10%。重新判定必须保留原产物和全部原始重复，并注明新规则。

## 独立迁移检查

使用 development.md 中的独立环境/缓存设置。CPU 测试包含 `test_batch_plan.py`、`test_mtp_plan.py`、`test_independent_sampler.py`；CUDA 测试包含 `test_independent_kernels.py`、`test_independent_decode_attention.py`、`test_decode_graph.py`、`test_elementwise.py`。实际模型探针将 `tests/probe_worker.py` 指定为 `OH_MY_VLLM_WORKER_PYTHON`，并设置 `OH_MY_VLLM_ENFORCE_EAGER=1`；MTP 另加 `OH_MY_VLLM_PROBE_MTP=1` 和 `--num-speculative-tokens 4`。下述探针容差保持不变。

独立路径已通过普通/MTP 文本、MTP 前缀/抢占 batch、12 个 Chat/Responses 约束案例和服务生命周期。独立运行时的冻结基线九组矩阵通过。两种最终 oh-my-pi 工作流均以 MTP4 完成，包含真实源码读取和工具结果回传。回答审查局限及耗时见 acceptance.md。

## 环境与 CPU 检查

全部命令通过 scripts/with-env.sh 设置 PYTHONPATH 和 CARGO_TARGET_DIR。全部测试及模型 Worker 使用 conda oh-my-vllm。该历史基线已冻结，不得重新生成。旧适配器及测试已移除。

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python -m unittest discover -s tests -p test_runtime_tools.py
scripts/with-env.sh python -m unittest discover -s tests -p test_benchmark_tools.py
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_*.py'
scripts/with-env.sh python -m unittest discover -s tests -p test_bridge_logging.py
scripts/with-env.sh python -m unittest discover -s tests -p test_bridge_abort.py
```

Rust 测试覆盖池/释放/hash/前缀记账、分块 prefill、到达、重计算抢占、MTP 接受/拒绝、私有前缀命中状态、跨多块推测槽迁移及完整释放。CPU 基准测试检查无效 MTP 配置及后代进程清理。

## GPU 选择与真实文本

每条单 GPU 命令使用 scripts/with-gpu.sh 等待任意空闲 B200。每次使用唯一 socket，不终止无关 GPU 进程。

```bash
scripts/with-env.sh cargo build --release -p oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

添加 --num-speculative-tokens 4 启用 MTP；添加 --context-repeats 100 跨越 784-token 边界。--prefix-hit 与 --context-repeats 100 组合先预填 prompt，并要求解码真实答案前非零缓存命中。--binary 选择隔离构建。普通/MTP 路径均已生成连贯中文。与吞吐基线相同，固定输出上限有意忽略 EOS。

## 实际路径 FP64 参考探针

在真实文本命令的 with-env.sh 之后添加：

```bash
env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON=/data0/shared/dongwu.chen/oh-my-vllm/tests/probe_worker.py
```

MTP 还需 OH_MY_VLLM_PROBE_MTP=1 和 --num-speculative-tokens 4。探针观测选定真实 FlashInfer GQA 层、target GDN prefill/recurrent 验证调用及 MTP 分页 decode，不替换生产 kernel。检查输出和递归状态，包含每个推测状态。必需覆盖类别全部出现后才能成功关闭。这是单请求 eager 诊断，绝不用于吞吐运行。禁用图捕获，诊断 CPU 复制只观测真实请求。

参考使用实际舍入输入进行 CPU FP64 计算。BF16 输出 atol=rtol=0.03。递归状态要求归一化 RMS 误差 <=1%，最大绝对误差 <= 参考峰值的 2%；两项均记日志。接近零元素的相对状态误差不稳定。这些检查诊断单个 kernel/状态路径，不保证完整模型 FP8 token 相等。历史独立 SDPA 测试是补充，不能证明实际路径覆盖。

## 功能组合与抢占

Rust bench 支持 --arrival-interval（请求间隔步数）、--prefix-hit（计时 batch 前预填）、--warmup 和 --repetitions。全局 --scheduler-blocks 缩小 Rust 池以制造确定性压力，不改变物理 GPU 分配。必须确认结果 preemptions >0；池受限本身不能证明发生抢占。

256 个物理块、max-model-len 8192 的示例：

- 普通模式：scheduler-blocks 10、batch-size 2、input-len 2048、output-len 1024。
- MTP/前缀/错峰：num-speculative-tokens 4、batch-size 2、input-len 2048、output-len 128、prefix-hit、arrival-interval 3。

结果报告精确输出数量、缓存命中、抢占、proposed/accepted draft。功能冒烟可用 warmup 0/repetitions 1，不是性能验收。

用两条不同真实中文 prompt、错峰到达及受限池，检查重计算期间请求隔离：

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

每请求输入 2048、输出 1024 token。脚本要求实际抢占，检查文本开头和尾部对应正确城市，并打印全文供审阅。MTP 额外要求接受 draft；prefix 要求 `initial_prefix_hit_tokens > 0`，排除抢占后重计算命中。这是语义冒烟，不是完整模型等价证明。Rust `run --prompt-file PATH` 每行接收以空白分隔的 token ID；`--arrival-interval 3` 三步后接纳下一请求。CPU 取消测试覆盖运行、等待和被抢占请求，以及清理 Python 注册、采样、MTP 和 Worker 状态的 finished-only 通知。

## 冻结基线性能验收

基准只启动框架，严格校验工作负载与 `bench/baseline/2026-09-19-acceptance.json` 一致，包括固定 SHA256。允许内容相同的副本，拒绝修改或替代数据。没有执行基线的代码路径。每组两次预热、三次测量：

```bash
scripts/with-gpu.sh scripts/with-env.sh taskset -c 8 python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 --warmup 2 --repetitions 3 --output /tmp/independent-ordinary-bs1.json
```

ordinary、mtp、prefix 各跑 bs1/2/4。原始记录依次使用 CPU 核 8..16；沿用各组原始掩码并记录。默认输入 32768、输出 4096、容量单位 1024（341 逻辑块）。MTP 为四 proposal、BF16 GDN 状态；ordinary/prefix 为 FP32 GDN。受控前缀命中必须恰好 batch_size*32144；预填在计时外。计时含注册直至完成通知。确定性 token ID 工作负载、采样、缓存策略和上下文限制必须匹配冻结元数据。

driver 记录原产物 hash、源码 HEAD/status/diff、每个 Python 源文件 hash、可执行文件 hash、当前独立运行库版本、GPU UUID、CPU 掩码及原始测量，执行 hash 校验的私有二进制副本。测量期间不改 Python 源码。最终验收要求干净实现提交，诊断明确分开。九组各自吞吐中位数 >=95%，出现明显方差时补测。

全程监控 GPU 客户端。同 GPU 外部进程使运行失效，并触发自有进程组清理；绝不杀其他用户进程。UUID 固定和协作锁不能代替监控。轮询无法排除任意短干扰，因此仍需调查无法解释的方差。超时清理包括子 worker。CPU 测试覆盖这些所有权规则，并拒绝数字 GPU ID、无效 MTP 设置和误用旧解释器。

## 服务 CPU、客户端与 GPU 覆盖

```bash
scripts/with-env.sh cargo build
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_serving*.py'
```

Rust 测试覆盖增量 reasoning/XML、字面字符串、请求映射和响应状态。test_serving_worker.py 使用真实 tokenizer/XGrammar，覆盖 strict 约束、有效/无效推测前缀回滚、reasoning-end 跨越、EOS、stop 和 UTF-8。test_serving_http.py 启动真实 Rust 服务和脚本化 CPU ZMQ worker，覆盖两 API 表示、usage、工具历史、存储续接/删除/过期、长度截断、取消、并发、错误请求和 worker 致命错误。这些不是 GPU/模型证据。

scripts/agentic-acceptance.py 见 serving.md。在 UUID 固定的真实 B200、MTP4 服务上，两种 OMP provider 必须分别执行规定任务，回传工具结果并生成基于文件的答案。还需以 MTP 测 strict JSON/工具约束，确认实际 draft proposal 非零，检查接受数、时延/吞吐日志和取消后清理。历史 V2 验收见 acceptance.md；当前独立验收状态见 handoff.md。用 scripts/serving-acceptance.py 复现约束案例，结合保存响应检查逐请求 MTP 计数。脚本化输出绝不是 GPU 证据。

## 严格 Rust 格式

```bash
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh python -m unittest discover -s tests -p test_rust_style.py
scripts/with-env.sh pre-commit run cargo-fmt --all-files
scripts/with-env.sh pre-commit run rust-line-width --all-files
```

rustfmt.toml 使用稳定 Rust 2024 格式、100 字符宽度、Unix 换行及多行 if/else、let/else。pre-commit 检查而非静默改写。硬行宽 hook 读取相同配置，检查已跟踪/未忽略的新 `.rs`，包含宏、注释和字面字符串，并按 rustfmt tab 大小展开制表符。单用 rustfmt 可能不处理超长 `json!` 宏体，需手动拆字段、用 `concat!` 拆长字面量且保持值不变。配置和 hook 修改触发两项 Rust 样式检查。

## 独立算子与模型探针

在新 conda 环境中运行。独立 FlashInfer/Triton 缓存根目录防止悄悄复用旧环境编译产物。模型探针是短 eager 诊断，不是服务或性能验收。独立 Worker 是唯一执行路径；该历史完整性能矩阵通过。最终服务/agentic 证据见 acceptance.md。

```bash
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p test_independent_sampler.py
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p test_serving_worker.py
scripts/with-gpu.sh scripts/with-env.sh env \
  FLASHINFER_WORKSPACE_BASE=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent \
  TRITON_CACHE_DIR=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent/triton \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MAX_JOBS=8 \
  python -m pytest tests/test_independent_kernels.py -x -q
scripts/with-gpu.sh scripts/with-env.sh env \
  FLASHINFER_WORKSPACE_BASE=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent \
  TRITON_CACHE_DIR=/data0/shared/dongwu.chen/.cache/oh-my-vllm/independent/triton \
  OMP_NUM_THREADS=1 MAX_JOBS=8 python scripts/probe-independent-model.py
```

算子测试对 block-scaled FP8、分页 GQA、gated delta 递归、因果卷积、RMS 归一化和部分旋转位置编码使用 CPU 参考。覆盖多 FP8 行、不连续逻辑注意力页、784/785-token 边界及逐候选 MTP 快照。不能替代实际模型 FP64 探针或最终服务/性能验收。

## TTFT 与长上下文回归（2026-09-22）

新 12 组协议见 REQ-PERF-002，验收仍待完成。`test_ttft_metrics.py` 拒绝尾时延失败、缺少预热、错误输出数、意外缓存命中、错误来源及未验证编译。仅基线采集器是导入审计的明确隔离例外。`test_independent_decode_attention.py` 用 FP64 参考覆盖普通和分组验证 kernel 中超过有符号 32 位范围的页地址。coordinator 测试在共享/独立状态池上运行 MTP 迁移、前缀和拒绝案例。最终模型测试须覆盖完整 262144-token 边界。

## TileLang 迁移验收

TileLang 替换保持现有测试和容差不变。运行完整 GPU pytest、实际模型普通/MTP FP64 探针、文本和抢占/前缀检查、MTP 服务/agentic 检查、六组 262144-token 边界，以及冻结吞吐/TTFT 的全部 12 组对比。临时 TileFoundry/算子实验放在仓库外，见 [tilelang-development.md](tilelang-development.zh.md)。TileFoundry 分析或代表性 HIR 检查不能替代这些要求。
