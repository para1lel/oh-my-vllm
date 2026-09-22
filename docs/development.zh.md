# 开发指南

## 环境

全部 Rust 工具、Python 包、测试和模型执行使用 `/data0/shared/dongwu.chen/conda-envs/oh-my-vllm`。cargo/Python 命令使用 `scripts/with-env.sh`，设置 PATH、PYTHONPATH、CARGO_TARGET_DIR 和 Worker Python。旧适配器和旧环境默认值已移除。

通过 uv 向该环境安装 Python 依赖。直接依赖在 python/pyproject.toml，requirements/runtime.txt 固定完整稳定 Linux/Python3.12 依赖闭包。主机 CUDA13.1/编译工具用于构建 kernel，Torch2.14 使用已安装的 CUDA13.0 运行库。NVIDIA 驱动由宿主机提供。

```bash
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python -r requirements/runtime.txt
uv pip check --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python
```

包装脚本在 `/data0/shared/dongwu.chen/.cache/oh-my-vllm/tilelang-ffi012` 下选择独立 FlashInfer/TileLang/第三方 Triton 缓存。通过 OH_MY_VLLM_RUNTIME_CACHE 覆盖公共根目录。FlashInfer 首次使用可能编译 kernel，或下载其带版本的 NVIDIA GEMM cubin；这些是独立库产物，不是旧 vLLM 构建输出。启动记录库版本，关闭时检查导入模块及映射库。环境中不得安装 vLLM。

`OH_MY_VLLM_ENFORCE_EAGER=1` 仅用于诊断，禁用 target/MTP 图。只有明确授权的隔离采集器可更新 EngineCore 基线。性能验收绝不启用正确性探针。

## 构建与检查

Rust workspace 使用 edition2024 和现有 conda 工具链。

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh pre-commit run --all-files
```

hook 是 .pre-commit-config.yaml 中的本地/system hook。只暂存有意修改的路径，包括有变更的 Cargo.lock。避免未暂存的 hook 修改与 pre-commit 临时 stash 冲突。不禁用 hook，不用 git add .。

## GPU 执行

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

多条真实 prompt 可用 `run --prompt-file PATH`：每行是一条以空白分隔的 token ID 序列，最多 32 个请求。`--arrival-interval N` 按调度步骤错峰接纳；`--prefix-hit` 先预填 prompt 并要求初始命中。输出包含有序 token batch 和功能计数。带显存压力的双请求 `scripts/smoke-batch.py` 及抢占后文本检查见 testing.md。

GPU 包装脚本等待空闲 B200：无计算进程、显存使用不超过 64 MiB、报告利用率为零。选择 UUID 并持有协作 flock；无关程序不一定遵守该锁。基准在引擎运行期间额外检测同 GPU 外部客户端，使受影响测量失效。不得中断无关任务。使用唯一 socket；仅确认所有者退出后才删除遗留 socket。

历史 benchmark 默认输入 32768/输出 4096、容量单位 1024、预热两次和测量三次。运行 ordinary、mtp、prefix 模式。精度探针、组合和验收详见 testing.md。验收不启用 eager/probe。--binary 支持隔离构建，driver 执行经过 hash 校验的私有副本。运行中不修改 Python 源码。

## 日志与诊断

日志写 stderr，结果写 stdout。Rust tracing 和 Python JSON 行使用 UTC 时间戳。OH_MY_VLLM_RUN_ID 关联两端，step_id 关联 RPC。INFO 记录初始化、容量和 batch 摘要。RUST_LOG=debug 与 OH_MY_VLLM_LOG_LEVEL=DEBUG 开启调度时间/空闲块、RPC 耗时、Python 消息解码、主机执行和编码/发送计时。

主机耗时不是 CUDA kernel 耗时。GPU 计时单独运行 profiling，trace 留在仓库外。默认 INFO 避免逐步骤 I/O。控制器对串行调度/RPC 流使用一个 Tokio 线程。CPU 亲和性实验需给两引擎相同的继承掩码并记录；taskset 可放在 benchmark Python 命令前。实测结果、失败情况和待完成验收见 handoff.md。

## 故障排查

- 初始化失败：看启动日志中 Python traceback 和 conda 解释器，检查固定依赖、checkpoint 配置和独立缓存。
- 地址占用：换 socket，或确认遗留 socket 的所有者已退出。
- worker 意外退出：重跑前检查 stderr；子进程在连接前退出会及时报告，不会耗尽整个初始化超时。
- 启动显存不足：重新等空闲卡。选择后可能有外部任务进入，不能降低限制掩盖被污染的配对。

## OpenAI 服务开发

启动/OMP 命令及协议兼容矩阵见 serving.md。Rust 使用 Axum/serde_json/tokio-stream，Python 使用固定的 tokenizer 和 XGrammar 包。

在 GPU 主机上做纯 CPU tokenizer/schema/HTTP 测试时，启动 Python 前设置 CUDA_VISIBLE_DEVICES=''，避免设备初始化；这不验证 CUDA 执行。tests/fixtures/serving_worker.py 仅为集成测试提供脚本化输出，不得用于报告模型验收或速度。

对正在运行的 MTP4 服务执行真实约束输出检查：

```bash
scripts/with-env.sh python scripts/serving-acceptance.py --output-dir /tmp/oh-my-vllm-serving-constraints
```

脚本保存完整请求/响应。用匹配的 response/request ID 检查服务日志中的真实 MTP proposal/acceptance 计数。Python DEBUG 日志展示消息解码、执行和编码/发送时间。

## 严格 Rust 格式

```bash
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh python -m unittest discover -s tests -p test_rust_style.py
scripts/with-env.sh pre-commit run cargo-fmt --all-files
scripts/with-env.sh pre-commit run rust-line-width --all-files
```

rustfmt.toml 设置稳定 Rust 2024 格式、100 字符宽度、Unix 换行和多行 if/else、let/else 表达式。pre-commit 检查格式，不静默重写。独立硬行宽 hook 读取同一配置，检查已跟踪和未忽略的新 `.rs` 文件，包括宏、注释、字符串；按 rustfmt tab 大小展开制表符。仅 rustfmt 可能保留超出 max_width 的 `json!` 宏体，需手动拆字段并用 `concat!` 换行且保持字面量值不变。配置和 hook 修改会触发两项 Rust 样式 hook。

## 隔离参考采集（2026-09-22 授权）

`benchmarks/baseline/enginecore.py` 仅用于基线，用单独维护的 vllm 环境运行，不向 oh-my-vllm 安装 vLLM。要求官方上游 SHA e9f169d16b9408bb9ae44f75072b91a5521d733c。独立 `benchmarks/ttft.py` 读取该 JSON，只启动自有 worker。此明确例外不改变日常构建/测试/运行依赖隔离。

长上下文运行在 CLI 子命令前添加 `--max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128`。这些是 GPU 验证中的暂定容量，代表 1400 个 FA 槽和 128 个独立 GDN 槽。

## TileLang 与 TileFoundry

固定 fork、源码安装、完整算子 HIR、生产 runtime twin 和审查/验证流程见[自定义 kernel 工作流](tilelang-development.zh.md)。启动推理 worker 不需要 TileFoundry。运行时使用 TileLang；独立依赖内部仍可使用 Triton。
