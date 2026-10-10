# 开发

## 环境安装

安装 `uv`, Python 3.12 或更高版本, 以及支持 edition 2024 的 Rust 工具链.
已验证的完整运行依赖位于 `requirements/runtime.txt`.
Python 直接依赖位于 `python/pyproject.toml`.
NVIDIA 驱动由主机提供.
使用主机或项目环境中的兼容 CUDA / 编译器工具.

在仓库根目录创建并选择环境:

```bash
uv venv --python 3.12 .venv
export OH_MY_VLLM_CONDA_PREFIX="$PWD/.venv"
uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" -r requirements/runtime.txt
uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" -r requirements/tools.txt
uv pip check --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python"
export OH_MY_VLLM_MODEL="/path/to/Qwen3.8-27B-FP8"
```

`requirements/tools.txt` 固定 pytest, Ruff, pre-commit, clang-format 及受约束依赖.
验证工具修改后, 使用 runtime constraint 从 `requirements/tools.in` 重新生成.

也可以使用现有 conda 环境.
将 `OH_MY_VLLM_CONDA_PREFIX` 设置为其位置, 无需创建 `.venv`.
实际位置记入被忽略的 `LOCAL.md`.
高层依赖在此环境中用 `uv` 安装, 不直接使用 `pip`.
vLLM 及其缓存与项目运行时隔离.

已验证组合包括 Torch 2.14.0, Transformers 5.17.0, FlashInfer 0.6.18.post1, TileLang 0.1.14 和 apache-tvm-ffi 0.1.12.
记录中的构建使用 CUDA 13.1 工具和 Torch CUDA 13.0 运行库.
这些版本描述已测试组合; 完整依赖以 lockfile 为准.

## 配置选择

`scripts/with-env.sh` 按以下顺序选择环境:

1. `OH_MY_VLLM_CONDA_PREFIX`.
2. 已激活的 `VIRTUAL_ENV`.
3. 已激活的 `CONDA_PREFIX`.
4. 仓库 `.venv`.

选定环境不存在时即报错.
包装脚本将其 `bin` 加在 `PATH` 前, 将仓库 `python` 加在 `PYTHONPATH` 前.
包装脚本将 `UV_PYTHON` 设为所选 Python. uv 的显式 `--python` 选项优先.

`CARGO_TARGET_DIR` 默认为仓库 `target`.
`OH_MY_VLLM_WORKER_PYTHON` 默认为选定环境的 Python.
不使用包装脚本时, Rust 使用此变量或 `PATH` 中的 `python3`.
直接调用 Rust library 时显式提供 `WorkerConfig.model_path`.

| 变量 | 用途 |
|---|---|
| `OH_MY_VLLM_MODEL` | Checkpoint 目录; 显式 `--model` 优先. |
| `OH_MY_VLLM_DRAFT_MODEL` | DSpark checkpoint 目录; 显式 `--draft-model` 优先. |
| `OH_MY_VLLM_RUNTIME_CACHE` | 全部项目命令使用的独立运行时缓存根目录. |
| `XDG_CACHE_HOME` | 默认缓存基础目录; 未设置时使用用户 `.cache`. |
| `FLASHINFER_WORKSPACE_BASE` | 覆盖 FlashInfer workspace. |
| `TRITON_CACHE_DIR` | 覆盖第三方 Triton 缓存. |
| `TILELANG_CACHE_DIR` | 覆盖冻结 TileLang 缓存. |
| `TVM_FFI_CACHE_DIR` | 覆盖 native CUDA 构建缓存. |
| `TVM_FFI_CUDA_ARCH_LIST` | CUDA 架构; B200 默认 `10.0a`. |
| `OH_MY_VLLM_KERNEL_BACKEND` | 显式选择 `cuda` 或冻结 `tilelang`. |
| `OH_MY_VLLM_ENFORCE_EAGER` | 设置为 `1` 时启用显式诊断模式. |
| `OH_MY_VLLM_MULTI_STREAM` | GDN 分支 stream. 默认 `1`; 诊断值 `0` 禁用. |
| `OH_MY_VLLM_CUDA_PDL` | 自有 CUDA programmatic launch. 默认 `1`; 诊断值 `0` 禁用. |
| `OH_MY_VLLM_GPU_UUID` | 可选的空闲 B200 选择, 供 `scripts/with-gpu.sh` 使用. |
| `PYTORCH_ALLOC_CONF` | 显式 allocator 选项; 此名称优先于旧名称. |
| `PYTORCH_CUDA_ALLOC_CONF` | 新名称未设置时使用的旧 allocator 选项. |
| `OMP_NUM_THREADS` | 主机线程数量; 默认 `1`. |

默认缓存后缀为 `oh-my-vllm/tilelang-ffi012`, 与已验证库组合兼容.
子目录隔离 FlashInfer, 第三方 Triton, TileLang 和 native CUDA 产物.
首次使用可能编译内核或下载带版本的 FlashInfer GEMM cubin.
运行时身份记录库版本, 拒绝意外 vLLM 导入或映射库.
性能验收期间不启用精度 probe 或 eager override.

两个 allocator 变量均未设置时, 包装脚本设置 `graph_capture_record_stream_reuse:True`.
此选项依据 graph 依赖, 在多 stream 捕获期间复用存储.
显式设置优先, 包括空值.
选择后包装脚本移除旧别名. Trace 身份记录 allocator backend.
在 worker 初始化前设置 stream 与 PDL 选项. 正式采集必须启用两项选项.

## 草稿模式选择

旧 `--num-speculative-tokens 0` 和 `4` 分别选择普通计算和原生 MTP4.
`--speculative-mode none`, `mtp` 或 `dspark` 显式选择 worker 模式.
模式与数量不兼容时在 worker 初始化前报错.
DSpark 通过 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model` 指定独立 checkpoint.

设置草稿 checkpoint 并使用显式模式:

```bash
export OH_MY_VLLM_DRAFT_MODEL="/path/to/local-DSpark-checkpoint"
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/dspark-text.ipc --speculative-mode dspark bench --input-len 32 --output-len 64 --batch-size 1
```

DSpark 至多提出 7 个 token, 原生 MTP4 保留 4-token 路径.

`--dspark-confidence-threshold` 设置累计前缀 confidence, 初始设置为 `0.2`.
使用 `0.0` 保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.
阈值必须有限, 大于等于 0 且小于 1.
worker 启动时选择一种模式. 在 worker 停止前保持该模式.
源码身份及测量范围见 [验收索引](acceptance.zh.md).

## 构建与检查

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/with-env.sh pre-commit run --all-files
```

钩子通过同一包装脚本使用本地系统工具.
CUDA 钩子使用 clang-format 23.1.3, 并检查模块至多 800 行.
`git commit` 前配置环境, 确保钩子选择正确环境.
参见 [贡献流程](../CONTRIBUTING.zh.md).

## TileFoundry 开发工具

源码依赖安装前先安装 metadata 工具:

```bash
git submodule update --init --recursive 3rdparty/TileFoundry
scripts/with-env.sh uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" setuptools-scm==10.2.3 vcs-versioning==2.4.1
scripts/with-env.sh uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" --no-build-isolation -r requirements/development.txt
scripts/with-env.sh tilefoundry --help
```

固定 fork 是开发依赖, 不属于推理依赖.
不为此 fork 创建上游 PR.
参见 [内核开发](kernels.zh.md).

## 部署记录

主机路径, 地址, 选定进程 ID, 缓存位置和维护命令放在被忽略的 `LOCAL.md` 中.
被忽略的 `.local/evidence/` 保存完整原始测量.
跟踪的 [证据索引](acceptance.zh.md) 标明可移植摘要和原始哈希.
新服务器需要自己的模型位置, 兼容环境, GPU 选择和当前源码测量.
仓库不含主机专属的默认模型或环境位置.
