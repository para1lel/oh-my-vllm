# 开发环境

开发使用 Python 3.15.2, Rust, pip, 以及 Node 24 或更新版本.
Conda 定义提供 Python 和 Rust.
`requirements/runtime.txt` 提供推理包.
`requirements/development.txt` 提供开发包, 并包含 runtime 文件.

## 安装

先安装 Conda 和 Node.
执行以下命令:

```bash
conda env create -f environment.yml
conda activate oh-my-vllm
export OH_MY_VLLM_CONDA_PREFIX="$CONDA_PREFIX"
PYZMQ_BACKEND=cffi python -m pip install --ignore-requires-python -r requirements/development.txt
npm --prefix development/javascript ci --ignore-scripts --no-audit --no-fund
scripts/with-env.sh pre-commit install
```

已有可用环境时, 执行最后四条命令.
wrapper 只接受 `OH_MY_VLLM_CONDA_PREFIX` 和 `oh-my-vllm` Conda environment.
`PYZMQ_BACKEND=cffi` 设置使 CPython 3.15 binding 使用所选 Conda ZeroMQ library 构建.
由于当前 AgentX wheel 有过时的 Python 上限, 必须使用 `--ignore-requires-python`.

Python 和 Cargo 命令通过 `scripts/with-env.sh` 执行.
wrapper 配置可执行文件路径, Python 模块路径, worker 解释器和 Cargo 构建输出.
不要创建 repository virtual environment.
development 安装包含 runtime 包.

## Git hooks

```bash
scripts/with-env.sh pre-commit run --all-files
```

| Hook | 契约 |
|---|---|
| cuda-format | clang-format 23.1.3, 原生后端模块不超过 800 行 |
| cargo-fmt | Rust 格式化 |
| rust-line-width | Rust 行宽不超过 100 列 |
| definition-spacing | 通过 Python, Rust 和 JavaScript/TypeScript 语法树检查函数及模块间的空行 |
| cargo-clippy | 所有 workspace 成员, targets 和 features, 警告视为错误 |
| cargo-test | 仓库中的 Rust workspace 测试和 Python 服务测试 |
| kernel-contracts | 模型证据, HTTP 计费及 worker 约束 |
| ruff-format | Python 格式化 |
| ruff-check | Python lint 规则 |
| whitespace | 无行尾空白, 文件末尾保留一个换行 |
| documentation | 翻译配对, 词数限制, 主机路径, 链接和源码锚点 |
| glossary | 生成词表与源文件一致 |

Cargo hooks 包含所有自有 manifest. 每个 workspace 只处理一次.
无适用文件的 hook 没有检查对象. 文档 hook 始终运行.

新增测试前遵守[测试策略](../CONTRIBUTING.zh.md).
代理代维护者执行仓库中的测试.

## 原生模块

原生构建使用 CUDA 13.4 toolkit. 主机路径保存在忽略的 `LOCAL.md` 中.
将 `CUDA_HOME` 设为 CUDA 安装目录.
development 文件安装原生构建包, 并包含 `requirements/runtime.txt`.
如果没有 CUDA 13.4, 在原生 rebuild 前停止. 不要将 CUDA 13.2 headers 与 CUDA 13.4 Torch runtime 混用.

将 `CUTLASS_ROOT` 设为外部依赖目录.
获取指定头文件:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

构建仅接受干净的 Git revision `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
将 `OH_MY_VLLM_KERNEL_CACHE` 设为外部构建目录.
构建原生模块:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build
```

推理 loader 不导入开发工具, 也不使用编译器.
它根据构建 manifest 检查源码和二进制 identity.
每个二进制在外部 cache 中具有不可变副本和 manifest.

模型命令见 [Qwen 验证](models/qwen38/README.zh.md).
