# 开发环境

开发使用 Python 3.12, Rust, uv, 以及 Node 24 或更新版本.
Conda 定义提供 Python 和 Rust. Python 和 JavaScript 锁文件固定 hook 依赖.
此环境与后续推理依赖分开.

## 安装

执行以下命令前, 先安装 Conda, uv 和 Node:

```bash
conda env create -f environment.yml
conda activate oh-my-vllm
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/development.txt
npm --prefix development/javascript ci --ignore-scripts --no-audit --no-fund
scripts/with-env.sh pre-commit install
```

已有环境时, 激活该环境后执行最后四条命令.
wrapper 依次选择 `OH_MY_VLLM_CONDA_PREFIX`, `CONDA_PREFIX`, `VIRTUAL_ENV`, 仓库内的 `.venv`.
Python 和 Cargo 命令通过 `scripts/with-env.sh` 执行.
wrapper 统一配置可执行文件查找, Python 模块查找及 Cargo 构建输出, 供 hooks 和直接命令使用.

## 提交 hooks

```bash
scripts/with-env.sh pre-commit run --all-files
```

| Hook | 契约 |
|---|---|
| cuda-format | clang-format 23.1.3, 原生后端模块不超过 800 行 |
| cargo-fmt | Rust 格式化 |
| rust-line-width | Rust 行宽硬限制为 100 列 |
| definition-spacing | 通过语法解析检查 Python, Rust 和 JavaScript/TypeScript 函数及模块间的空行 |
| cargo-clippy | 所有 targets 和 features, 警告视为错误 |
| cargo-test | 已有 Rust workspace 测试 |
| ruff-format | Python 格式化 |
| ruff-check | Python lint 规则 |
| whitespace | 无行尾空白, 文件末尾保留一个换行 |
| documentation | 中英配对, 词数限制, 可移植路径, 链接及源码锚点 |
| glossary | 存在词表时, 生成内容须与词表源文件一致 |

Cargo hooks 覆盖自有 manifest, 每个 workspace 只执行一次.
无适用文件的 hooks 没有检查对象. 文档 hook 始终运行.
[测试策略](../CONTRIBUTING.zh.md)约束新增测试. 代理自行运行已有测试, 维护者无需手动操作.
