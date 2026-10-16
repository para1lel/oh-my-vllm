# 开发环境

开发使用 Python 3.12, Rust, uv, 以及 Node 24 或更新版本.
Conda 定义提供 Python 和 Rust.
使用 Python 和 JavaScript 锁文件中的 hook 依赖版本.

开发和运行时依赖使用不同文件.

## 安装

先安装 Conda, uv 和 Node.
执行以下命令:

```bash
conda env create -f environment.yml
conda activate oh-my-vllm
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/development.txt -r requirements/runtime.txt
npm --prefix development/javascript ci --ignore-scripts --no-audit --no-fund
scripts/with-env.sh pre-commit install
```

已有可用环境时, 执行最后四条命令.
wrapper 依次选择 `OH_MY_VLLM_CONDA_PREFIX`, `CONDA_PREFIX`, `VIRTUAL_ENV`, 仓库内的 `.venv`.

Python 和 Cargo 命令通过 `scripts/with-env.sh` 执行.
wrapper 配置可执行文件路径, Python 模块路径, worker 解释器和 Cargo 构建输出.

开发服务时安装[运行时包](../requirements/runtime.txt).

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
| kernel-contracts | 负载覆盖, 性能下界, 数值参考及 worker 约束 |
| ruff-format | Python 格式化 |
| ruff-check | Python lint 规则 |
| whitespace | 无行尾空白, 文件末尾保留一个换行 |
| documentation | 翻译配对, 词数限制, 主机路径, 链接和源码锚点 |
| glossary | 生成词表与源文件一致 |

Cargo hooks 包含所有自有 manifest. 每个 workspace 只处理一次.
无适用文件的 hook 没有检查对象. 文档 hook 始终运行.

新增测试前遵守[测试策略](../CONTRIBUTING.zh.md).
代理代维护者执行仓库中的测试.
