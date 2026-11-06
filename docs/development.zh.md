# 开发环境

开发使用 Python 3.15.2, Rust, pip, 以及 Node 24 或更新版本.
Conda 定义提供 Python 和 Rust.
`requirements/runtime.txt` 提供推理包.
`requirements/development.txt` 提供开发包, 并包含 runtime 文件.
主机设置保存在被忽略的 `LOCAL.md` 中.

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
scripts/with-env.sh python scripts/check.py doctor
```

已有可用环境时, 从 export 命令开始执行.
wrapper 只接受 `OH_MY_VLLM_CONDA_PREFIX` 和 `oh-my-vllm` Conda environment.
`PYZMQ_BACKEND=cffi` 使 CPython 3.15 binding 使用所选 Conda ZeroMQ library 构建.
AgentX wheel 有过时的 Python 上限. 安装时使用 `--ignore-requires-python`.

Python 和 Cargo 命令通过 `scripts/with-env.sh` 执行.
wrapper 配置可执行文件路径, Python 模块路径, worker 解释器和 Cargo 构建输出.
仓库命令使用 Conda environment.
development 安装包含 runtime 包.

## 开发循环

使用[检查 CLI](../scripts/check.py) 选择检查层级:

```bash
scripts/with-env.sh python scripts/check.py --help
```

| wrapper 和 Python 后的命令 | 范围 |
|---|---|
| `scripts/check.py doctor` | 工具, 指定的 parser 版本, parser 运行, 已安装的 Git hook |
| `scripts/check.py size` | 全部仓库代码文件和函数范围 |
| `scripts/check.py static` | 只读尺寸, 格式, lint, 间距, 空白, 文档及词表检查 |
| `scripts/check.py cpu` | 禁用 CUDA 的 Rust workspace 测试及发现的 Python 测试 |
| `scripts/check.py hooks` | 全部已配置的 pre-commit hooks, 含其常规修正 |

修改期间使用 `static`.
对修改的行为运行定向 CPU 检查:

```bash
scripts/with-env.sh python scripts/check.py cpu --list
scripts/with-env.sh python scripts/check.py cpu --python-only tests.test_constraints
```

第一条命令列出唯一的 Python test IDs, 不运行测试或 Cargo.
第二条命令运行所选 Python 模块.
完整 CPU 范围使用不带选项的 `cpu`.
Cargo 为 Python HTTP 服务用例提供可执行文件. Python discovery runner 不重复这些用例.

使用[验证技能](../.claude/skills/verify-oh-my-vllm/SKILL.zh.md)驱动可见的 CLI 和 HTTP 行为.
其中的 CPU fixture 只检查服务和传输.
它不证明 GPU 推理, 数值正确性, 模型容量或性能.
使用[测试范围](testing.zh.md)选择模型验收检查.
最终修改后, 为交付运行全部 hooks.
修改或失败产生新的理由时, 再重复检查.

## 代码尺寸

全部仓库代码文件的上限为 500 个物理行.
全部函数的上限为 80 个物理行.
限制适用于生产代码, 测试, 工具, CUDA headers 和隐藏的技能 helper.
[尺寸检查器](../scripts/check_code_size.py)使用语法树确定函数范围.
范围内的空行和注释计入物理行数.
检查器接受 LF 和 CRLF 换行, 并拒绝后面没有 LF 的 CR.

使用 inventory 查看每个文件, 函数, 行范围和违规项:

```bash
scripts/with-env.sh python scripts/check.py size --inventory --json > "$ARTIFACT_ROOT/code-size.json"
scripts/with-env.sh python scripts/check_code_size.py --all
```

执行 inventory 命令前, 将 `ARTIFACT_ROOT` 设置为外部存储.
Inventory 模式报告尺寸违规, 但不因此返回失败退出码.
解析错误和缺少工具会导致失败.
未选择 inventory 模式时, 尺寸检查遇到尺寸违规会返回失败.

超过限制时, 按职责拆分代码.
调用方有依赖时, 保持公开导入和 CLI 入口稳定.
使用输入和输出清晰的小函数.
保持已有数值顺序, 资源所有权, 错误行为及冻结验收值.

## Git hooks

```bash
scripts/with-env.sh python scripts/check.py hooks
```

[Pre-commit 配置](../.pre-commit-config.yaml)是 hook 来源.
hooks 检查代码尺寸, 格式, lint, Rust 行为, Python 契约, 文档及词表一致性.
完整运行包含 Clippy 和测试.
static 层级不运行 Clippy 和测试.
hooks 可能修改格式化文件及源码链接行号.
暂存前审查这些修改.

保持 hooks 启用.
审查和 Git 交付使用[贡献流程](../CONTRIBUTING.zh.md#交付).
原始日志和证据保存在仓库外部.
检查后停止自有 worker 和临时服务.

## 原生模块

原生构建使用 CUDA 13.4 toolkit.
在本地配置中将 `CUDA_HOME` 设为其安装目录.
development 文件安装原生构建包, 并包含 `requirements/runtime.txt`.
如果没有 CUDA 13.4, 在原生 rebuild 前停止.
不要将 CUDA 13.2 headers 与 CUDA 13.4 Torch runtime 混用.

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
它根据构建 manifest 检查源码和二进制身份.
每个二进制在外部 cache 中具有不可变副本和 manifest.
源码移动可能改变这些身份. Checkpoint 推理前先构建已修改模块.
启动使用[服务指南](serving.zh.md), 验收使用所选模型指南.
