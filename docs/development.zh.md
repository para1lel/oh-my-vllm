# 开发环境

使用支持 CUDA 的 Linux 主机, 一张空闲 B200, Python 3.12 或更新版本, Rust, 以及教程所需的 Node 24 LTS.
NVIDIA driver 由主机提供. 模型权重与源码分开保存.

## 安装

初始化固定的原生头文件与开发源码:

```bash
git submodule update --init --recursive
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/runtime.txt -r requirements/tools.txt
uv pip install --python .venv/bin/python --no-deps -e python
```

[运行时元数据](../python/pyproject.toml) 声明依赖, [运行时 lock](../requirements/runtime.txt) 固定依赖闭包.
[工具输入](../requirements/tools.in)与[工具 lock](../requirements/tools.txt) 将维护工具分开管理.
重新生成工具版本时使用运行时约束并执行兼容性检查.

CUDA C++ 使用固定的 [CUTLASS 源码](../3rdparty/cutlass/)以及所选 CUDA toolkit 的 CCCL 头文件.
Python 算子库不属于运行依赖.
当前原生迁移仍需完成[交接记录](handoff.zh.md)中的最终 GPU 验收.

## 配置选择

Cargo 与 Python 命令使用 `scripts/with-env.sh`.
以其[环境选择与默认值](../scripts/with-env.sh)为准.
已有环境可设置 `OH_MY_VLLM_CONDA_PREFIX`, 或激活 virtual/conda 环境.
否则 wrapper 选择仓库 `.venv`.
所选环境不存在时, 执行命令前即失败.

模型执行前设置 `OH_MY_VLLM_MODEL` 或传入 `--model`.
DSpark 设置 `OH_MY_VLLM_DRAFT_MODEL` 或传入 `--draft-model`.
主机路径与维护记录放在被忽略的 `LOCAL.md` 中.

## GPU 所有权

```bash
scripts/with-gpu.sh scripts/with-env.sh python -m pytest tests/numerical/test_ir_fp8.py -q
```

Wrapper 等待空闲 B200, 获取锁并暴露其 UUID.
每次运行后停止任务拥有的 GPU 进程及子进程, 确认退出与显存释放.
使用进程出生记录. 不停止其他用户的进程.
仅用户明确要求时才保留运行中的服务.

## 构建与检查

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/with-env.sh pre-commit install
scripts/with-env.sh pre-commit run --all-files
scripts/test.sh cpu
```

局部与完整运行见[测试分层和 GPU 选择](testing/suites.zh.md).
Hooks 检查格式, 定义空行, Rust 行宽, 双语文档, 单词上限和源码引用.
保持 hooks 启用, commit 前配置其环境.

## TileFoundry 开发工具

```bash
scripts/with-env.sh uv pip install setuptools-scm==10.2.3 vcs-versioning==2.4.1
scripts/with-env.sh uv pip install --no-build-isolation -r requirements/development.txt
scripts/with-env.sh tilefoundry --help
```

[固定 fork](../3rdparty/TileFoundry/) 已 rebase 到当前上游, 保留项目兼容性修改.
它属于开发依赖. 生产原生头文件独立固定版本.
没有指示时不 push, 不开上游 pull request.
调优与独立数学验证见[算子 harness](kernels.zh.md).

## 部署记录

主机配置放在 `LOCAL.md`, 临时工作与原始 trace 放在仓库外, 可移植摘要放在[证据索引](acceptance.zh.md).
新环境须使用自己的 checkpoint, GPU 选择和有源码范围的测量.
历史结果不证明当前验收通过.
