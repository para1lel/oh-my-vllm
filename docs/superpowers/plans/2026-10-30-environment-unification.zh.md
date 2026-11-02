# Environment Unification Implementation Plan

> **执行说明** - Use superpowers: executing-plans to implement this plan task-by-task.

**目标:**  让 repository code 只使用 `oh-my-vllm` environment. 使用一个 `requirements/development.txt`, Python 3.15.2 和官方 PyTorch 2.15.0 CUDA 13.4 wheel.

**架构:**  Conda 管理 Python 和 Rust. 一个 pip requirements file 管理 runtime, development, kernel 和 AgentX packages. `requirements/model-reference.lock` 只用于明确的 reference generation. vLLM/SGLang 使用 repository 外部的专用 environment.

**技术栈:**  Conda, Python 3.15.2, PyTorch 2.15.0+cu134, CUDA 13.4 wheel index, CUDA native compiler, pip, Rust, pre-commit.

**规格:**  `docs/superpowers/specs/2026-10-30-torch-native-runtime-design.md`

## 全局约束

- 只使用一个 install file: `requirements/development.txt`.
- repository code 使用 `oh-my-vllm` Conda environment.
- 目标是 Python 3.15.2 和 CUDA 13.4 wheel 的 PyTorch 2.15 或更新版本.
- repository tests, service execution, native builds 和 AgentX 不使用第二个 environment.
- 明确的 reference generation 和 vLLM/SGLang runs 可以使用文档指定的特殊 environment.
- host paths 放在 ignored `LOCAL.md`.
- 不删除或 reset 无关的 worktree changes.

## Review Focus

- 非 Conda command 必须失败. 不得回退到 `.venv`. Task 1 测试 `scripts/with-env.sh` 的 valid 和 invalid environment.
- resolver 必须为 CPython 3.15 选择 `torch==2.15.0+cu134`. Task 2 检查 version, CUDA runtime 和 wheel metadata.
- AgentX 必须使用同一个 Conda interpreter. Task 3 检查 launcher command identity.
- Native build identity 分别记录 CUDA toolkit 和 PyTorch CUDA runtime. Task 4 检查 manifest fields.
- Documentation 不得保留 split requirements instructions. Task 5 检查 repository references 和 bilingual documentation.

### Task 1: Enforce the repository Conda environment

**文件:**
- Modify: `scripts/with-env.sh:1-20`
- Test: `tests/test_environment.py`
- Modify: `LOCAL.md` only when local host values change

**接口:**
- `OH_MY_VLLM_CONDA_PREFIX` 是唯一的 repository interpreter selection.
- 缺少 prefix, non-Conda prefix 和名称不是 `oh-my-vllm` 的 environment 都失败.

- [ ] 为 valid `oh-my-vllm`, missing prefix 和 wrong environment name 写 tests.
- [ ] 运行 `scripts/with-env.sh` tests. 两个 invalid cases 必须在 child command 前失败.
- [ ] 修改 `scripts/with-env.sh`. 强制使用 Conda prefix, 检查 `conda-meta` 和存在时的 `CONDA_DEFAULT_ENV`.
- [ ] 运行 environment tests 和打印 `sys.executable` 的 shell command. 输出必须来自 selected Conda prefix.
- [ ] 保留现有 worker 和 Cargo path exports.

### Task 2: Replace split requirements with one CUDA 13.4 requirements file

**文件:**
- Modify: `requirements/development.txt`
- Delete: `requirements/agentx.txt`
- Delete: `requirements/kernel-development.txt`
- Delete: `requirements/kernels.txt`
- Delete: `requirements/runtime.txt`
- Delete: `requirements/model-reference.txt`
- Keep: `requirements/model-reference.lock` for explicit reference generation only
- Modify: `environment.yml:1-6`

**接口:**
- `requirements/development.txt` 包含 runtime, development, kernel build, AgentX, `torch==2.15.0+cu134` 和 `apache-tvm-ffi` dependencies.
- CUDA wheel source 是官方 `https://download.pytorch.org/whl/cu134/` index.

- [ ] 先加入 merged direct dependencies 和 `torch==2.15.0+cu134`.
- [ ] 使用 Python 3.15 做 resolver dry-run. 不得依赖 split file.
- [ ] 将 `environment.yml` 更新为 Python 3.15.2. 保留 Rust toolchain.
- [ ] 删除 obsolete split `.txt` files. 不删除 reference lock file.
- [ ] 使用 `conda` 创建或更新 Conda environment, 使用环境内的 `python -m pip` 安装 Python packages.
- [ ] 检查 `python --version`, `torch.__version__`, `torch.version.cuda` 和 `importlib.metadata.version("apache-tvm-ffi")`.

### Task 3: Make AgentX use the repository interpreter

**文件:**
- Modify: `docs/models/qwen38/agentx.md`
- Modify: `docs/models/qwen38/agentx.zh.md`
- Modify: `docs/models/deepseek_v41/agentx.md`
- Modify: `docs/models/deepseek_v41/agentx.zh.md`
- Modify: `development/serving/agentx.py` and its tests if it creates or validates an external interpreter

**接口:**
- AgentX 接收 `$OH_MY_VLLM_CONDA_PREFIX/bin/python`.
- `ARTIFACT_ROOT` 仍是 external directory, 用于 raw traces.

- [ ] 增加 launcher identity assertion. AgentX interpreter 必须等于 repository interpreter.
- [ ] 从两个 language documents 删除 `AGENTX_ENV` creation 和 installation instructions.
- [ ] 运行 AgentX launcher unit tests 和 dry-run identity check.

### Task 4: Record wheel and native toolkit identities

**文件:**
- Modify: `development/native_build.py`
- Modify: `python/oh_my_vllm/operators/runtime/native.py`
- Test: `tests/test_native_identity.py`

**接口:**
- Native manifests 包含 `python_version`, `torch_version`, `torch_cuda_version`, `cuda_home` 和 `cuda_compiler`.
- Load 在 Torch CUDA runtime 或 compiler identity 不同的时候失败.

- [ ] 写 matching 和 mismatching identity tests.
- [ ] 先运行 tests, 检查预期 mismatch failures.
- [ ] 增加 identity collection 和 validation. 不改变 kernel source fingerprints.
- [ ] 使用 selected compiler toolkit 和 Torch 2.15 CUDA 13.4 runtime 重建 native modules.
- [ ] 运行 native identity 和 operator smoke tests.

### Task 5: Update instructions and remove stale references

**文件:**
- Modify: `docs/development.md`
- Modify: `docs/development.zh.md`
- Modify: `docs/serving.md`
- Modify: `docs/serving.zh.md`
- Modify: `development/serving/benchmark.py`

**接口:**
- All repository install commands 只引用 `requirements/development.txt`.
- Benchmark source identity hash `requirements/development.txt`.

- [ ] 替换 English 和 Chinese documents 中的 split installation commands.
- [ ] 更新 benchmark identity, 使其 hash single requirements file.
- [ ] 运行 documentation, whitespace 和 source-reference checks.
- [ ] 搜索 repository. 不得有 deleted `.txt` files 或 `AGENTX_ENV` 的 active command reference.

## Delivery

- 在 new Conda environment 中运行 CPU and GPU unit tests.
- 运行 native build checks 和 Qwen/DeepSeek numerical smoke checks.
- 每次 delivery 前获取 independent review.
- 只 commit environment files 和本计划的 documentation.
- Review 后 push `nightly`.
