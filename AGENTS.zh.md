# AGENTS.md — oh-my-vllm 协作规则

本文件为中文译本，供用户阅读；agent 阅读并维护英文版，以英文版为准。

## 项目目标（用户已确认）

构建以 Rust 为主的推理框架，在**单张 B200 GPU 上运行 Qwen3.8-27B-FP8**，在 `docs/requirements.md` 规定的工作负载上达到 vLLM EngineCore 吞吐的至少 **95%**。Rust 负责服务、调度和逻辑 KV 缓存，Python 负责 GPU 计算。两端使用 ZMQ DEALER 和 msgpack。

运行时采用项目自有实现及独立库，过渡期 vLLM 适配器已移除。构建、测试和推理不得安装、导入或链接 vLLM，不得使用其源码检出目录，也不得依赖旧 vllm conda 环境或构建缓存。

## 不可违反的约束

- **目标模型：** `/data0/shared/Qwen3.8-27B-FP8`（Qwen3.8-27B，48 层 GDN + 16 层 FA）。
- **目标硬件：** 单张 B200 GPU；通过 `scripts/with-gpu.sh` 等待任意空闲 B200，并固定所选 GPU UUID。
- **GPU 清理：** 任务启动的 GPU 程序在测试/运行结束后立即停止，包括推理服务器及子 worker。完成、失败、取消或交接后都不能遗留，除非用户明确要求保持运行。确认自有进程退出且不再出现在 `nvidia-smi` 中；不得停止无关进程。
- **性能目标：** bs=1/2/4、输入 32768、输出 4096 时，吞吐至少达到 vLLM EngineCore 的 95%。
- **目标 Python 环境：** `/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python`。设置 `PYTHONPATH=/data0/shared/dongwu.chen/oh-my-vllm/python:$PYTHONPATH`。
- **依赖：** 按依赖顺序选择较新且兼容的稳定版本，验证后固定组合。高层依赖放在 oh-my-vllm 环境中。方便时使用可正常工作的系统 CUDA/编译工具，否则放在该环境中维护。驱动由宿主机提供。
- **范围：** 保留全部现有功能。未来模型架构、单机多 GPU、其他 NVIDIA 后端及本地 DSpark checkpoint 只需简要说明扩展方式，不添加空接口，也不声称支持未经测试的能力。
- **回归基线（2026-09-22）：** 用户授权将隔离的 vLLM 仓库/环境更新到官方 main 并冻结 SHA，然后重测全部 12 组 TTFT 和吞吐。旧产物作为历史保留。仅基线工具可以使用隔离的 vLLM 环境；项目构建、测试和推理必须保持独立。每组要求吞吐 >=95%、TTFT <= 新基线的 110%。至少完整预热两次、测量五次；极差/中位数 >5% 时调查并重测。
- **上下文：** 支持输入与输出合计最多 262144 token。普通模式及 MTP4 的 batch 1/2/4 边界测试必须无 OOM、无重计算抢占地完成。按需优化显存布局，保留 block784 和现有正确性。
- **Rust 环境：** `/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/cargo`；`CARGO_TARGET_DIR=/data0/shared/dongwu.chen/oh-my-vllm/target`。
- **块大小：** 784 token（Qwen3.8 混合模型要求，不可更改）。
- **KV 分组：** FA 组（16 层）+ Mamba 组（48 层，`mamba_cache_mode="align"`）。
- **每次提交的署名行：** `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`。
- **回复语言：** 用户用中文时回复中文，否则回复英文。
- **分支策略：** 只在 `main` 开发和提交。不创建任何本地/远程新分支，包括功能分支及依附分支的 worktree。用户授权的例外：TileFoundry fork 使用适配分支，并固定为 `3rdparty/TileFoundry`；不得向上游提 PR。
- **Kernel 迁移：** 将项目自有 Triton kernel 全部替换为 TileLang，保留现有正确性测试和容差。第三方实现不在范围内。TileFoundry 是同一 conda 环境中的开发专用源码依赖。fork 中简单的不兼容问题直接修复；重大修复先与用户讨论。
- **临时 kernel 实验：** 微基准/测试脚本及记录放在仓库外。只保留用户要求或文档规定的测试和正式验收证据。不增加相对 Triton 的验收门槛。

## 禁止事项

- 不得将 `block_size` 改为 784 以外的值。
- 不得换用 vLLM 自有调度器（Rust 拥有调度权是项目核心）。
- 不得在 oh-my-vllm 环境中用 pip 安装 Python 依赖。
- 没有明确指令时不得 `git push` 或提 PR。
- 不得 `git add .`；只暂存有意修改的文件。
- 不得用 `--no-verify` 禁用 pre-commit hook。
- 不得提交秘密信息、`.env` 值或 GPU profiling trace。

## 任务开始检查

1. 执行 `git status --short`，修改前记录已有未暂存改动；确认当前分支为 `main`。
2. 查看 `docs/handoff.md` 中的当前任务状态和阻塞项。
3. 若相关子目录有 `AGENTS.md`，先阅读（当前尚无，仍需检查）。
4. 任何 `cargo` 或 `python` 调用前设置上述两个环境变量。

## 任务结束检查

1. Rust 的 `cargo test --workspace` 必须通过；还需通过 `scripts/with-env.sh` 运行 `cargo fmt --all --check` 和 `python scripts/check_rust_line_width.py`。
2. Python 的 `ruff format python/ && ruff check python/` 必须通过。
3. `cargo clippy --all-targets --all-features -- -D warnings` 必须通过。
4. 只暂存修改的文件，禁止 `git add .`。
5. 每个里程碑后启动子 agent 代码审查；解决正确性、性能及最佳实践问题后，以 Conventional Commits 格式和规定署名提交。
6. 更新 `docs/handoff.md`：完成了什么、验证了什么、还剩什么。
7. 停止所有任务自有 GPU 服务器/worker，验证 GPU 资源和临时服务端口释放。用户明确要求保持服务运行的例外需记入交接文档；不能从“测试/运行”的要求推导出保持运行授权。

## 验证要求

- 每个新功能需要测试，或在文档中解释为何不适合测试。
- 吞吐数值（tok/s）必须来自实际运行，不能估算。
- 跳过测试或没有运行某项检查时，在提交正文中明确说明。

## 工程标准

完整规则见 CONTRIBUTING.md，摘要如下：

- Rust：使用 `rustfmt.toml`、`cargo fmt --all --check` 和 `clippy -D warnings`。pre-commit 强制每行最多 100 字符（展开制表符），包含宏体、字符串和注释。长 JSON 宏拆成字段，长字面量用 `concat!` 拆分；仅 rustfmt 成功并不足够。`unsafe` 块需有 `// SAFETY:` 注释。
- Python：使用 `ruff format` 和 `ruff check`，配置在 `ruff.toml`；Python ≥3.12。
- `.pre-commit-config.yaml` 中的所有 hook 都通过 `scripts/with-env.sh` 运行。

## 文档更新触发条件

项目自有 Markdown 文档在同目录提供 `<stem>.zh.md` 中文译本。英文源文档修改时，在同一次改动中创建或更新中文版本。保留命令、标识符、数值和证据限制，翻译时不得将历史结果当成当前结果。agent 阅读和维护英文权威版本；中文版供用户阅读。第三方 submodule、生成文件和 vendored 依赖不在范围内。不要为已有 `.zh.md` 再生成译本。

| 事件 | 需要更新 |
|---|---|
| 需求变化 | `docs/requirements.md` |
| 架构变化 | `docs/architecture.md` |
| 值得记录为 ADR 的新决策 | `docs/decisions/` |
| 任务完成或阻塞 | `docs/handoff.md` |
| 构建/测试命令变化 | `docs/development.md`、`docs/testing.md` |
| 提交/PR 约定变化 | `CONTRIBUTING.md` |

## 不同任务的最少阅读路径

| 任务 | 优先阅读 |
|---|---|
| 任意任务 | 本文件 + `docs/handoff.md` |
| Rust 调度器/KV 缓存 | `docs/architecture.md` + `docs/decisions/` |
| Python 模型 runner | `docs/architecture.md` 的 “Python side” |
| 性能工作 | `docs/requirements.md` 的 REQ-PERF-* + `docs/profiling.md` |
| 新功能 | `docs/requirements.md` + `docs/architecture.md` |
| 排查失败 | `docs/handoff.md` 的 “Known issues” + `docs/testing.md` |
