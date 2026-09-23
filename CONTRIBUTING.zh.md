# 参与 oh-my-vllm 开发

## 分支策略

直接在 `main` 上开发和提交。不要创建本地或远程功能分支，也不要创建依附分支的 worktree。编辑前确认已检出 `main`。推送和提 PR 仍需用户明确指令（见 AGENTS.md）。唯一的分支例外是用户批准的 TileFoundry submodule fork：适配分支属于 para1lel/TileFoundry，不向上游提 PR。

## 提交约定

每条提交消息遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <short description>

[optional body]

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

AI 协助完成的每次提交都必须包含署名行。类型：`feat`、`fix`、`refactor`、`test`、`docs`、`chore`、`perf`。scope 与 crate 或模块对应：`kv-cache`、`scheduler`、`zmq-worker`、`zmq-bridge`、`spec-decode`、`bench`、`docs`。

示例：

```
feat(scheduler): add preemption by recompute

fix(zmq-bridge): remove swap_space from CacheConfig, add mamba_cache_mode

docs(handoff): add full engineering handoff documentation suite
```

## 提交前

每个里程碑启动子 agent 代码审查，检查最佳实践及正确性/性能风险。修复问题、审查通过后再提交。重写模块时同步架构/决策文档。

逐项暂存有意修改的路径；`Cargo.lock` 有变化时一并暂存：

```bash
git add Cargo.lock <your other changed files>
```

pre-commit hook 自动执行：

- `cargo fmt --all --check` 与 `cargo clippy --all-targets --all-features -- -D warnings`。
- `python scripts/check_rust_line_width.py`：对全部已跟踪及未被忽略的新 Rust 文件检查 `rustfmt.toml` 规定的硬性行宽，包括宏体。
- `cargo test --all`：完整测试必须通过。
- 对暂存的 Python 文件运行 `ruff format` 和 `ruff check --fix`。
- `scripts/fix_whitespace.py`。

hook 失败时修复所报问题并重新暂存，不要使用 `--no-verify`。常规格式问题用 `scripts/with-env.sh cargo fmt --all` 修复。宏体和长字面量可能需手工换行：`rustfmt` 无法可靠格式化任意宏语法，其 `max_width` 也不是硬性校验规则。

## 代码标准

每份项目自有 Markdown 文档都在同目录维护 `<stem>.zh.md` 中文版。英文修改时同步翻译，包括需求、命令、证据和任务状态。agent 以英文版为准。第三方 submodule 和生成的依赖除外。

**Rust：** 遵循 `rustfmt.toml` 和每行 100 字符硬限制（展开制表符），包含宏、字符串和注释。JSON 字段分别占行，长字面量使用 `concat!`。没有解释保留理由的注释时，不得使用 `#[allow(dead_code)]` 或 `#[allow(unused)]`。所有 `unsafe` 块必须有说明所维持不变量的 `// SAFETY:` 注释。

**Python：** 用 `ruff` 约束格式及 lint（PEP 8 和选定规则）。禁止裸 `except:`。全部公开函数签名添加类型提示。

## 完成定义

任务完成需满足：

1. 代码已提交，pre-commit hook 全部通过。
2. 受影响测试通过：调度器/KV 缓存修改运行 Rust 单元测试；注意力/状态修改运行实际路径 GQA/GDN FP64 探针，涉及 MTP 时也覆盖 MTP。
3. 改动涉及 `zmq_bridge.py`、`model_runner.py` 或 `client.rs` 时，端到端冒烟测试通过。
4. `docs/handoff.md`（阶段完成时还有 `docs/plan.md`）已反映新任务状态；修复审计问题时，同时在 `docs/audit-2026-09-23.md` 中更新其状态。
5. 运行结束后及时停止任务自有 GPU 服务器和 worker，失败或取消时同样如此。确认进程、GPU 分配和临时服务端口均已释放。只有用户明确要求时才保留服务，并在交接记录中注明例外。

## 添加新功能

1. 在 `docs/requirements.md` 添加或更新对应 `REQ-*`。
2. 编写 Rust 或 Python 代码。
3. 添加单元测试：Rust 放在同文件的 `#[cfg(test)]` 模块，Python 放在 `tests/`。
4. 在 `docs/testing.md` 更新测试位置和预期结果。
5. 按上述约定使用 `feat(...)` 提交。

## 手工运行测试

```bash
# 框架检查会设置两个必要的环境变量。
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/

# 连贯文本；实际路径 FP64 变体见 docs/testing.md。
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/contribution-smoke.ipc --max-tokens 64
```

## 不要做的事情

- 不要添加基于 swap 的抢占、多 GPU、gRPC 服务、LoRA 或多模态输入；它们明确不在范围内（REQ-OUT-SCOPE-001）。需要调整范围时先讨论。OpenAI 兼容 HTTP 服务已实现并通过验收（REQ-SERVE-001）；修改服务时须重跑其真实 GPU/agentic 检查。
- 不得提交秘密信息、token、`.env` 值或生产凭据。
- 未获项目负责人明确确认，不得向任何分支强制推送。
- 升级 Rust `zeromq` crate 前，检查 `DealerSocket` 和 `ipc-transport` 是否仍正常，见 `docs/decisions/ADR-001-zmq-socket-type.md`。
