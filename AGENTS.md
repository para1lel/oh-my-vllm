# Collaboration rules

## Goal and constraints

Build a Rust inference framework for Qwen3.8-27B-FP8 on one B200 GPU.
Rust controls service, scheduling, and logical KV cache. Python controls GPU computation.
Use ZMQ DEALER and msgpack at the boundary.
Keep all existing features and numerical tolerances.

- Keep the production block size at 784 tokens.
- Keep 16 FA layers and 48 GDN layers with `mamba_cache_mode="align"`.
- Support 262144 total input and output tokens.
- Keep ordinary, MTP4, and DSpark boundary tests at batches 1, 2, and 4 free of OOM and recompute preemption.
- Keep each active criterion in [requirements](docs/requirements.md).
- Keep project builds, tests, and inference free of vLLM dependencies code, environments, and build caches.
- Use the isolated baseline collector only for independently authorized baseline work.
- Use CUDA for project kernels. Keep the pinned TileLang comparison.
- Keep formal operator cases, the harness, and summarized evidence in the repository.
- Keep temporary experiments and raw profiling traces in an external directory.
- Document future architectures, multiple GPUs, and other NVIDIA backends without untested support claims.
- Keep DSpark weights fixed. Use current-source paired measurements for its comparison with MTP4.

Use recent compatible stable dependencies in dependency order.
Pin the combination with satisfactory compatibility tests.
Install Python dependencies with `uv` in the project environment.
Use working system CUDA/compiler tools or environment tools. The host supplies the driver.

TileFoundry is a development dependency from the pinned fork.
Fix simple fork incompatibilities directly. Discuss substantial changes with the user first.

## Environment and GPU ownership

Read [development](docs/development.md) for portable environment selection.
Set `OH_MY_VLLM_MODEL` or supply `--model` before model execution.
Use `scripts/with-env.sh` for Cargo and Python commands.
The wrapper sets repository-relative `PYTHONPATH` and `CARGO_TARGET_DIR` defaults.
Keep host paths and maintenance records in ignored `LOCAL.md`.

Use `scripts/with-gpu.sh` to wait for an idle B200 and pin its UUID.
Stop task-owned GPU processes and children immediately after each run, failure, cancellation, or handoff.
Make sure of their exit with process records and `nvidia-smi`.
Stop only processes owned by the task.
Keep a service active only after an explicit user request.

Record that exception in `LOCAL.md` and its purpose in the handoff.

## Task procedure

1. Read `git status --short` and identify existing changes.
2. Make sure that the current branch is `main`.
3. Read [handoff](docs/handoff.md) and applicable directory rules.
4. Read the task guides in this table.
5. Make the authorized changes and applicable tests.
6. Start an review by another sub-agent after each milestone.
7. Correct correctness, performance, and engineering findings before the commit.
8. Update the handoff with results, verification limits, and open work.
9. Stage only intentional paths.
10. Commit on `main` with the format in [CONTRIBUTING](CONTRIBUTING.md).
11. Make sure that task-owned GPU workers and temporary service ports are released.

Do not create branches or branch-backed worktrees.
The existing TileFoundry adaptation branch is the user-authorized exception.
Do not open an upstream pull request for that fork.
Do not push or open pull requests without explicit instruction.
Do not use `git add .` or disable hooks.

Do not commit secrets, environment values, or raw GPU traces.

## Required checks

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/check_docs.py
```

Use applicable Python, GPU, tutorial, and acceptance checks from [testing](docs/testing.md).
Each new feature must have a test or a documented reason for its absence.
Use collected measurements for performance claims.
Identify skipped checks in the commit body.
Rust has a hard 100-character line limit. This includes macros and strings.

Each `unsafe` block must have a `// SAFETY:` comment.

## Documents

Maintain the English Markdown source and its full same-directory `<stem>.zh.md` translation in one change.
Agents use the English source as authoritative.
Keep commands, identifiers, numbers, evidence scope, and limitations equal in the two languages.
Exclude third-party submodules and generated or vendored dependencies.
Follow [writing rules](docs/writing.md) and the [technical glossary](docs/glossary.md).

| Change | Documents |
|---|---|
| Requirement | `docs/requirements.md` |
| Architecture | `docs/architecture.md` |
| Key decision | `docs/decisions/` |
| Task completion or blocker | `docs/handoff.md` |
| Build or test command | `docs/development.md`, `docs/testing.md` |
| Commit convention | `CONTRIBUTING.md` |

| Task | Read first |
|---|---|
| Any task | This file and `docs/handoff.md` |
| Scheduler or KV cache | `docs/architecture.md`, `docs/decisions/` |
| Python runner | Python ownership in `docs/architecture.md` |
| Performance | `docs/requirements.md`, `docs/profiling.md`, `docs/testing.md` |
| New feature | `docs/requirements.md`, `docs/architecture.md` |
| Failure | Open work in `docs/handoff.md`, `docs/testing.md` |
| Audit finding | `docs/audit.md` |
