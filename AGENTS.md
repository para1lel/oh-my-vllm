# AGENTS.md — oh-my-vllm Collaboration Rules

## Project goal (user-confirmed)

Build a Rust-first inference framework for **Qwen3.5-27B-FP8 on a single B200 GPU**
that matches vLLM EngineCore throughput at least **95%** on the workloads in
`docs/requirements.md`. Rust owns scheduling and KV cache; Python wraps vLLM's
`GPUWorker` for the model runner. The two sides talk over a ZMQ DEALER socket
using msgpack.

## Non-negotiable constraints

- **Target model:** `/data0/shared/Qwen3.8-27B-FP8` (Qwen3.5-27B, 48 GDN + 16 FA layers)
- **Target hardware:** single B200 GPU, wait for any idle B200 with `scripts/with-gpu.sh`; pin the selected GPU UUID
- **GPU cleanup:** Promptly stop GPU programs started for the task as soon as
  their tests/runs finish, including inference servers and child workers. Do not
  leave them running after completion, failure, cancellation or handoff unless
  the user explicitly asks to keep them running. Verify owned processes have
  exited and no longer appear in `nvidia-smi`; never stop unrelated processes.
- **Performance target:** at least 95% of vLLM EngineCore on bs=1/2/4, in=32768, out=4096
- **Python env for model runner:** `/data0/shared/dongwu.chen/conda-envs/vllm/bin/python`
  — this env has vLLM, torch, CUDA. The oh-my-vllm Python package is not installed
  there; set `PYTHONPATH=/data0/shared/dongwu.chen/oh-my-vllm/python:$PYTHONPATH`
- **Rust env:** `/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/cargo`
  (`CARGO_TARGET_DIR=/data0/shared/dongwu.chen/oh-my-vllm/target`)
- **Block size:** 784 tokens (Qwen3.5 hybrid requirement — do not change)
- **KV groups:** FA group (16 layers) + Mamba group (48 layers, `mamba_cache_mode="align"`)
- **Attribution line on every commit:**
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- **Reply language:** Chinese when user writes Chinese, English otherwise
- **Branch policy:** Develop and commit only on `main`. Do not create new local
  or remote branches, including feature branches and branch-backed worktrees.

## Forbidden actions

- Do not change `block_size` away from 784
- Do not swap-in vLLM's own scheduler (the whole point is Rust owns scheduling)
- Do not add Python dependencies via conda — use pip inside the vllm env
- Do not `git push` or open PRs without explicit instruction
- Do not `git add .` — stage only the files you intentionally changed
- Do not disable pre-commit hooks (`--no-verify`)
- Do not commit secrets, `.env` values, or GPU profiling traces

## Task start checklist

1. `git status --short` — note any unstaged changes before touching anything
   Confirm the current branch is `main` before making changes.
2. Check `docs/handoff.md` for current task state and blockers
3. Read the relevant sub-directory `AGENTS.md` if one exists (none yet, but check)
4. Set the two environment variables above before any `cargo` or `python` invocation

## Task end checklist

1. `cargo test --workspace` must pass (Rust)
2. `ruff format python/ && ruff check python/` must pass (Python)
3. `cargo clippy --all-targets --all-features -- -D warnings` must pass
4. Stage only changed files; never `git add .`
5. Start a sub-agent code review after each milestone; fix correctness, performance, and best-practice findings before committing with Conventional Commits format + attribution line
6. Update `docs/handoff.md` with what was done, what was verified, what remains
7. Stop all task-owned GPU servers/workers and verify GPU resources and temporary
   service ports are released. Record any explicitly requested running service
   exception in the handoff; do not infer one from a request to test or run it.

## Verification requirements

- Any new feature needs a test or documented reason why one is impractical
- Benchmark numbers (tok/s) must come from actual runs, not estimates
- If a test is skipped or a check is not run, say so explicitly in the commit body

## Engineering standards

See `CONTRIBUTING.md` for full rules. Short version:
- Rust: `rustfmt` + `clippy -D warnings`; `unsafe` blocks need a `// SAFETY:` comment
- Python: `ruff format` + `ruff check` (config in `ruff.toml`); Python ≥ 3.12
- All hooks defined in `.pre-commit-config.yaml` run via `scripts/with-env.sh`

## Document update triggers

| Event | Update these |
|---|---|
| Requirements change | `docs/requirements.md` |
| Architecture changes | `docs/architecture.md` |
| New ADR-worthy decision | `docs/decisions/` |
| Task completes or blocks | `docs/handoff.md` |
| Build/test commands change | `docs/development.md`, `docs/testing.md` |
| Commit/PR conventions change | `CONTRIBUTING.md` |

## Minimum reading path by task type

| Task | Read first |
|---|---|
| Any task | This file + `docs/handoff.md` |
| Rust scheduler/KV cache | `docs/architecture.md` + `docs/decisions/` |
| Python model runner | `docs/architecture.md` section "Python side" |
| Performance work | `docs/requirements.md` (REQ-PERF-*) + `docs/profiling.md` |
| New feature | `docs/requirements.md` + `docs/architecture.md` |
| Debug a failure | `docs/handoff.md` "Known issues" + `docs/testing.md` |
