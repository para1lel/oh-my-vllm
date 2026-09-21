# Contributing to oh-my-vllm

## Branch policy

Develop and commit directly on `main`. Do not create local or remote feature
branches or branch-backed worktrees. Confirm `main` is checked out before editing.
Pushing and opening PRs still require explicit user instructions (AGENTS.md).

## Commit conventions

Every commit message must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

The attribution line is required on every commit made with AI assistance. Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`. Scope matches the crate or module: `kv-cache`, `scheduler`, `zmq-worker`, `zmq-bridge`, `spec-decode`, `bench`, `docs`.

Examples:
```
feat(scheduler): add preemption by recompute

fix(zmq-bridge): remove swap_space from CacheConfig, add mamba_cache_mode

docs(handoff): add full engineering handoff documentation suite
```

## Before committing

At every milestone, start a sub-agent code review for best practices and
correctness/performance risks. Fix findings and obtain a passing review before
committing. Update architecture/decision documents when rewriting modules.

Stage each intentional changed path; include Cargo.lock when it changed:

```bash
git add Cargo.lock <your other changed files>
```

The pre-commit hooks run automatically:
- `cargo fmt --all` and `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all` (the complete suite must pass)
- `ruff format` and `ruff check --fix` on staged Python files
- `scripts/fix_whitespace.py`

If a hook fails, fix the reported issue and re-stage. Do not use `--no-verify`.

## Code standards

**Rust:** follow `rustfmt` defaults (enforced by hook). No `#[allow(dead_code)]` or `#[allow(unused)]` without a comment explaining why the item must be kept. All `unsafe` blocks must have a `// SAFETY:` comment stating the invariant being upheld.

**Python:** `ruff` enforces formatting and linting (PEP 8 + selected rules). No bare `except:`. Type hints on all public function signatures.

## Definition of done

A task is done when:
1. The code change is committed and the pre-commit hooks pass.
2. Affected tests pass (Rust unit tests for scheduler/kv-cache changes; actual-path GQA/GDN FP64 probes for attention/state changes, including MTP when affected).
3. The end-to-end smoke test passes if the change touches `zmq_bridge.py`, `model_runner.py`, or `client.rs`.
4. `docs/plan.md` and `docs/handoff.md` are updated to reflect the new task status.
5. Task-owned GPU servers and workers are stopped promptly after their runs,
   including on failure or cancellation. Verify their processes, GPU allocations
   and temporary service ports are gone. Leave a service running only when the
   user explicitly requests it, and document that exception in the handoff.

## Adding a new feature

1. Add or update the relevant `REQ-*` entry in `docs/requirements.md`.
2. Write the Rust or Python code.
3. Add unit tests (Rust: `#[cfg(test)]` module in the same file; Python: `tests/` directory).
4. Update `docs/testing.md` with the new test location and expected result.
5. Commit with a `feat(...)` message following the convention above.

## Running tests manually

```bash
# Framework checks set both required environment variables.
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/

# Coherent text; actual-path FP64 variants are documented in docs/testing.md.
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-text.py --socket /tmp/contribution-smoke.ipc --max-tokens 64

```

## What not to do

- Do not add swap-based preemption, multi-GPU, gRPC serving, LoRA, or multimodal inputs — these are explicitly out of scope (REQ-OUT-SCOPE-001). Open a discussion first if scope needs to change. OpenAI-compatible HTTP serving is implemented (REQ-SERVE-001); its real GPU/agentic acceptance is still required.
- Do not commit secrets, tokens, `.env` values, or production credentials.
- Do not force-push to any branch without explicit confirmation from the project owner.
- Do not upgrade the `zeromq` Rust crate without checking that `DealerSocket` and `ipc-transport` still work correctly — see `docs/decisions/ADR-001-zmq-socket-type.md`.
