# Contribution procedure

Read [collaboration rules](AGENTS.md) and [current work](docs/handoff.md) before changes.
Develop and commit only on `main`.
Keep existing changes from other work.
Push or open a pull request only after explicit user instruction.

## Implementation

Keep Rust service, scheduler, and logical cache ownership.
Keep GPU computation in Python and third-party libraries.
Keep the block size, numerical tolerances, API features, and active acceptance criteria.
Select stable compatible dependencies and pin the combination with satisfactory compatibility tests.
Install Python dependencies with `uv`. Do not use `pip` directly.

Use `rustfmt.toml` and a hard Rust line width of 100 characters.
This limit includes macros, strings, comments, and expanded tabs.
Split long JSON macros into fields and long literals with `concat!`.
Put a `// SAFETY:` comment at each `unsafe` block.
Use Python 3.12 or later and `ruff.toml`.

Use clang-format 23.1.3 for owned CUDA modules. Keep each backend code file at most 800 lines.

Add a meaningful test for new behavior.
If a test is impractical, record the reason.
Measure performance with the formal protocol. Identify estimates as estimates.
Keep temporary tuning and raw traces in an external directory.

## Review and checks

Start an review by another sub-agent after each milestone.
Correct its correctness, performance, and engineering findings before the commit.
Complete the required [checks](AGENTS.md#required-checks) and applicable [tests](docs/testing.md).
Use `scripts/with-env.sh` for environment selection.
The local hooks in `.pre-commit-config.yaml` use this wrapper.

Update each changed English Markdown document and its full Chinese companion together.
Follow [writing rules](docs/writing.md), [glossary](docs/glossary.md), and the document update table in `AGENTS.md`.
Keep host configuration in ignored `LOCAL.md`.
Keep original evidence in ignored local storage. Commit clearly marked portable summaries.

## Commit

Stage only the paths that belong to the change:

```bash
git add path/to/changed-file path/to/another-file
scripts/with-env.sh pre-commit run --all-files
git commit
```

The commit hooks must have the same configured environment as the manual checks.
Keep hooks active. Do not use `git add .` or `--no-verify`.
Use a Conventional Commits title, such as `docs: make deployment guides portable`.
Give the resulting behavior, validation, skipped checks, and open limitations in the body.

Include this attribution line in each commit:

```text
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

Update `docs/handoff.md` with current results and open work.
Stop owned GPU processes and temporary listeners.
For a user-requested active service, record its purpose in the handoff and its host details in `LOCAL.md`.
