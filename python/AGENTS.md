# GPU and worker ownership

Read [Python ownership](../docs/architecture.md) before runner changes.
Python owns tensors, model computation, sampling, and CUDA Graphs. It must not invent logical cache blocks.
Keep accepted-state writes, speculative rollback, graph restoration, and caller-stream lifetime contracts.

Keep model prompts, grammar constraints, and numerical tolerances effective.

## Code and checks

Use the project environment wrapper and install Python dependencies with `uv`.
Prefer explicit types, names, and cohesive modules. Remove comments that repeat code.
Keep concise interface, numerical, lifetime, and concurrency contracts.

Use Ruff and syntax-aware definition spacing. Do not separate decorators from their function.
Keep independent mathematical references and fault cases when ordinary integration coverage is not sufficient.
Run affected CPU tests and applicable GPU acceptance. Stop owned GPU workers after each run.
