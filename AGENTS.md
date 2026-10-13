# Collaboration principles

Build single-B200 inference for Qwen3.8-27B-FP8.
Rust owns service, scheduling, and logical KV. Python owns GPU computation.
Use ZMQ DEALER and msgpack between the processes.

## Invariants

Keep features, numerical tolerances, and [active requirements](docs/requirements.md).
Keep block size 784, 16 FA layers, 48 GDN layers, `mamba_cache_mode="align"`, and 262144 total tokens.
Do not change DSpark weights or introduce vLLM dependencies, environments, or caches.

Use repository CUDA for operations that Torch does not supply. Native CUDA libraries are permitted.
Keep development tools separate from inference dependencies.

## Workflow

Read Git status, make sure the branch is `main`, then read [handoff](docs/handoff.md) and applicable directory instructions.
Use the simplest change that meets the contracts. Add tests for critical behavior, not implementation details.
After each milestone, obtain an independent sub-agent review, correct findings, update the handoff, and commit intentional paths.

Follow [contribution rules](CONTRIBUTING.md). Keep hooks enabled.
Do not push, open pull requests, or create branches without explicit user instruction.
The existing TileFoundry adaptation branch is the permitted exception.

## Required checks

Use [development commands](docs/development.md#build-and-checks) and applicable [tests](docs/testing.md).
Measure performance claims. Report skipped checks and verification limits.

## Ownership

Use the environment and GPU wrappers. Stop owned GPU processes and temporary services after each run or handoff.
Only an explicit user request permits a service to stay active. Record it in ignored `LOCAL.md` and the handoff.
Keep host configuration in `LOCAL.md`, raw traces and experiments in external storage, and current documents short.

Keep English documents and their Chinese companions equal in scope.
