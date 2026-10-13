# Current state and open work

Updated: 2026-10-14. Refactor work is active. Final delivery is pending.

## Accepted scope

Keep single-B200 Qwen3.8-27B-FP8 inference, Rust service/scheduler/logical cache, Python GPU ownership, and ZMQ DEALER/msgpack.
Keep block size 784, 16 FA layers, 48 GDN layers, `mamba_cache_mode="align"`, and 262144 total tokens.
Keep API behavior, numerical tolerances, ordinary/MTP4/DSpark, and unchanged DSpark weights.

Use native CUDA libraries through repository CUDA. Remove external Python operator providers and project comparison dependencies.
Keep independent numerical/state contracts. Framework phase latency must meet the 2-times lower-bound gate, with the prefix-hit prefill exception.
Keep the theoretical model, two warmups, five repetitions, and 10% spread gates.

## Milestones

1. Document hierarchy, source anchors, word limits, and definition spacing: complete and independently reviewed.
2. Cohesive code modules and critical test layers: complete and independently reviewed.
3. Native CUDA provider migration, independent references, TileFoundry rebase, and harness adaptation: pending.
4. Profiler, request plots, performance tuning, and complete acceptance: pending.

Obtain independent review before each milestone commit. Commit on `main` and continue until all work passes.
Do not push or open pull requests.

## Verification scope

Milestone one passed required Rust/style/document checks, 55 selected Python tests, 14 Node tests, and 9 browser tests.
The tutorial build passed. Independent review findings were corrected.

Milestone two passed required Rust/style/document checks, 14 Node tests, and 9 browser tests.
The full CPU run passed 1572 cases and 156 subtests. One stale environment-path assertion failed.
After correction, all seven environment tests passed. No other CPU failure occurred.

GPU selection kept all 487 nodes, with six isolated boundary cases. This was collection validation, not GPU execution.

GPU acceptance follows the native CUDA migration and source freeze. It did not run for these milestones.
No new-source GPU acceptance is claimed. Historical repair results belong to source `b98a83a` and earlier recorded commits.

Current summaries with obsolete comparison provenance will leave the working tree. Git history keeps their original results.
See the [acceptance index](acceptance.md) for current collection scope.

## Active exception and cleanup

The user-requested CPU tutorial preview stays active. Its host address and process birth identity are in ignored `LOCAL.md`.
Stop task-owned GPU workers, children, and temporary inference listeners after each run.
Agent tests use new external workspaces in disposable sandboxes. Remove copied tools, generated files, containers, and owned images afterward.
Keep raw measurements and temporary tuning in external storage.
