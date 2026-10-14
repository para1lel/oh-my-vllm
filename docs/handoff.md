# Current state and open work

Updated: 2026-10-15. Development is active. Final delivery is pending.

## Accepted scope

Keep single-B200 Qwen3.8-27B-FP8 inference, Rust service/scheduler/logical KV, Python GPU computation, and ZMQ DEALER/msgpack.
Keep block size 784, 16 FA layers, 48 GDN layers, `mamba_cache_mode="align"`, and 262144 total tokens.
Keep ordinary/MTP4/DSpark, API features, and DSpark checkpoint bytes. Do not introduce vLLM dependencies, environments, or caches.
Use repository CUDA and native CUDA libraries. Keep development tools separate from inference dependencies.

The user authorizes numerical-threshold review against vLLM, FlashInfer, and FLA operator and model tests.
A small measured numerical error can be accepted for a large measured performance benefit.
Current thresholds stay source-bound until a reviewed change supplies new error, performance, and model-output evidence.
Keep state commit/rollback, cache isolation, memory safety, and workload contracts.

The user requests reproduction of the previous operator implementations as the optimization starting point.
Keep only contracts and open work in guides until the required performance goals pass.

## Verification checkpoint

Clean `96ad82c` passes all 539 formal operator numerical cases, with complete coverage and no interference.
Its [report](evidence/attention-merge/final-numerical.json) binds the unchanged collection and independent review.
The earlier CPU collection passes 1682 tests and 153 subtests without skips.
The latest documentation checkpoint passes all eleven hooks and twenty documentation tests.

Source `3324a7f` has six maximum-context and MTP/DSpark service/recovery records.
These historical records do not supply final-source acceptance. See the [acceptance index](acceptance.md).

The ordinary 32768-input/4096-output batch-one diagnostic has prefill/decode ratios 4.305/2.181.
The two latency gates fail. Spreads pass. Full fifteen-workload and 27-latency-gate acceptance is pending.

## Active work

1. Compare the previous FlashInfer GDN and small TRT-LLM FP8 implementations with the current complete operations.
2. Review matched numerical thresholds and collect error and model-output quality evidence for proposed tradeoffs.
3. Finish current isolated CUDA diagnostics. Keep all outcomes and exact source identities.
4. Complete the final-source numerical, CPU/GPU regression, capacity, service, and performance checks.
5. Run MTP4 and DSpark full-book translation collections, each for at least one hour in four isolated OMP sandboxes.

The book is downloaded. Translation work on OMP Issue 4 stays in its separate checkout and awaits integration.
Its two complete configured CPU attempts each have 1802 passing tests and 153 subtests, with one failed Chat deadline proof.

Issue 19 records the native/SDK completion-clock mismatch above the unchanged 2 ms guard.

Local oh-my-pi repair and isolated verification are authorized. Upstream publication is not authorized.
Keep the installed client and failed evidence while the repaired client is verified.

Use [contribution](../CONTRIBUTING.md) and [collaboration](collaboration.md) for independent reviews, checks, intentional paths, and reviewed publication.
Codex uses `main`. OMP uses its assigned branch. Rebase, repeat checks, squash, and fast-forward before integration.
Project Issues and reviewed milestones are authorized. TileFoundry's existing adaptation branch is the permitted publication exception.

## Ownership and records

The user-authorized CPU tutorial preview stays active. Its host configuration is in ignored `LOCAL.md`.
Stop owned GPU processes and temporary services after each run. Keep raw traces, experiments, and failed records in external storage.
Current guides contain contracts and open work. Historical records stay source-scoped.
