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

Numerical comparisons always use an independent FP64 reference.
Full-model comparisons use small experiments. Use individual operators for large numerical comparisons.
Previous implementations supply timing baselines only.

The user requests reproduction of the previous operator implementations as the optimization starting point.
Keep only contracts and open work in guides until the required performance goals pass.

## Verification checkpoint

The historical `96ad82c` collection passes 539 cases with its original mixed-precision references.
Its [report](evidence/attention-merge/final-numerical.json) stays unchanged and does not supply independent FP64 acceptance.

The new [FP64 reference record](evidence/fp64-reference/verification.json) covers independent equations and 22 small CPU tests.
A CPU experiment reads the checkpoint and runs 64 prefix tokens and 8 continuation tokens through all 64 layers.
Its 260 floating-point artifact tensors use FP64. Independent review accepts this reference only for its exact checkpoint and token sequence.
This record does not supply candidate numerical or performance acceptance.

The complete CPU collection passes 1828 tests and 153 subtests without skips, with 637 GPU tests deselected.
The earlier attempt with missing corpus configuration keeps its one skip.
The scoped collection passes 75 tests, with 13 GPU tests deselected. All eleven hooks pass.

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

The book is downloaded. Reviewed PR 21 is integrated at `004765b` by fast-forward from `828d5e9`.
Issue 19 is closed. The [clock report](evidence/omp-clock/verification.json) keeps the older source-scoped collection and failed-attempt identities.
The 2 ms native/SDK completion-difference guard stays unchanged.

Local oh-my-pi repair and isolated verification are authorized. Upstream publication is not authorized.
Use the verified isolated client for Issue 4. Keep the installed client and failed evidence.
Issue 4's full-book collections and final-source acceptance stay open.

Use [contribution](../CONTRIBUTING.md) and [collaboration](collaboration.md) for independent reviews, checks, intentional paths, and reviewed publication.
Codex uses `main`. OMP uses its assigned branch. Rebase, repeat checks, squash, and fast-forward before integration.
Project Issues and reviewed milestones are authorized. TileFoundry's existing adaptation branch is the permitted publication exception.

## Ownership and records

The user-authorized CPU tutorial preview stays active. Its host configuration is in ignored `LOCAL.md`.
Stop owned GPU processes and temporary services after each run. Keep raw traces, experiments, and failed records in external storage.
Current guides contain contracts and open work. Historical records stay source-scoped.
