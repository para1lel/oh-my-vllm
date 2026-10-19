# Tile execution implementation plan

> For agentic execution: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` for task-by-task implementation.

**Goal:** Move all GPU tensor computation to tile-dependent MegaKernel execution and pass the final HTTP performance gate.

**Architecture:** One plan interface supplies finite task graphs, runtime metadata, and state permissions.
Specialized persistent kernels can share its dependency contract.
Rust keeps service scheduling and logical cache ownership.

**Tech stack:** Rust, Python, ZMQ DEALER, msgpack, CUDA C++/CUTLASS, and permitted CuTe DSL.

**Spec:** [Tile execution design](tile-execution.md).
This plan supplies implementation steps. The design controls scope and acceptance.

## Global constraints

- Use `nightly` and one NVIDIA B200 with Qwen3.8-27B-FP8.
- Keep 48 GDN layers, 16 FA layers, block size 784, and the 262144-token total limit.
- Keep the model values, numerical tolerances, private-state rules, and HTTP/offline functions.
- The final target covers all GPU tensor computation, also during initialization and preparation.
- Migration has no performance threshold. Final acceptance must include all 24 HTTP configurations at 50% or more.
- Keep build tools separate from runtime imports.
- Use GPT-6.1 Sol with high effort for subagents in this work.
- State why each new test is necessary before adding it. Reuse independent references where possible.
- Get an independent review before each delivery.
- Run applicable hooks before each delivery.
- Update the two README tables before each delivery.
- Commit only selected files.
- Push the working branch after each delivery.
- Keep reports external to the repository. Stop temporary processes after use.

## Review focus

- Reused epochs and full queues must not publish stale data or prevent termination.
- Physical call splitting must not increase reference work or manufacture MTP work.
- Positions 783, 784, and 785 must obey state and page rules across plan reuse.
- MTP rejection and short tails must keep accepted history, constraints, state slots, and RNG consistent.
- Vision preprocessing and startup tensor preparation must not escape the final task coverage audit.

## Task 1: Separate HTTP accounting from old operator tools

**Files:** Create `development/serving/accounting.py` and `development/native_build.py`.
Modify `development/serving/roofline.py`, `trace.py`, `benchmark.py`, and `test_roofline.py` in that directory.
Move necessary references to `development/reference/`.
Modify `.pre-commit-config.yaml`, setup commands, and affected bilingual guides.

**Interfaces:** `canonical_passes(events: list[dict], manifest: dict) -> list[dict]` supplies logical passes.
`reference_work(passes: list[dict]) -> dict[str, float]` supplies bytes and work by precision.
`reference_seconds(work: dict[str, float]) -> float` uses the design's nominal constants.
The build entry point is `python -m development.native_build` with the existing build options.

- [ ] Extend `test_roofline.py` with one parameterized accounting regression.
  It must detect inflated work from extra physical splits, changed task counts, padding, or copied state.
  It must also reject fabricated proposals and inconsistent accepted positions.
  Check duplicate/missing request IDs and arrival-order permutations against the benchmark manifest.
  Existing tests bind the old calibration and cannot prove the new invariant.

- [ ] Run `scripts/with-env.sh python -m unittest development.serving.test_roofline`. Make sure that the new check fails.

- [ ] Implement deterministic reference scheduling with the 32768-token budget, decode priority, and at most 4 running requests.
  Use request ordinals from the benchmark, ordered remainder allocation, and cache-aligned prefill chunks.
  Replay recorded MTP dependencies without creating new proposal sequences.

- [ ] Bind the benchmark manifest to the artifact identity.
  Map client request ordinals to response/trace IDs, mode, and configured input/output lengths.
  Reject duplicate IDs, incomplete membership, and inconsistent lengths before reference grouping.

- [ ] Separate reference mathematical work from implementation-selected algorithms.
  Specify scalar work and omissions. Exclude removable intermediate traffic and redundant work.

- [ ] Replace calibration inputs with fixed nominal rates and an accounting version.
  Keep hardware, model, executable, source, and timing identities.

- [ ] Move the native builder, independent numerical references, and state fixtures before deletion.

- [ ] Remove old operator timing/audit commands, public comparison tools, peak calibration, and unused helpers.
  Update imports, hooks, and commands in the same change.

- [ ] Run the accounting and moved reference tests. Update all 24 README rows without mixing accounting versions.

- [ ] Get an independent review. Run the applicable hooks. Commit the selected files. Push the working branch.

## Task 2: Build the task engine and its resource contract

**Files:** Create `python/oh_my_vllm/kernels/tile_plan.py`, `mega.py`, `cuda_backend/mega.cu`, and `cuda_backend/tile_runtime.cuh`.
Create `development/tile/test_engine.py` and task fixtures in `development/tile/`.
Update the native builder and loader.

**Interfaces:** `prepare(kind, capacity, bindings) -> Plan`, `execute(plan, metadata) -> Completion`, and `close(plan) -> None` live in `mega.py`.
`Plan` owns graph descriptors, task resources, stable buffers, and native leases.
`StepMetadata` carries epoch, query lengths, page tables, read/write slots, and valid rows.
`Completion` supplies epoch, error status, and completed outputs.

- [ ] Add one engine protocol test with a small independent producer/consumer graph.
  Exercise fan-in, repeated epochs, buffer reuse, queue capacity failure, and invalid metadata.
  Numerical references do not test queue publication or progress.

- [ ] Run `scripts/with-env.sh python -m unittest development.tile.test_engine`. Make sure that the check fails before implementation.

- [ ] Implement immutable topology, runtime task ranges, bounded queues, and data/state/lifetime edges.

- [ ] Implement publication ordering, asynchronous completion, error status, and finite termination.
  Check occupancy and progress for the full launched grid, not just one task type.

- [ ] Connect failure to worker removal and epoch cleanup. Do not reuse partially modified state.

- [ ] Run CPU protocol checks and GPU fixtures through `scripts/with-env.sh scripts/with-gpu.sh`.
  Use the existing step deadline and Compute Sanitizer for the GPU protocol fixture.

- [ ] Get an independent review. Run the applicable hooks. Commit the selected files. Push the working branch.

## Task 3: Complete ordinary decode

**Files:** Add language and tensor task bodies in `cuda_backend/tile_tasks/`.
Modify `worker/model.py`, `worker/inference.py`, `worker/sampling.py`, and the FFN/FA/GDN wrappers.
Split vocabulary projection from sampling in the native interface.

**Interfaces:** A decode `Plan` consumes the current model bindings and step metadata.
It supplies hidden results, logits, and candidate cache/state writes.
Independent sampling consumes those logits without a second vocabulary projection.

- [ ] Adapt existing numerical and state checks to compare the new full decode path with independent references.
  Cover supported batch capacities, inactive rows, changed page tables, and positions 783/784/785.

- [ ] Implement embedding, all 64 layers, final normalization, and vocabulary projection as task execution.
  Share finite task bodies across layers. Keep precision and state permissions explicit.

- [ ] Submit decode plans through CUDA Graph with stable addresses and updated metadata.

- [ ] Run the adapted checks and ordinary HTTP configurations.
  Remove replaced decode implementations after correctness passes.

- [ ] Get an independent review. Run the applicable hooks. Commit the selected files. Push the working branch.

## Task 4: Complete prefill

**Files:** Add prefill task bodies in `cuda_backend/tile_tasks/`.
Modify `tile_plan.py`, `worker/model.py`, and the FA/GDN/FFN wrappers.

**Interfaces:** A prefill `Plan` supplies prompt hidden rows, required checkpoints, and endpoint results.
It uses the same completion and state-permission contract as decode.

- [ ] Adapt existing FA and GDN references for variable lengths, chunk boundaries, and noncontiguous page tables.

- [ ] Implement large projection, attention, GDN chunk, metadata, and reduction task families.
  Keep the prescribed FP32 live carry and BF16 checkpoint behavior.

- [ ] Check chunked execution, prefix reuse, and final-row state against the model contract.

- [ ] Run the adapted checks and prefill-sensitive HTTP cases with 64 output tokens.
  Remove replaced prefill computation paths.

- [ ] Get an independent review. Run the applicable hooks. Commit the selected files. Push the working branch.

## Task 5: Complete MTP and sampling

**Files:** Add sampling and MTP task bodies in `cuda_backend/tile_tasks/`.
Modify `worker/speculation.py`, `worker/draft_chain.py`, `worker/sampling.py`, `kernels/sample.py`, and `kernels/mtp.py`.

**Interfaces:** Sampling plans consume logits and sampling metadata.
MTP plans supply proposals, accepted counts, selected state slots, and RNG results.
CPU constraint state changes only after accepted-token results.

- [ ] Adapt seeded sampling and MTP references for acceptance counts 0 through 4 and short tails.
  Keep penalty, grammar, random, greedy, and mixed-batch coverage.

- [ ] Implement context, proposal, verification, acceptance, gather, sorting, and random-selection tasks.

- [ ] Keep CPU decision boundaries where the service contract uses them.
  A finite submission can return for a CPU decision without leaving GPU tensor work on the old path.

- [ ] Run the adapted references, constraint tests, and all 18 MTP4 HTTP configurations.
  Remove replaced sampling and MTP implementations.

- [ ] Get an independent review. Run the applicable hooks. Commit the selected files. Push the working branch.

## Task 6: Complete vision

**Files:** Add vision task bodies in `cuda_backend/tile_tasks/`.
Modify `worker/media.py` and `kernels/vision.py`.

**Interfaces:** Vision plans consume decoded media tensors and shape metadata.
They supply ordered language embeddings with the existing image/video position semantics.

- [ ] Adapt vision references for pixel resizing, frame isolation, patch order, partial frame groups, and spatial merge.

- [ ] Implement GPU preprocessing, positions, all 27 layers, merge, and embedding gather through task dependencies.

- [ ] Check image and video service paths with the inference worker, not the placeholder.
  The text-only HTTP performance table does not prove vision correctness.

- [ ] Remove replaced vision implementations after the adapted checks pass.

- [ ] Get an independent review. Run the applicable hooks. Commit the selected files. Push the working branch.

## Task 7: Close preparation coverage and final acceptance

**Files:** Add preparation and auxiliary task bodies in `cuda_backend/tile_tasks/`.
Modify `worker/weights.py`, `worker/device.py`, `worker/cache.py`, and native registration/build lists.
Add coverage checking to `development/tile/test_engine.py`.

**Interfaces:** Preparation plans use the same engine for weight conversion, scales, tables, and tensor initialization.
The coverage record maps each runtime GPU computation entry to a task family and correctness check.

- [ ] Complete weight preparation, activation tables, indexed transfers, initialization, and auxiliary tensor tasks.

- [ ] Audit startup, warmup, capture, replay, and error paths against native entry points and execution traces.
  Standalone cuBLAS/CUB/CUDA tensor calls cannot bypass task execution at final acceptance.
  CUDA transfer and resource-management operations keep their infrastructure role.

- [ ] Add a coverage regression that rejects an unclassified runtime tensor-compute entry.
  Existing output comparisons cannot detect an old computation path that still produces correct output.

- [ ] Remove unused operator code, registrations, build inputs, and stale guides.

- [ ] Run all applicable correctness checks, hooks, and the full 24-case HTTP comparison.
  Tune task algorithms and resource layouts until each row meets 50% at the specified precision.

- [ ] Bind final artifacts to one source/accounting identity. Update the two README tables.

- [ ] Get a whole-change review. Correct its findings. Commit the selected files. Push the working branch. Stop temporary processes.

## Plan updates

Research or measurements can change task implementations, kernel families, and migration order after the first decode milestone.
Update this plan and its Chinese translation when those choices change.
Keep final coverage, numerical contracts, and the HTTP gate fixed.
