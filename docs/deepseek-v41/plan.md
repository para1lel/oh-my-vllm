# DeepSeek V4.1 phased implementation plan

> Agent workers: use `superpowers:executing-plans` or `superpowers:subagent-driven-development` after maintainer approval. Complete one reviewed phase at a time.

**Goal:** Add four-GPU DeepSeek inference without loss of existing Qwen functions.

**Architecture:** Model adapters supply mathematics, state layouts, and codecs. Rust controls scheduling and state transactions. Python executes native operators.

**Tech stack:** Rust, Python, ZMQ DEALER, msgpack, CUDA C++ or CuTe DSL, and native NCCL.

**Spec:** [Architecture proposal](design.md). This plan is proposed, not executed.

## Global constraints

- Work on `nightly`, from base `b99701a`.
- Use GPT-6.1 Sol with `high` effort for agents started in this session.
- Keep Qwen ordinary decode, MTP4, 262144 tokens, and its single-B200 configuration.
- Add DeepSeek TP4, TEP4, DEP4, DSpark with five drafts, 64 active requests, and 1048576 total tokens per request.
- Keep Engram weights in pinned host memory.
- Keep Rust state ownership, DEALER/msgpack, and the whole-Execute Eager/Graph rule.
- Do not add external operator libraries or vLLM dependencies.
- Announce each new test and its missing coverage before adding it.
- Keep English documents below 2000 words.
- Supply full Chinese translations.
- Use the [measurement policy](validation.md), not new README result tables.

## Review focus

- A rank fails during a collective: fail the group without partial state commit. Phase 2 owns this check.
- A rejected draft crosses a compression boundary: restore all accepted state. Phases 5 and 6 own this check.
- A transfer outlives cancellation: keep its host and device storage alive. Phases 3 and 5 own this check.
- Sixty-four long requests compete for memory: bound allocation without hidden serialization. Phases 5 and 8 own this check.
- A model has different media or tool syntax: use its codec without Qwen regressions. Phases 1 and 7 own this check.

## Working method

Each phase produces working, independently checked software or a frozen prerequisite contract.
Write a small implementation checklist for its interfaces before coding that phase.
Use failing behavior checks before implementation when new behavior has no existing check.

Do not copy this roadmap into one large implementation change.

For each phase, get independent review.
Correct the findings.
Run applicable hooks.
Update the handoff.
Commit selected files.
Push the working branch.

Keep workers, services, GPU locks, and external artifacts with explicit ownership.
Stop all processes started for the phase.

## Phase 0: Freeze model and acceptance contracts

**Files:** Add `docs/deepseek-v41/model.md` and its translation. Add `development/model/deepseek_v41/identity.py`.
**Inputs:** Local checkpoint and fixed upstream sources.
**Output:** A versioned identity manifest and approved mathematical, precision, state, and measurement contracts.

- [ ] Validate all 48 shards, tensor names, offsets, shapes, scales, and file identities without loading all tensors.
- [ ] Compare local weights with the fixed DeepInfer revision and its different default checkpoint.
- [ ] Specify CED, CSA2, FP4, Engram hashing, prompt encoding, media encoding, and DSpark probability rules.
- [ ] Freeze independent reference inputs, full vocabulary outputs, numerical limits, and operator performance limits before kernel tuning.
- [ ] Plan layer-streamed reference computation instead of a full FP64 weight copy.
- [ ] Get approval of the parallel definitions.
- [ ] Resolve the HTTP strategy assumption in the validation plan.
- [ ] Calculate per-rank and host budgets for the combined capacity target.

**Gate:** No unresolved weight mapping or acceptance rule can affect Phase 3 operators.
**Test reason:** Existing identity checks accept one Qwen configuration and cannot detect DeepSeek format mismatches.

## Phase 1: Extract model contracts and keep Qwen behavior

**Files:** Add `crates/model-contract/src/lib.rs`, its Cargo manifest, and `python/oh_my_vllm/models/{api,registry}.py`.
Modify `Cargo.toml`, `crates/zmq-worker/src/{main,protocol}.rs`, and `python/oh_my_vllm/worker/inference.py`.
Move model-specific code into `python/oh_my_vllm/models/qwen38/` in small steps.
**Interfaces:** Define `ModelDescriptor`, `CacheSchema`, `ParallelPlan`, `SpeculationSpec`, and `ModelAdapter`.
The descriptor binds identity, context limit, capabilities, formats, and schema version.
The adapter supplies preparation, execution, result validation, state selection, and resource closure.

- [ ] Add registry rejection checks for unknown architectures, unsupported formats, and capability mismatches.
- [ ] Move fixed Qwen dimensions, tokenizer rules, media positions, and MTP4 selection out of shared control.
- [ ] Keep current API validation and defaults for Qwen.
- [ ] Run existing Rust, service, constraints, execution, and frozen Qwen numerical checks.

**Gate:** Qwen works through the adapter with unchanged accepted outputs and state semantics.
**Test reason:** Existing checks do not exercise selection between model families.

## Phase 2: Add typed cache groups and four-rank control

**Files:** Modify `crates/kv-cache/src/{group,coordinator,snapshot}.rs`, `crates/scheduler/src/{lib,cache,output}.rs`, and the protocol.
Add `crates/zmq-worker/src/group.rs` and `python/oh_my_vllm/operators/runtime/distributed.py`.
Extend `scripts/with-gpu.sh` and `development/serving/device.py`.
**Interfaces:** Add `StateTransaction`, `RankReady`, `RankResult`, and group epoch fields to the Phase 1 contract.

- [ ] Generalize related state reservations without changing Qwen copy-on-write behavior.
- [ ] Add versioned rank readiness, capacity aggregation, completion validation, deadlines, and group shutdown.
- [ ] Acquire all four GPU locks in stable order, or release all acquired locks before waiting.
- [ ] Recheck all selected devices after locking. Do not change other users' clocks or processes.
- [ ] Test empty ranks, uneven schedules, stale results, startup failure, and failure during a collective.
- [ ] Validate fixed collective order and buffer lifetime for decode graph capture and replay.

**Gate:** Four-rank synthetic execution cannot publish partial results or leak processes and locks.
**Test reason:** Single-worker lifecycle checks cannot detect distributed commit or collective failures.

## Phase 3: Load weights and implement native primitives

**Files:** Add `models/deepseek_v41/{config,weights}.py` below `python/oh_my_vllm/`.
Add operator packages `fp4`, `moe`, `engram`, `hyper_connection`, `compression`, and `sparse_attention`.
Extend `development/native_build.py` and `development/operators/`.
**Interfaces:** Weight plans consume the identity manifest and parallel plan. Operators use the existing API/plan/workspace pattern.

- [ ] Write small fixed reference fixtures for quantization, routing, compression, and Engram hash/lookup boundaries.
- [ ] Implement native FP8/FP4 unpacking, scale interpretation, projections, routing, hyper-connections, and sparse-attention primitives.
- [ ] Load rank-owned weight slices without full FP16 or FP32 expansion.
- [ ] Allocate pinned Engram storage with bounded staging and separate host KV accounting.
- [ ] Check shared registration, NUMA placement, memory pressure, cancellation, and transfer completion ownership.
- [ ] Run numerical checks, operator latency checks, and Compute Sanitizer for changed memory paths.

**Gate:** Operators agree with approved references and fit their startup budgets.
**Test reason:** Qwen fixtures do not cover FP4, UE8M0, sparse routing, or host-backed model weights.

## Phase 4: Run ordinary decode with all parallel strategies

**Files:** Add `models/deepseek_v41/{model,parallel,cache}.py` and native communication plans.
Add full-model capture and reference adapters in `development/model/deepseek_v41/`.
**Interfaces:** Consume Phase 1 descriptors, Phase 2 transactions, and Phase 3 operators.
Produce target logits and accepted state through the common adapter.

- [ ] Implement full short-context CED/CSA2 target computation before decoder-work reduction.
- [ ] Implement TP4 first.
- [ ] Implement TEP4 after TP4.
- [ ] Implement DEP4 after TEP4.
- [ ] Specify placement and communication for shared experts, output heads, vision, and future DSpark weights.
- [ ] Test uneven expert loads, zero-token ranks, top-k ties, and request-order changes.
- [ ] Compare all 129280 endpoint logits with frozen references in Eager prefill and graph decode.

**Gate:** All three strategies serve ordinary text through offline and HTTP paths.
**Test reason:** Operator agreement alone cannot prove full distributed-model composition.

## Phase 5: Reach long context and 64-request capacity

**Files:** Extend the DeepSeek cache adapter, typed Rust groups, scheduler, and worker offload paths.
**Interfaces:** Cache schema specifies shared producers, persistent state, partial groups, rollback, and rank placement.

- [ ] Implement chunked prefill, bounded index workspaces, compressed shared KV, and private speculative state.
- [ ] Implement prefix reuse, pressure-driven offload, exact restoration, and safe partial-prefix continuation.
- [ ] Validate decoder replay against full computation before selecting a reduced-work path.
- [ ] Do not substitute approximate bounded replay for exact restoration without separate approval and numerical limits.
- [ ] Exercise 400000, 800000, and 1048576-token boundaries, with output capacity reserved.
- [ ] Test 64 active long requests with ordinary decode in each strategy.
- [ ] Test cancellation during memory pressure and repeated prefix eviction and restoration.

**Gate:** Per-rank memory accounting agrees with allocation and the ordinary-decode capacity case completes.
The final DSpark capacity check belongs to Phase 6.
**Test reason:** Existing 32769-token Qwen checks do not prove compressed-cache boundaries or four-rank long-context capacity.

## Phase 6: Add DSpark with five draft tokens

**Files:** Add `models/deepseek_v41/dspark.py` and `operators/dspark/`.
Extend shared speculation metadata, scheduler budgets, constraints, and traces.
**Interfaces:** Proposal results contain token IDs, proposal probabilities, confidence, positions, and private state references.
Acceptance returns committed lengths separately from new proposals.

- [ ] Implement three prediction layers with a maximum of five proposed tokens, not five recurrent MTP layers.
- [ ] Implement fixed-length verification first, with up to six target rows.
- [ ] Add confidence-scheduled verification after fixed-length acceptance passes its reference checks.
- [ ] Keep exact probability correction unless a different bounded-error rule receives explicit approval.
- [ ] Test rejection at each position, all-accept, EOS, stop strings, output limits, schema masks, and transfer restoration.
- [ ] Test RNG and penalty history with partial verification and reordered batches.
- [ ] Run the final combined capacity check with DSpark in each strategy.

**Gate:** DSpark works in all three strategies without changing accepted sampling or state contracts.
**Test reason:** Qwen MTP4 cannot detect DSpark block, confidence, or six-row verification errors.

## Phase 7: Complete API and media parity

**Files:** Add DeepSeek prompt, parser, media, and vision adapters. Extend `tests/test_service.py` and existing constraint checks.
Modify `crates/zmq-worker/src/serving/` only for model-selected behavior.
**Interfaces:** Keep shared request/response objects. Model adapters supply codecs and capability validation.

- [ ] Complete the function checklist in the validation plan.
- [ ] Check encoding against fixed local reference tokens for tools, reasoning, images, and conversation continuation.
- [ ] Implement validated video-frame ordering and timestamps without Qwen position assumptions.
- [ ] Exercise ordinary and DSpark paths with HTTP streaming, offline requests, cancellation, and worker replacement.

**Gate:** No advertised Qwen-equivalent function is missing from DeepSeek.
**Test reason:** Text inference does not exercise model-specific tool syntax or multimodal history.

## Phase 8: Measure, review, and deliver

**Files:** Generalize `development/serving/{benchmark,agentx,device,trace,accounting,workload}.py`.
Add `development/serving/profiles/deepseek_v41.py`.
Extend existing driver validation tests.
Write approved results in the separate documents named by the validation plan.

- [ ] Validate exact case enumeration, four-rank identities, prompt lengths, output lengths, and request joins.
- [ ] Run the approved HTTP and AgentX matrices only after correctness gates pass.
- [ ] Keep Qwen regression measurements separate from DeepSeek results.
- [ ] Report measured latency, throughput, acceptance, memory, failures, and resource identities without copied Qwen efficiency values.
- [ ] Follow the phase delivery procedure in Working method.

**Gate:** Results bind the delivered source, native modules, weights, tokenizer, topology, and measurement configuration.
**Test reason:** Existing drivers fix Qwen case sizes, one GPU, and four AgentX session trees.
