# Work state

Use `nightly` for development.
Read the continuation index in ignored `LOCAL.md` before work.
The index gives external contracts, coordinator state, resource ownership, and Git delivery records.
The maintainer authorized the current cache and service changes.
This delivery keeps the current implementation progress for continuation in a new session.

## Delivery targets

| Model | HTTP and AgentX scope | Long-output requirement |
|---|---|---|
| Qwen | Full 24-case HTTP matrix and approved AgentX workload | Each of 12 long-output cases has efficiency at least 40% |
| DeepSeek-V4.1 | Full 38-case HTTP matrix and three approved AgentX workloads | Each of 19 long-output cases has efficiency at least 33% with `deepseek-http-roofline-v6` |

The coordinator owns planning, resource assignment, acceptance, and Git integration.
Use `opus` at `max` for implementation.
Use the current coordinator record for file ownership and GPU authorization.

## Current implementation

The project removes KV host storage, its protocol, and its configuration.
Engram CPU and file-backed weights stay available.
Necessary token, metadata, and draft transfers stay available.

The [Qwen cache](models/qwen38/cache.md) separates 8192-token GDN checkpoints from 128-token FP8 KV pages.
Usable persistent capacity follows `P = 64 * C`, with separate null, private, and scratch storage in the byte budget.
The radix policy keeps independent prefix evidence and orders successful session-tail replacement by admission generation.
The service uses the [version 2 execution protocol](adapters.md#execution-protocol) and model-specific completion rules.
DeepSeek keeps its resident cache, accepted-state restoration, and four-rank owner rules.

Current [HTTP response changes](../crates/zmq-worker/src/serving/engine/output.rs) connect responses to recorded Rust completions.
The [Qwen acceptance tool](../development/serving/qwen_fp8_acceptance.py) checks the 24-case HTTP matrix.
The [AgentX runner](../development/serving/agentx.py) writes reports after failures.
The [MTP adapter](../development/operators/adapters/mtp.py) adds cache evidence through its [cache helper](../development/operators/adapters/mtp_cache.py).

Current operator changes add [FA vector loads](../python/oh_my_vllm/operators/fa_fp8/cuda/attention.cuh).
The [production projection dispatch](../python/oh_my_vllm/operators/projection/cuda/projection.cu) selects grouped four-row computation for `G384/M24/N576/K5120`.
GPU numerical, state, and Graph checks have not run for this production projection binary.
Full FFN derivative checks in Eager and Graph modes have not run for the value and residual outputs.

The timing entry has a limited static review of its connections.
The latest timing-entry review did not recompute dependency hashes.
This review does not approve timing.

Public comparisons must give satisfactory results for all shapes that the operators use before model acceptance.
Until then, model acceptance with production checkpoint weights or GPU execution stays closed.
CPU unit tests are permitted.
Small CPU models can use generated or synthetic weights.

The new Qwen format, physical capacity, GPU behavior, full HTTP matrix, and AgentX acceptance are not verified.
Neither model has completed its full HTTP, AgentX, and per-case efficiency targets.
Existing evidence applies only to its recorded source, binary, and accounting identities.

## Next work

1. Read the current external implementation contracts and coordinator state from the `LOCAL.md` index.
   Check the assigned files and current source identities.
   Get reviews from other agents.
   Correct their findings.

2. Do GPU numerical, state, and Graph checks on the production projection module before fixed timing.
   Complete the full FFN derivative checks in Eager and Graph modes for the value and residual outputs.
   Make sure that the latest timing entry dependency hashes agree with the external evidence.
   Use the [execution rules](execution.md) for GPU ownership.
   An authorization record cannot replace a live resource lease.

3. Complete each operator's fixed timing only after its own numerical checks pass.
   The grouped operator timing does not depend on full FFN checks.
   Complete the public comparison gate for all shapes that the operators use before checkpoint or GPU model acceptance.

4. Examine the full Qwen budget before capacity acceptance.
   Count cache slabs, endpoint bundles, hidden values, scales, workspaces, metadata, MTP backups, Graphs, and allocator storage.
   Measure physical peaks when execution is authorized.
   Static capacity examples do not prove usable GPU capacity.

5. Complete applicable numerical, state, cancellation, copy-on-write, Graph, and full-model checks for the frozen implementation identity.
   Keep checkpoint weights, scales, frozen fixtures, numerical order, error status, and acceptance limits unchanged.
   Use the [test scope](testing.md) and model guides.
   Recheck changed identities after further source changes.

6. Complete the approved [Qwen HTTP](models/qwen38/performance.md) and [AgentX](models/qwen38/agentx.md) measurements.
   Complete the approved [DeepSeek HTTP](models/deepseek_v41/performance.md) and [AgentX](models/deepseek_v41/agentx.md) measurements.
   Use the DeepSeek performance guide's explicit v6 selection and acceptance command.
   Report request completion and session-tree completion separately.

## Evidence and delivery

The [Qwen HTTP guide](models/qwen38/performance.md#new-layout-http-accounting) uses `qwen-fp8-http-reference-v1` for the full cold-HTTP calculator.
The partial `resource_floor` calculation keeps `qwen-fp8-components-v1`.
The CLI defaults to `http-reference-v5` and returns `blocked` for the new layout.
Use the guide's explicit version selection for the 24 HTTP configurations.
This calculator does not cover AgentX prefix or tree accounting.

The [recorded Qwen results](models/qwen38/results.md) keep the approved v5 tables and their recorded scope.
Do not replace result tables before a new full matrix completes.
Keep raw records and checks external to the repository.
The ignored `LOCAL.md` gives the latest evidence index.

Use the [development loop](development.md#development-loop), [reporting policy](testing.md#reporting), and [contribution procedure](../CONTRIBUTING.md#delivery).
The [verification skill](../.claude/skills/verify-oh-my-vllm/SKILL.md) records Rust app behavior.
Its CPU fixture does not prove GPU inference, numerical correctness, or capacity.
Get implementation and document reviews from other agents before delivery.

The coordinator commits and pushes `nightly`, then completes the squash and push to `main`.
The coordinator then sets `nightly` to the same revision as `main` and pushes `nightly`.
Push only `main` and `nightly`.
Keep hooks active.
Stop owned workers and temporary services.
Check process identities, IPC cleanup, and resource release against the external coordinator record.
