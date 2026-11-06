# Work state

Use `nightly` for development.
Read the continuation index in ignored `LOCAL.md` before work.
The index gives external contracts, coordinator state, resource ownership, and Git delivery records.
The current deadline is 2026-11-07 at 09:00 CST.
The maintainer authorized the current cache and service changes.

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

Model acceptance with production checkpoint weights or GPU execution stays closed until public comparisons give satisfactory results for all shapes that the operators use. CPU unit tests are permitted, also for small models with generated or synthetic weights.
The new Qwen format, physical capacity, GPU behavior, full HTTP matrix, and AgentX acceptance are not verified.
The complete DeepSeek HTTP, AgentX, and efficiency targets are not complete.
Old measurements apply only to their recorded source, binary, and accounting identities.

## Next work

1. Read the current external implementation contracts and coordinator state from the `LOCAL.md` index.
   Check the assigned files and current source identities.
   Get reviews from other agents. Correct their findings.

2. Complete the public comparison gate for all shapes that the operators use before checkpoint or GPU model acceptance.
   CPU unit tests are permitted, also for small models with generated or synthetic weights.
   Use the [execution rules](execution.md) for GPU ownership.
   An authorization record cannot replace a live resource lease.

3. Check the complete Qwen budget before capacity acceptance.
   Count cache slabs, endpoint bundles, hidden values, scales, workspaces, metadata, MTP backups, Graphs, and allocator storage.
   Measure physical peaks when execution is authorized.
   Static capacity examples do not prove usable GPU capacity.

4. Complete applicable numerical, state, cancellation, copy-on-write, Graph, and full-model checks for the frozen implementation identity.
   Keep checkpoint weights, scales, frozen fixtures, numerical order, error status, and acceptance limits unchanged.
   Use the [test scope](testing.md) and model guides.
   Recheck changed identities after further source changes.

5. Complete the approved [Qwen HTTP](models/qwen38/performance.md) and [AgentX](models/qwen38/agentx.md) measurements.
   Complete the approved [DeepSeek HTTP](models/deepseek_v41/performance.md) and [AgentX](models/deepseek_v41/agentx.md) measurements.
   Use the DeepSeek performance guide's explicit v6 selection and acceptance command.
   Report request completion and session-tree completion separately.

## Evidence and delivery

The partial `qwen-fp8-components-v1` resource calculation is not a complete HTTP acceptance calculator.
The [recorded Qwen results](models/qwen38/results.md) keep the approved v5 tables and their recorded scope.
Do not replace result tables before a new full matrix completes.
Keep raw records and checks external to the repository.

Use the [development loop](development.md#development-loop), [reporting policy](testing.md#reporting), and [contribution procedure](../CONTRIBUTING.md#delivery).
The [verification skill](../.claude/skills/verify-oh-my-vllm/SKILL.md) records Rust app behavior.
Its CPU fixture does not prove GPU inference, numerical correctness, or capacity.
Get implementation and document reviews from other agents before delivery.

The coordinator commits and pushes `nightly`, then completes the squash and push to `main`.
The coordinator then sets `nightly` to the same revision as `main` and pushes `nightly`.
Push only `main` and `nightly`. Keep hooks active.
Stop owned workers and temporary services.
Check process identities, IPC cleanup, and resource release against the external coordinator record.
