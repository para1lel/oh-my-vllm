# DEP4 Expert-Parallel Load Balancing

Audit date: 2026-10-26. This is a read-only design study. It does not add code, start a GPU, or start a worker.

## Conclusion

Adopt two separate controls in the new DEP4 policy. Keep each request owner on one DEP4 rank for the complete step. Balance expert work with replicated hot experts and a slow control loop. Do not migrate an in-flight request or expert map.

The current seam can carry this rule, but does not enforce owner stability across steps. `ParallelPlan::validate` checks four logical ranks and checks that `Dep4` uses `rank_partitioned` placement ([model-contract](../../crates/model-contract/src/validation.rs)). `GroupGate::dispatch` filters requests by `owner_rank`, sends all ranks a schedule, and keeps zero-row ranks in the group ([group.rs](../../crates/zmq-worker/src/group.rs)). `GroupClient` has a transactional path that detaches state before an asynchronous exchange and applies state only after validated completion. Its borrowed path keeps owners live until close succeeds ([rank_client/model.rs](../../crates/zmq-worker/src/rank_client/model.rs), [transaction.rs](../../crates/scheduler/src/transaction.rs)).

This is a contract seam, not DEP4 support. It has no expert route fields, expert-map version, or load telemetry. Add those fields in a DeepSeek adapter and policy. Keep Qwen on its current TP1 path and do not add DeepSeek defaults to shared Qwen code ([adapters](../adapters.md#ownership), [Qwen adapter](../models/qwen38/README.md)).

## 1. Signals and their time

| Signal | First useful time | Use and limit |
|---|---|---|
| Prefill rows and tokens | Before packing | Exact Rust inputs. Use rows for kernel work and tokens for admission. Rows do not show expert skew. |
| Decode rows | Before packing | Usually one row per request, or more for verification. It does not show context length or expert skew. |
| Context, KV, attention, and cache bytes | Before packing | Derive from request state and the adapter cache contract. Attention bytes depend on cached length and sparse selection. These are estimates until the operator plan runs. |
| Engram lookup count and bytes | Before packing if token IDs are known | Count lookup entries and shard traffic. Exact transfer bytes are known after the lookup plan, not from request count. Keep asynchronous fetch ownership in Python. |
| Predicted active experts and histogram | Before packing, from a prior step or a route probe | A previous-step histogram is cheap but stale. A route probe is more accurate but adds work. For a new prompt, exact top-k routing is normally known only after the gate. |
| Observed expert histogram and active experts | After the gate and dispatch | DeepEP `EPHandle` carries per-expert received counts, prefix sums, source metadata, and the handle needed by combine ([`EPHandle`](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py#L27-L99)). |
| Route and communication bytes | With the dispatch plan | The route matrix gives source and destination rows. The adapter adds alignment padding, valid-row metadata, and send/receive byte estimates. DeepEP can report exact counts with CPU synchronization ([`dispatch`](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py#L562-L737)). |
| Device time and completion | After each rank finishes | Use per-rank forward, dispatch, expert GEMM, combine, attention, Engram, and transfer times. Rust must use only the completion contract for state commit. |

Use a weighted estimate before dispatch:

`cost(rank) = a * prefill_rows + b * decode_rows * (1 + context_tokens) + c * expert_rows + d * route_bytes + e * cache_bytes + f * Engram_bytes`.

The adapter owns coefficients and phase-specific terms. Rust owns the decision and records the estimate. After dispatch, compare the estimate with observed per-rank telemetry. DeepSeek reports the same principle at service level: its decode balancer considers request count and total KV length together ([DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L117-L119)).

DeepEP's exact CPU count is useful for a calibration or slow control path, not each latency-sensitive step. A cached handle can skip layout work, but its route metadata must stay unchanged until combine completes. Treat the handle as Python-owned execution state, not Rust ownership.

## 2. Safe scheduling rules

### Request placement and packing

1. Assign `owner_rank` at admission or at a completed-step boundary. Keep it for the full step.
2. Pack requests by phase: decode first, then complete prefill chunks, then residual prefill. Keep the existing Rust priority ([runtime contracts](../requirements.md)).
3. Apply three limits: global rows, predicted per-rank weighted cost, and route-buffer bytes. A request may enter only when all three limits hold.
4. Keep stable request order and stable row offsets. The adapter must return source-row IDs so combine restores the owner order.
5. Let expert dispatch split token rows across expert ranks. Do not split request ownership. The owner rank keeps model-local route state and output ownership. Rust keeps accepted-token state and state commit.

The current scheduler supplies computed tokens, draft tokens, row totals, and cache assignments, but no expert estimate ([`Request`](../../crates/scheduler/src/request.rs), [`planning.rs`](../../crates/scheduler/src/planning.rs)). A DeepSeek policy should add model-specific estimates rather than change Qwen quota rules.

### Rebalance and migration

Use expert replication for the main balance action. The first-party EPLB source replicates hot logical experts and returns physical-to-logical and logical-to-physical maps ([`rebalance_experts`](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913/eplb.py#L131-L166)). DeepSeek states a 100-step statistics window and 1000-step reassignment for prefill, and a 1000-step window and 10000-step reassignment for decode ([Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L93-L103)). Use these as starting values, not fixed project limits.

Switch an expert map only at a quiescent step boundary. All ranks must agree on `map_version`, a map-content digest, layer scope, topology, checkpoint identity, and weight-residency and capacity checks. Load the needed replica weights and use the new map in the next collective plan. No live dispatch handle may cross the switch.

Avoid request migration in the minimum contract. If migration is later needed, wait for all-rank completion. Snapshot accepted state, transfer or make the cache available to a new owner, allocate a new transaction, and dispatch the request in a later step. Never move detached state, a live route handle, or speculative state. A failed transfer keeps the old owner and aborts the new transaction.

### Overload, zero rows, and collectives

- If predicted load is too high, reduce the prefill chunk, defer the request, or use a less full batch. Do not silently drop rows or change the route after dispatch.
- If observed load exceeds the route-buffer limit, fail the step before model execution when possible. A typed collective or rank error follows the current group failure policy. It must not produce a partial state commit.
- A rank with zero request rows still receives the same ordered dispatch and collective plan, sends zero route counts where valid, and reports completion. Do not skip a collective on that rank. The current group contract explicitly calls for this ([`distributed.md`](../distributed.md#transactions-and-publication)).
- Use the same plan digest on all ranks. Add request order, owner ranks, row slices, `map_version`, map digest, route matrix, alignment, valid-row metadata, and operation order. The existing collective digest and all-rank completion checks provide a pattern, but do not contain these model fields ([`collectives.rs`](../../crates/zmq-worker/src/group/collectives.rs), [`GroupGate::insert_result`](../../crates/zmq-worker/src/group.rs)).
- Keep prefill Eager for the whole group. Add a Graph eligibility gate for rank shapes, route counts, addresses, and operation order. The current group selects Graph after prefill, but does not yet check these model-specific conditions ([`GroupGate::dispatch`](../../crates/zmq-worker/src/group.rs)).

Normal request cancellation stops future admission for that request. It does not release transactions, cache blocks, Engram buffers, or route handles while device work can still read them. Cancellation during an exchange leaves the scheduler disabled and transactions owned. Close or reap the group before ownership release or replacement with a new scheduler and epoch ([`distributed.md`](../distributed.md#native-completion-and-closure), [`state-arena.md`](../state-arena.md#lease-lifetime)).

## 3. Why request count or token count is not enough

Request count measures batch width, but not prompt length, verification rows, context length, cache bytes, or hot experts. One long decode request can cost more attention work than many short requests.

Token count measures input rows, but not where top-k routes go. Two equal-token batches can place most tokens on one expert, incur different alignment padding, and create different send and receive traffic. A hot expert can serialize its rank while other ranks are idle.

Neither count includes Engram lookup traffic, sparse attention candidates, cache restoration, route metadata, expert weight residency, or the difference between prefill and recurrent decode kernels. DeepSeek's public Q&A therefore balances decode by request count and total KV length, and balances experts with replicated hot experts. The repository must use the same multi-signal principle, while keeping model rules inside the adapter.

## 4. Minimum verifiable contract and acceptance

The smallest DEP4 contract should validate these fields before dispatch:

- `epoch`, `step_id`, unique request and transaction IDs, four ordered ranks, `owner_rank`, phase, row slice, and cache binding.
- A persistent DEP4 owner invariant that binds `owner_rank` to request ID, transaction identity, state generation, and cache ownership across steps. An owner change follows an explicit completed state-transfer protocol.
- model-owned `estimate_version` with prefill/decode rows, context/cache bytes, predicted expert histogram, active-expert count, route bytes, and Engram bytes.
- Python-owned observed telemetry with per-layer expert histogram, active experts, aligned and unaligned rows, rank send/receive bytes, attention/cache/Engram bytes, route-handle identity, and completion times.
- one deterministic plan digest. Include request order, owner and row metadata, expert `map_version`, and map-content digest. Include layer scope, topology, checkpoint identity, route counts, padding, valid-row metadata, collective order, and execution mode.
- all four rank completions, the exact transaction set, one output for each owner request, and no accepted state until Rust validates the complete set.

Test failure paths on CPU with synthetic route matrices. Cover stale maps, route overflow, duplicate requests, and missing zero-row completion.

Cover mismatched digests, rank timeout, worker error, and cancellation during exchange. Cover failed state transfer and stale transaction generation.

Use descriptor checks for Qwen isolation. Use synthetic schedule checks for Qwen isolation. Run complete Qwen HTTP and AgentX regression when shared runtime code changes.

Proposed performance gates are listed below.

- Keep steady-state max-rank/mean expert time at or below `1.10`.
- After warmup, keep p95 of the per-step max-rank/mean-rank wall-time ratio at or below `1.20`. Include dispatch, combine, attention, Engram, transfer, and expert time.
- Complete zero-row steps without missed collectives.
- Report observed-to-predicted cost error by phase.
- Produce no post-cancel commit or use-after-free.

Also report request latency, prefill and decode throughput, route bytes, cache and Engram bytes, and per-rank memory. These are acceptance targets, not measured results.

## Sources

- [DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md)
- [DeepEP `EPHandle` and `dispatch`](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py)
- [EPLB `rebalance_experts`](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913/eplb.py)
- [`oh-my-vllm` distributed contract](../distributed.md)
- [`oh-my-vllm` runtime contracts](../requirements.md)
- [`oh-my-vllm` state ownership](../state-arena.md)
- [`oh-my-vllm` DeepSeek seams](repository-deepseek-seams-and-gates.md)
