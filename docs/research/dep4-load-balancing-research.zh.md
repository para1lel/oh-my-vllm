# DEP4 Expert-Parallel 负载均衡

审计日期: 2026-10-26. 这是只读设计研究. 不添加代码, 不启动 GPU, 也不启动 worker.

## 结论

在新的 DEP4 policy 中使用两个独立控制. 在完整 step 中, 每个 request 保持在一个 DEP4 rank 上. 使用 hot expert 副本和较慢的 control loop 平衡 expert 工作. 不迁移 in-flight request 或 expert map.

当前 seam 可以承载此规则, 但不会跨 step 强制 owner 稳定. `ParallelPlan::validate` 检查四个 logical rank, 并要求 `Dep4` 使用 `rank_partitioned` placement ([model-contract](../../crates/model-contract/src/validation.rs)). `GroupGate::dispatch` 按 `owner_rank` 过滤 request, 向所有 rank 发送 schedule, 并保留 zero-row rank 在 group 中 ([group.rs](../../crates/zmq-worker/src/group.rs)). `GroupClient` 有一个 transactional path, 会在 asynchronous exchange 前 detach state, 并只在完成验证后应用 state. 它的 borrowed path 会保持 owner 存活到 close 成功 ([rank_client/model.rs](../../crates/zmq-worker/src/rank_client/model.rs), [transaction.rs](../../crates/scheduler/src/transaction.rs)).

这只是 contract seam, 不是 DEP4 support. 它没有 expert route 字段, expert-map version 或 load telemetry. 在 DeepSeek adapter 和 policy 中添加这些字段. 保持 Qwen 在当前 TP1 path, 不要把 DeepSeek default 加入共享 Qwen code ([adapters](../adapters.zh.md#职责), [Qwen adapter](../models/qwen38/README.zh.md)).

## 1. 信号及其可用时间

| 信号 | 首次有用时间 | 用途和限制 |
|---|---|---|
| Prefill rows 和 tokens | Packing 前 | Rust input 中精确可知. 使用 rows 表示 kernel work, 使用 tokens 做 admission. Rows 不显示 expert skew. |
| Decode rows | Packing 前 | 通常每个 request 一行, verification 时可以更多. 它不显示 context length 或 expert skew. |
| Context, KV, attention 和 cache bytes | Packing 前 | 从 request state 和 adapter cache contract 推导. Attention bytes 依赖 cached length 和 sparse selection. 在 operator plan 运行前这些是 estimate. |
| Engram lookup count 和 bytes | 已知 token IDs 后的 packing 前 | 统计 lookup entry 和 shard traffic. 精确 transfer bytes 在 lookup plan 后才知道, 不能从 request count 得到. 在 Python 中保持 asynchronous fetch ownership. |
| Predicted active experts 和 histogram | Packing 前, 使用前一步或 route probe | 前一步 histogram 成本低但可能过时. Route probe 更准确但增加 work. 对新 prompt, 精确 top-k routing 通常只能在 gate 后知道. |
| Observed expert histogram 和 active experts | Gate 和 dispatch 后 | DeepEP `EPHandle` 携带每个 expert 的 received count, prefix sum, source metadata 和 combine 所需 handle ([`EPHandle`](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py#L27-L99)). |
| Route 和 communication bytes | Dispatch plan 生成时 | Route matrix 给出 source 和 destination rows. Adapter 另外提供 alignment padding, valid-row metadata 和 send/receive byte estimate. DeepEP 可以通过 CPU synchronization 报告精确 count ([`dispatch`](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py#L562-L737)). |
| Device time 和 completion | 每个 rank 完成后 | 使用每个 rank 的 forward, dispatch, expert GEMM, combine, attention, Engram 和 transfer time. Rust 必须只使用 completion contract 做 state commit. |

在 dispatch 前使用 weighted estimate:

`cost(rank) = a * prefill_rows + b * decode_rows * (1 + context_tokens) + c * expert_rows + d * route_bytes + e * cache_bytes + f * Engram_bytes`.

Adapter 拥有 coefficient 和 phase-specific term. Rust 拥有 decision 并记录 estimate. Dispatch 后比较 estimate 和 observed per-rank telemetry. DeepSeek 在 service level 报告了相同原则: decode balancer 同时考虑 request count 和 total KV length ([DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L117-L119)).

DeepEP 的 exact CPU count 适合 calibration 或 slow control path, 不适合每个 latency-sensitive step. Cached handle 可以跳过 layout work, 但在 combine 完成前其 route metadata 必须保持不变. 把 handle 视为 Python-owned execution state, 不是 Rust ownership.

## 2. 安全调度规则

### Request placement 和 packing

1. 在 admission 或已完成 step 的边界分配 `owner_rank`. 在完整 step 中保持它.
2. 按 phase packing request: 先 decode, 再 complete prefill chunk, 最后 residual prefill. 保持现有 Rust priority ([runtime contracts](../requirements.zh.md)).
3. 应用三个 limit: global rows, predicted per-rank weighted cost 和 route-buffer bytes. 只有同时满足三个 limit 的 request 才能进入.
4. 保持稳定的 request order 和 row offset. Adapter 必须返回 source-row ID, 让 combine 恢复 owner order.
5. 允许 expert dispatch 在 expert rank 之间拆分 token row. 不拆分 request ownership. Owner rank 保留 model-local route state 和 output ownership. Rust 保留 accepted-token state 和 state commit.

当前 scheduler 提供 computed tokens, draft tokens, row totals 和 cache assignment, 但没有 expert estimate ([`Request`](../../crates/scheduler/src/request.rs), [`planning.rs`](../../crates/scheduler/src/planning.rs)). DeepSeek policy 应添加 model-specific estimate, 不要修改 Qwen quota rule.

### Rebalance 和 migration

使用 expert replication 作为主要 balance action. 第一方 EPLB source 复制 hot logical expert, 并返回 physical-to-logical 和 logical-to-physical map ([`rebalance_experts`](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913/eplb.py#L131-L166)). DeepSeek states prefill 使用 100-step statistics window 和 1000-step reassignment, decode 使用 1000-step window 和 10000-step reassignment ([Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L93-L103)). 这些值是起始值, 不是项目固定 limit.

只在 quiescent step boundary 切换 expert map. 所有 rank 必须同意 `map_version`, map-content digest, layer scope, topology, checkpoint identity 以及 weight residency 和 capacity check. Load 需要的 replica weight, 并在下一个 collective plan 使用新 map. 任何 live dispatch handle 都不能跨越切换.

在最小 contract 中避免 request migration. 如果以后需要 migration, 等待 all-rank completion. Snapshot accepted state, transfer 或让新 owner 使用 cache, allocate 新 transaction, 再在后续 step dispatch request. 不要移动 detached state, live route handle 或 speculative state. Transfer 失败时保留旧 owner, abort 新 transaction.

### Overload, zero rows 和 collectives

- 如果 predicted load 过高, 缩小 prefill chunk, defer request, 或使用更空的 batch. 不要静默丢弃 row, 也不要在 dispatch 后改变 route.
- 如果 observed load 超过 route-buffer limit, 尽可能在 model execution 前失败 step. Typed collective 或 rank error 按当前 group failure policy 处理. 不能产生 partial state commit.
- Row 为零的 rank 仍接收相同的有序 dispatch 和 collective plan, 在有效处发送 zero route count, 并报告 completion. 不要在该 rank 跳过 collective. 当前 group contract 明确调用此规则 ([`distributed.md`](../distributed.zh.md#事务和发布)).
- 所有 rank 使用相同的 plan digest. 添加 request order, owner rank, row slice, `map_version`, map digest, route matrix, alignment, valid-row metadata 和 operation order. 现有 collective digest 和 all-rank completion check 提供模式, 但还不包含这些 model fields ([`collectives.rs`](../../crates/zmq-worker/src/group/collectives.rs), [`GroupGate::insert_result`](../../crates/zmq-worker/src/group.rs)).
- 整个 group 对 prefill 使用 Eager. 添加 Graph eligibility gate, 检查所有 rank 的 shape, route count, address 和 operation order. 当前 group 在 prefill 后选择 Graph, 但还没有检查这些 model-specific condition ([`GroupGate::dispatch`](../../crates/zmq-worker/src/group.rs)).

普通 request cancellation 停止该 request 的后续 admission. 当 device work 仍可能读取它们时, 它不能释放 transaction, cache block, Engram buffer 或 route handle. Exchange 中的 cancellation 会让 scheduler 保持 disabled, 并保持 transaction owned. 释放 ownership 或替换 scheduler 和 epoch 前, 必须先 close 或 reap group ([`distributed.md`](../distributed.zh.md#原生完成和关闭), [`state-arena.md`](../state-arena.zh.md#lease-生命周期)).

## 3. 为什么 request count 或 token count 不够

Request count 测量 batch width, 但不测 prompt length, verification rows, context length, cache bytes 或 hot expert. 一个 long decode request 的 attention work 可能超过许多 short request.

Token count 测量 input rows, 但不测 top-k route 的去向. 两个 token 数相同的 batch 可能把大多数 token 放到一个 expert, 产生不同 alignment padding, 以及不同 send 和 receive traffic. Hot expert 可能让其 rank serialise, 而其他 rank idle.

两种 count 都不包含 Engram lookup traffic, sparse attention candidate, cache restoration, route metadata, expert weight residency, 或 prefill 与 recurrent decode kernel 的差异. 因此 DeepSeek 的公开 Q&A 同时按 request count 和 total KV length 平衡 decode, 并使用 hot expert replica 平衡 expert. 仓库也必须使用相同的 multi-signal 原则, 同时把 model rule 保持在 adapter 中.

## 4. 最小可验证 contract 和验收

最小 DEP4 contract 应在 dispatch 前验证以下字段:

- `epoch`, `step_id`, 唯一 request 和 transaction ID, 四个有序 rank, `owner_rank`, phase, row slice 和 cache binding;
- 跨 step 持久的 DEP4 owner invariant, 将 `owner_rank` 与 request ID, transaction identity, state generation 和 cache ownership 绑定. 改变 owner 需要显式且已完成的 state-transfer protocol;
- model-owned `estimate_version`, 以及 prefill/decode rows, context/cache bytes, predicted expert histogram, active-expert count, route bytes 和 Engram bytes;
- Python-owned observed telemetry, 以及 per-layer expert histogram, active expert, aligned 和 unaligned rows, rank send/receive bytes, attention/cache/Engram bytes, route-handle identity 和 completion time.
- 一个 deterministic plan digest. 包含 request order, owner 和 row metadata, expert `map_version` 以及 map-content digest. 还包含 layer scope, topology, checkpoint identity, route count, padding, valid-row metadata, collective order 和 execution mode;
- 四个 rank 的 completion, exact transaction set, 每个 owner request 一个 output, 并且 Rust 在验证完整 set 前不接受 state.

使用 synthetic route matrix 在 CPU 上测试 failure path. 覆盖 stale map, route overflow, duplicate request 和 missing zero-row completion. 覆盖 mismatched digest, rank timeout, worker error 和 exchange 中 cancellation. 覆盖 failed state transfer 和 stale transaction generation. 使用 descriptor check 做 Qwen isolation. 使用 synthetic schedule check 做 Qwen isolation. 共享 runtime code 变化时运行完整 Qwen HTTP 和 AgentX regression.

建议 performance gate 如下.

- Steady-state max-rank/mean expert time 不高于 `1.10`.
- warmup 后, per-step max-rank/mean-rank wall-time ratio 的 p95 不高于 `1.20`. 包含 dispatch, combine, attention, Engram, transfer 和 expert time.
- Zero-row step 完成且没有 collective 遗漏.
- 报告 observed-to-predicted cost error 的 phase.
- 不产生 post-cancel commit 或 use-after-free.

同时报告 request latency, prefill 和 decode throughput, route bytes, cache 和 Engram bytes 以及 per-rank memory. 这些是验收 target, 不是 measured result.

## 来源

- [DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md)
- [DeepEP `EPHandle` 和 `dispatch`](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py)
- [EPLB `rebalance_experts`](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913/eplb.py)
- [`oh-my-vllm` distributed contract](../distributed.zh.md)
- [`oh-my-vllm` runtime contracts](../requirements.zh.md)
- [`oh-my-vllm` state ownership](../state-arena.zh.md)
- [`oh-my-vllm` DeepSeek seams](repository-deepseek-seams-and-gates.zh.md)
