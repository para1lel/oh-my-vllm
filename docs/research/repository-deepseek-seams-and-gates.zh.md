# Repository DeepSeek seam 和 gate

审计日期: 2026-10-26. 范围: 只读检查当前 repository 和历史 DSV4.1 验收计划. 本报告映射集成 seam. 本报告不声称 DSV4.1 service inference 已实现.

## 当前边界

当前 repository 只选择 Qwen adapter 或显式 `stub` backend. Python registry 包含一个 adapter, Rust selector 包含 `Qwen38` 和 `Stub`. 未知 architecture 在 model startup 前失败. 参见 [architecture](../architecture.zh.md#进程边界), [adapter ownership](../adapters.zh.md#职责), [Python registry](../../python/oh_my_vllm/models/registry.py) 和 [Rust model selector](../../crates/zmq-worker/src/models/policy.rs).

当前 model tree 没有 DSV4.1 adapter, model policy, checkpoint reader, codec 或 DSV4.1 benchmark. 历史 DSV4.1 文件已从当前 tree 删除. checkpoint contract report 记录 model-side facts 和未解决选择. 参见 [checkpoint contract](dsv41-model-checkpoint-contract.zh.md).

## Seam 映射

| Seam | 当前 owner | DSV4.1 在此 seam 的工作 |
|---|---|---|
| Adapter selection | Python `registry.py`; Rust `models/policy.rs` | 添加 DSV4.1 family. 从原始 config bytes 匹配 architecture 和 model type. 在 GPU imports 前验证支持的 format 和 identity. |
| Descriptor 和 readiness | `models/descriptor.py`; `crates/model-contract`; `model_ready.rs` | 定义 DSV4.1 capability, cache group, rank plan, tokenizer identity, context limit 和 DSpark5 limit. 使 Python 和 Rust validator 接受相同 fields. |
| HTTP 和 admission | Rust `serving` module | 复用 HTTP routing 和 request lifecycle. 将 DSV4.1 request limit, context check, overflow behavior 和 served name 放在 model policy 或明确的 service contract 中. |
| Scheduling 和 logical cache | Rust `scheduler` 和 `kv-cache` crate | 复用 admission, token budget, preemption, state commit 和 logical ownership. 通过 model policy 提供 DSV4.1 cache group, block unit, prefix rule, restoration 和 request placement. |
| ZMQ worker boundary | `worker/zmq_bridge.py`, `protocol.rs` | 复用 DEALER 和 msgpack transport. 将 model computation 和 token syntax 保持在 adapter 中. 只添加 DSV4.1 rank 或 state contract 所需的 message field. |
| Rank group 和 transaction | `rank_client`, `group`, `group/collectives.rs`, Python distributed runtime | 定义 TP4, TEP4 和 DEP4 的四 rank ownership. 将 rank order, group epoch, transaction ID, request placement, output ownership 和 collective plan 绑定到 readiness 和 execution. |
| Python model execution | `models/qwen38`, shared `worker/execution.py`, native operators | 添加 DSV4.1 adapter, 负责 weights, target forward, DSpark proposal 和 verification, codec, media, cache state 和 accepted-state restoration. 保持 shared execution 负责 lifecycle 和 resource ownership. |
| Output parsing 和 constraints | Rust model output parser 和 shared constraints | 添加 DSV4.1 output parser 和 grammar integration. 在 verification 前保持 proposal state 分离, 只在 accepted output 后更新 history. |
| HTTP benchmark | `development/serving/benchmark.py` 和 Qwen performance docs | 添加 DSV4.1 专用 runner 或 model-neutral configuration layer. 绑定 strategy, DSpark5, batch, exact token count, sampling, identity 和 artifact accounting. |
| AgentX benchmark | `development/serving/agentx.py` 和 AgentX docs | 添加 DSV4.1 strategy 和 tree-concurrency parameter. 记录 observed HTTP concurrency, queueing, overflow, cancellation 和 result status count. |

[Adapter guide](../adapters.zh.md#职责) 是 ownership rule. Rust 负责 service control, scheduling, logical cache identity 和 validated state commit. Python 负责 tensor, private model state, computation, codec 和 model-specific speculation. Bridge 负责 transport.

## Descriptor 和 strategy seam

Shared descriptor 已有 `tp1`, `tp4`, `tep4` 和 `dep4` parallel strategy. 它要求 world size 为 `1` 或 `4`, logical rank 为 `0` 到 `world_size - 1`, 且仅 `dep4` 使用 `rank_partitioned` placement. 当前 speculation method 只有 `Mtp`. 参见 [descriptor types](../../crates/model-contract/src/lib.rs) 和 [descriptor validation](../../crates/model-contract/src/validation.rs).

历史 DSV4.1 计划使用四 rank group 和 DSpark5. Model-side value 是 3 个 predictor stage, 5 个 draft token 和 6 个 verification row. 因此历史 descriptor shape 是有用的 seam, 但不是当前 runtime contract. 参见[历史 validation plan](https://github.com/para1lel/oh-my-vllm/blob/76514a7/docs/models/deepseek_v41/validation.md) 和 [checkpoint contract](dsv41-model-checkpoint-contract.zh.md#dspark).

最小的安全 descriptor seam 是一个 model-local DSV4.1 policy, 它提供一个经过验证的 descriptor. 不要让 `tp4` 或 `dspark` 在全局隐含 DSV4.1 behavior. Policy 必须把 strategy, context, cache group, rank placement, weight format 和 speculation limit 绑定为一个 identity.

## Rust service, scheduler 和 cache gate

HTTP service 在 body processing 前预留 request state. 它的 resource budget 有 `64` 个 queued reservation 和 `64` 个 live reservation. Engine 从 `max_num_seqs` 推导 running-request limit, 上限为 `64`; 当前 Qwen default 将其设为 `4`. 参见 [service resources](../../crates/zmq-worker/src/serving/resources.rs), [service engine](../../crates/zmq-worker/src/serving/mod.rs) 和 [Qwen policy](../../crates/zmq-worker/src/models/qwen38/policy.rs).

Scheduler 和 KV crate 提供 admission, token budget, preemption, logical block, prefix publication 和 state commit 的可复用 seam. 它们不能证明 model cache layout 正确. DSV4.1 必须先定义 cache group, block unit, shared producer, rollback state 和 restoration rule, 才能使用 generic scheduler 做验收. 参见 [architecture](../architecture.zh.md#控制和状态), [scheduler](../../crates/scheduler/src/lib.rs) 和 [KV cache](../../crates/kv-cache/src/lib.rs).

目标 HTTP matrix 要求 batch `4`, `16` 和 `64`, input length `400000` 和 `800000`, output length `64` 和 `8192`. 最大 request context 是 `808192`, 小于历史 `1048576` limit. 当前 Qwen policy 拒绝 TP4, 超过 `262144` 的 context, 以及 ordinary decode 或 MTP4 以外的 speculation. 因此每个 DSV4.1 matrix row 都与当前 Qwen policy 冲突.

## ZMQ, rank 和 collective gate

Single-worker protocol 携带 `Init`, `Prepare`, `Execute`, cache operation, `Ready`, prepared prompt ID, execute result 和 error. Python bridge 使用一个 ZMQ DEALER socket 和 msgpack frame. 参见 [worker protocol](../../crates/zmq-worker/src/protocol.rs) 和 [ZMQ bridge](../../python/oh_my_vllm/worker/zmq_bridge.py).

Rank protocol 增加 group epoch, ordered GPU identity, 每个 owned process 一个 endpoint, readiness nonce, transaction check, output ownership 和 all-rank completion. Collective plan 验证 operation count, topology, route dimension, buffer size 和由 Rust/Python 共享的 digest. 参见 [distributed contract](../distributed.zh.md#rank-启动), [transaction](../distributed.zh.md#事务和发布), [collective plan](../distributed.zh.md#集合通信计划和存储), [rank client](../../crates/zmq-worker/src/rank_client.rs) 和 [Python distributed runtime](../../python/oh_my_vllm/operators/runtime/distributed.py).

最小 distributed seam 是一个四 rank DSV4.1 client model, 将一个 logical schedule 转换为每个 rank 的 model plan. TP4 和 TEP4 复制 request work. DEP4 按 rank 保持 request ownership. Descriptor 本身不能证明 expert ownership, dispatch, combine, output ownership 和 cache state 一致.

## Python worker 和 operator gate

Shared worker 保持 transport, execution policy, transfer, constraint, trace 和 resource lifetime. Selected adapter 负责 model dimension, token syntax, weight, computation order, physical cache, media 和 speculation threshold. 参见 [worker instructions](../../python/oh_my_vllm/worker/AGENTS.md), [execution rules](../execution.zh.md) 和 [Qwen adapter](../models/qwen38/README.zh.md).

当前 Qwen adapter 隔离在 `python/oh_my_vllm/models/qwen38/` 中, 使用 Qwen-specific FP8 layout, cache group, tokenizer, output parser, MTP4 state 和 native operator. Output parser registry 也只有 Qwen behavior. DSV4.1 adapter 不应把 model dimension 或 DSpark rule 加入 shared Qwen module.

主要 Python gate 是 checkpoint 和 tokenizer identity, descriptor validation, readiness capacity, operator precision 和 shape check, cache ownership, proposal rollback, state restoration 和 native completion. Shared execution test 可以检查 lifecycle 和 failure path, 但不能证明 DSV4.1 numerical correctness 或 capacity.

## Benchmark 和 acceptance gate

当前 HTTP benchmark 是 Qwen-specific. 它 hard-code served model name, TP1, context `262144`, maximum sequence count `4`, block size `784`, token budget `32768`, 以及 `ordinary` 和 `mtp4` mode. 它启动完整 service, 检查 tool 和 exact token usage, 记录 execution trace, 并将 artifact 绑定到 source, binary, native module, checkpoint, tokenizer, device 和 workload identity. 参见 [benchmark implementation](../../development/serving/benchmark.py) 和 [Qwen performance procedure](../models/qwen38/performance.zh.md#测量命令).

当前 AgentX launcher hard-code MTP4, context `262144`, concurrency `4`, dataset identity, seed `42` 和至少 `900` 秒 profiling time. 文档说明 child agent 可以产生多于 session-tree concurrency 的 HTTP request. 参见 [AgentX implementation](../../development/serving/agentx.py) 和 [AgentX procedure](../models/qwen38/agentx.zh.md#配置).

历史 DSV4.1 plan 定义 `38` 个 HTTP case: `36` 个 DSpark5 case 加 `2` 个 ordinary TP4 case. 它定义 `3` 个 AgentX case: tree concurrency `4` 配 TP4, `16` 配 TEP4, `64` 配 DEP4. 它要求记录 observed HTTP concurrency 和 queueing. 当前没有 runner 可以表达这个 matrix.

## Known gate 和风险

- Model selection gate: 当前不支持 DSV4.1 config.
- Descriptor gate: 当前 shared contract 没有 `dspark` 和 DSV4.1 weight format.
- Readiness gate: 没有 DSV4.1 reader 提供 descriptor, capacity 或 cache schema.
- Rank gate: 当前 service path 没有启动 DSV4.1 四 rank model group.
- Collective gate: 没有 DSV4.1 operation plan 证明 expert dispatch, combine 或 replicated output semantics.
- Numerical gate: 没有当前 DSV4.1 reference adapter, frozen fixture 或 accepted limit.
- Memory gate: model 有大型 Engram table 和 long-context state. checkpoint report 记录未解决的 host residency 和 cache representation 选择.
- API gate: 没有连接 DSV4.1 codec, chat template decision, media path 或 output parser.
- Capacity gate: `1048576` context boundary 和 `64` active-request requirement 尚未测试.
- Performance gate: 历史 matrix 没有当前 executable runner 或 result document.

handoff 保持 public DeepSeek inference disabled. `LOCAL.md` 和记录的 v6/v7 TP4 diagnostic 显示准备过程在未改变的 1 GiB gate 处停止. 这些记录仅是 evidence. 它们不能证明 preparation, execution 或 serving success.

主要 Qwen isolation risk 是 shared enum expansion, shared default, shared output parsing, shared cache assumption 和 benchmark reuse. 将 DSV4.1 rule 保持在其 model directory 和 policy 后面. 仅当 field 确实 model-neutral 时扩展 shared contract, 并添加 DSV4.1 policy check 拒绝 Qwen-only value.

最小集成路径是: 添加 DSV4.1 selection 和 identity; 添加 model-local descriptor 和 policy; 实现 Python adapter 和 codec; 连接四 rank plan; 通过 CPU 和 collective gate; 运行小 context HTTP correctness case; 最后运行 capacity, `38` 个 HTTP case 和 `3` 个 AgentX case.
