# Repository DeepSeek Seams and Gates

Audit date: 2026-10-26. Scope: read-only inspection of the current repository and the historical DSV4.1 validation plan. This report maps integration seams. It does not claim that DSV4.1 service inference is implemented.

## Current boundary

The repository currently selects only the Qwen adapter or the explicit `stub` backend. The Python registry contains one adapter, and the Rust selector contains `Qwen38` and `Stub`. Unknown architectures fail before model startup. See [architecture](../architecture.md#process-boundary), [adapter ownership](../adapters.md#ownership), [Python registry](../../python/oh_my_vllm/models/registry.py), and [Rust model selector](../../crates/zmq-worker/src/models/policy.rs).

The current model tree has no DSV4.1 adapter, model policy, checkpoint reader, codec, or DSV4.1 benchmark. The historical DSV4.1 files were removed from the current tree. The checkpoint contract report records the model-side facts and unresolved choices. See [checkpoint contract](dsv41-model-checkpoint-contract.md).

## Seam map

| Seam | Current owner | DSV4.1 work at this seam |
|---|---|---|
| Adapter selection | Python `registry.py`, Rust `models/policy.rs` | Add a DSV4.1 family. Match architecture and model type from original config bytes. Validate supported formats and identity before GPU imports. |
| Descriptor and readiness | `models/descriptor.py`, `crates/model-contract`, `model_ready.rs` | Define DSV4.1 capabilities, cache groups, rank plan, tokenizer identity, context limit, and DSpark5 limits. Make Python and Rust validators accept the same fields. |
| HTTP and admission | Rust `serving` module | Reuse HTTP routing and request lifecycle. Keep DSV4.1 request limits, context checks, overflow behavior, and served name in model policy or an explicit service contract. |
| Scheduling and logical cache | Rust `scheduler` and `kv-cache` crates | Reuse admission, token budgets, preemption, state commit, and logical ownership. Supply DSV4.1 cache groups, block sizes, prefix rules, restoration, and request placement through model policy. |
| ZMQ worker boundary | `worker/zmq_bridge.py`, `protocol.rs` | Reuse DEALER and msgpack transport. Keep model computation and token syntax in the adapter. Add only message fields required by the DSV4.1 rank or state contract. |
| Rank group and transactions | `rank_client`, `group`, `group/collectives.rs`, Python distributed runtime | Define four-rank ownership for TP4, TEP4, and DEP4. Bind rank order, group epoch, transaction IDs, request placement, output ownership, and collective plan to readiness and execution. |
| Python model execution | `models/qwen38`, shared `worker/execution.py`, native operators | Add a DSV4.1 adapter for weights, target forward, DSpark proposal and verification, codecs, media, cache state, and accepted-state restoration. Keep shared execution responsible for lifecycle and resource ownership. |
| Output parsing and constraints | Rust model output parser and shared constraints | Add a DSV4.1 output parser and grammar integration. Keep proposal state separate until verification and update history only after accepted output. |
| HTTP benchmark | `development/serving/benchmark.py` and Qwen performance docs | Add a DSV4.1-specific runner or a model-neutral configuration layer. Bind strategy, DSpark5, batch, exact token counts, sampling, identity, and artifact accounting. |
| AgentX benchmark | `development/serving/agentx.py` and AgentX docs | Add DSV4.1 strategy and tree-concurrency parameters. Record observed HTTP concurrency, queueing, overflow, cancellations, and result status counts. |

The [adapter guide](../adapters.md#ownership) is the ownership rule. Rust owns service control, scheduling, logical cache identities, and validated state commit. Python owns tensors, private model state, computation, codecs, and model-specific speculation. The bridge owns transport.

## Descriptor and strategy seam

The shared descriptor already has `tp1`, `tp4`, `tep4`, and `dep4` parallel strategies. Its validation checks world size `1` or `4`, logical ranks `0` through `world_size - 1`, and `rank_partitioned` placement only for `dep4`. It currently has only `Mtp` as a speculation method. See [descriptor types](../../crates/model-contract/src/lib.rs) and [descriptor validation](../../crates/model-contract/src/validation.rs).

Historical DSV4.1 planning uses a four-rank group and DSpark5. The model-side values are three predictor stages, five draft tokens, and six verification rows. The historical descriptor shape is therefore a useful seam, but it is not a current runtime contract. See the [historical validation plan](https://github.com/para1lel/oh-my-vllm/blob/76514a7/docs/models/deepseek_v41/validation.md) and [checkpoint contract](dsv41-model-checkpoint-contract.md#dspark).

The smallest safe descriptor seam is a model-local DSV4.1 policy that supplies one validated descriptor. Do not make `tp4` or `dspark` globally imply DSV4.1 behavior. The policy must bind strategy, context, cache groups, rank placement, weight formats, and speculation limits as one identity.

## Rust service, scheduler, and cache gates

The HTTP service reserves request state before body processing. Its resource budget has `64` queued reservations and `64` live reservations. The engine derives its running-request limit from `max_num_seqs`, capped at `64`. Current Qwen defaults set this to `4`. See [service resources](../../crates/zmq-worker/src/serving/resources.rs), [service engine](../../crates/zmq-worker/src/serving/mod.rs), and [Qwen policy](../../crates/zmq-worker/src/models/qwen38/policy.rs).

The scheduler and KV crate provide the reusable seam for admission, token budgets, preemption, logical blocks, prefix publication, and state commit. They do not prove that a model cache layout is correct. DSV4.1 must define its cache groups, block units, shared producers, rollback state, and restoration rules before the generic scheduler can be used for acceptance. See [architecture](../architecture.md#control-and-state), [scheduler](../../crates/scheduler/src/lib.rs), and [KV cache](../../crates/kv-cache/src/lib.rs).

The target HTTP matrix uses batch `4`, `16`, and `64`, with input lengths `400000` and `800000`, and output lengths `64` and `8192`. The largest request context is `808192`, below the historical `1048576` limit. The current Qwen policy rejects TP4, contexts above `262144`, and speculation other than ordinary decode or MTP4. Therefore all DSV4.1 matrix rows conflict with the current Qwen policy.

## ZMQ, rank, and collective gates

The single-worker protocol carries `Init`, `Prepare`, `Execute`, cache operations, `Ready`, prepared prompt IDs, execute results, and errors. The Python bridge uses one ZMQ DEALER socket and msgpack frames. See [worker protocol](../../crates/zmq-worker/src/protocol.rs) and [ZMQ bridge](../../python/oh_my_vllm/worker/zmq_bridge.py).

The rank protocol adds group epochs, ordered GPU identities, one endpoint per owned process, readiness nonces, transaction checks, output ownership, and all-rank completion. The collective plan validates operation count, topology, route dimensions, buffer sizes, and a digest shared by Rust and Python. See [distributed contracts](../distributed.md#rank-startup), [transactions](../distributed.md#transactions-and-publication), [collective plans](../distributed.md#collective-plans-and-storage), [rank client](../../crates/zmq-worker/src/rank_client.rs), and [Python distributed runtime](../../python/oh_my_vllm/operators/runtime/distributed.py).

The smallest distributed seam is a four-rank DSV4.1 client model that translates one logical schedule into one rank-local model plan. TP4 and TEP4 replicate request work. DEP4 keeps request ownership by rank. A descriptor alone cannot prove that expert ownership, dispatch, combine, output ownership, and cache state agree.

## Python worker and operator gates

The shared worker keeps transport, execution policy, transfers, constraints, tracing, and resource lifetime. The selected adapter owns model dimensions, token syntax, weights, computation order, physical cache, media, and speculation thresholds. See [worker instructions](../../python/oh_my_vllm/worker/AGENTS.md), [execution rules](../execution.md), and [Qwen adapter](../models/qwen38/README.md).

The current Qwen adapter is isolated in `python/oh_my_vllm/models/qwen38/` and uses Qwen-specific FP8 layout, cache groups, tokenizer, output parser, MTP4 state, and native operators. The output parser registry also has only Qwen behavior. A DSV4.1 adapter must not add model dimensions or DSpark rules to shared Qwen modules.

The main Python gates are checkpoint and tokenizer identity, descriptor validation, readiness capacities, and operator precision and shape checks. Other gates cover cache ownership, proposal rollback, state restoration, and native completion. Shared execution tests can check lifecycle and failure paths, but they cannot prove DSV4.1 numerical correctness or capacity.

## Benchmark and acceptance gates

The current HTTP benchmark is Qwen-specific. It hard-codes the served model name, TP1, context `262144`, maximum sequence count `4`, block size `784`, token budget `32768`, and modes `ordinary` and `mtp4`. It starts a full service and checks tools and exact token usage. It records an execution trace and binds artifacts to source, binary, native modules, checkpoint, tokenizer, device, and workload identities. See [benchmark implementation](../../development/serving/benchmark.py) and [Qwen performance procedure](../models/qwen38/performance.md#measurement-command).

The current AgentX launcher hard-codes MTP4, context `262144`, concurrency `4`, dataset identity, seed `42`, and a minimum profiling time of `900` seconds. Its documentation states that child agents can create more HTTP requests than the session-tree concurrency. See [AgentX implementation](../../development/serving/agentx.py) and [AgentX procedure](../models/qwen38/agentx.md#configuration).

The historical DSV4.1 plan defines `38` HTTP cases: `36` DSpark5 cases plus `2` ordinary TP4 cases. It defines `3` AgentX cases: tree concurrency `4` with TP4, `16` with TEP4, and `64` with DEP4. It records observed HTTP concurrency and queueing. No current runner can express this matrix.

## Known gates and risks

- Model selection gate: DSV4.1 config is currently unsupported.
- Descriptor gate: `dspark` and DSV4.1 weight formats are absent from the current shared contract.
- Readiness gate: no DSV4.1 reader supplies a descriptor, capacities, or cache schema.
- Rank gate: no current service path launches a DSV4.1 four-rank model group.
- Collective gate: no DSV4.1 operation plan proves expert dispatch, combine, or replicated output semantics.
- Numerical gate: no current DSV4.1 reference adapter, frozen fixtures, or accepted limits exists.
- Memory gate: the model has large Engram tables and long-context state. The checkpoint report records unresolved host residency and cache representation choices.
- API gate: no DSV4.1 codec, chat template decision, media path, or output parser is connected.
- Capacity gate: the `1048576` context boundary and `64` active-request requirement are untested.
- Performance gate: the historical matrix has no current executable runner or result document.

The handoff keeps public DeepSeek inference disabled. `LOCAL.md` and recorded v6/v7 TP4 diagnostics show that preparation stopped at the unchanged 1 GiB gate. These records are evidence only. They do not prove preparation, execution, or serving success.

The main Qwen isolation risks are shared enum expansion, shared defaults, shared output parsing, shared cache assumptions, and benchmark reuse. Keep DSV4.1 rules behind its model directory and policy. Extend shared contracts only when the field is genuinely model-neutral, and add a DSV4.1 policy check that rejects Qwen-only values.

The smallest integration path has clear stages. Add DSV4.1 selection and identity. Add a model-local descriptor and policy. Implement the Python adapter and codec.

Connect four-rank plans. Pass CPU and collective gates. Run small HTTP correctness cases. Then run capacity, the `38` HTTP cases, and the `3` AgentX cases.
