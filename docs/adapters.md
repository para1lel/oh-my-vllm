# Model adapters

Each supported family owns its implementation and acceptance status in its [model directory](models).
Only implemented adapters can start a service.

## Ownership

| Layer | Responsibility |
|---|---|
| Rust service | HTTP, CLI, admission, transport, worker lifecycle |
| Rust scheduler | Request queues, token budgets, cancellation, validated state commit |
| Rust model policy | Defaults, capabilities, readiness, output parsing, cache assignments |
| Python model adapter | Weights, computation, physical state, codecs, media, speculation |
| Shared Python worker | ZMQ/msgpack, execution policy, transfers, constraints, repetition checks, tracing |
| Operators | Native computation, execution plans, workspace ownership |

The [Python interface](../python/oh_my_vllm/models/api.py) supplies `ready`, `handle`, and `close`.
Adapters do not own transport.
The [cache policy](cache-policy.md) separates scheduling from model storage semantics.
The [state arena](state-arena.md) and [rank control](distributed.md) supply typed ownership and group agreement.
These interfaces do not prove model correctness or GPU capacity.

Model code belongs in `python/oh_my_vllm/models/<family>/` and the related Rust model module.
Do not put model dimensions, token syntax, or speculation counts in shared mechanisms.
Share operators only when their computation and resource contracts agree.
Keep interfaces compatible with beneficial fusion and prepared device metadata.

## Selection and readiness

The [Python registry](../python/oh_my_vllm/models/registry.py) and [Rust selector](../crates/zmq-worker/src/models/policy.rs) use original configuration bytes.
Architecture and model type select the adapter. Directory names and public HTTP names do not.
Unknown architectures, unsupported formats, and capability mismatches cause startup errors before GPU imports or process startup.

Configuration input has a 1 MiB limit.
Python rejects duplicate JSON keys and non-object configuration values.
Each family verifies its supported formats and its declared checkpoint identity.
A selected adapter supplies defaults and validates explicit options.
A public served-model alias cannot change the selected family.

Init specifies `descriptor_version=1` and `protocol_version=2`, with the worker epoch.
Ready returns the same protocol version and epoch, a validated descriptor, and a model-selected cache layout before scheduler construction.
The [Rust types](../crates/model-contract/src/lib.rs) and [Python validator](../python/oh_my_vllm/models/descriptor.py) use the same versioned named-map fields.
They reject unknown fields, unsupported versions, invalid values, duplicate groups, and invalid producer chains.
Python boolean values cannot substitute for integers.

Descriptors bind checkpoint, configuration, tokenizer, vocabulary, context, capabilities, formats, cache groups, parallel strategy, and speculation.
Shared validation checks structure. The family policy checks supported combinations and loaded capacities.
Metadata count and name limits bound validation work.
Transport also limits bytes before deserialization.

Speculation keeps predictor stages separate from draft count and verification capacity.
Verification capacity is the maximum draft count plus one.
A descriptor does not enable an unimplemented execution mode.

## Execution protocol

The [wire types](../crates/zmq-worker/src/protocol.rs) use version 2 named-map msgpack through ZMQ DEALER.
Init and Ready must agree on protocol version, worker epoch, descriptor version, and the selected cache layout.
The [layout validator](../crates/zmq-worker/src/protocol/layout.rs) checks the model-specific layout and capacities.
Replacement cannot silently change the accepted layout.

| Field | Values and meaning |
|---|---|
| `purpose` | `advance` changes generation progress. `finalize` completes accepted state |
| `action` | `forward` computes an explicit token range. `head_only` uses saved hidden values. `export_only` exports an endpoint |
| `sampling` | `none`, `target`, or `verify` with an explicit draft count |
| `forward_start`, `forward_end` | Explicit half-open range, independent of the returned token count |
| `accepted_end`, `endpoint` | Accepted history and the selected operation endpoint |
| `model_state` | Tagged model-specific assignment |

Non-forward actions have zero token rows and equal forward bounds.
Finalization uses `none` sampling. A saved-hidden head operation uses the current request's target sampler.
A successful `none` result emits zero tokens. A successful `target` result emits one token.
A successful `verify` result emits the accepted draft count plus one token.

A scheduled row always has exactly one reply, also when it has zero token rows.

`ExecutionKey` contains `worker_epoch`, `step_id`, `request_id`, and `lease_id`.
The scheduler uses `in_flight: Option<ExecutionKey>` to track execution, independently of token-row count.
The [reply validator](../crates/zmq-worker/src/client/execution/validation.rs) checks the complete output set and execution identity.
RPC identity, rank schedules, endpoint exports, and page valid ends also pass validation before state publication.
Stale, duplicate, unknown, or missing rows cause an error before commit.

The transport tags are `qwen_radix_v1` and `deepseek_resident_v1`.
The scheduler uses typed policy assignments without interpreting those wire tags.
Qwen uses the [independent pools and exact endpoints](models/qwen38/cache.md).
DeepSeek keeps its [resident paired128 layout](models/deepseek_v41/cache.md) and DEP4 owner placement.
All four DEP4 ranks return acknowledgements, also when a rank has no scheduled requests.
DeepSeek completion uses its computed resident endpoint without an additional Qwen final-token computation.

The protocol has no KV host storage messages, host capability, or host byte-budget fields.
Finished-request and preempted-request notifications still release physical state.
Engram host weights and necessary token, metadata, and draft transfers have separate contracts.

## State and output

Rust publishes a result only after output and state validation.
Distributed execution also checks all necessary rank completions.
Python keeps accepted state separate from speculative writes and applies model-specific restoration rules.
Cancellation does not release storage while transfers or kernels can use it.
Use the [execution rules](execution.md) for Graph and transfer lifetimes.

The validated model family selects the incremental output parser.
The client keeps this family for its lifetime and checks it during worker replacement.
Tool grammar comes from the adapter. Enabled tools without a grammar callback cause an error.
JSON object and schema constraints can operate without a tool grammar callback.
Only accepted tokens change constraint and request history.

## Identities and compatibility

Identity validation must check consumed bytes, not only digest syntax or file timestamps.
Checkpoint and tokenizer identities use sorted file-label/digest maps with compact ASCII JSON and no final newline.
The SHA256 input begins with one versioned domain:

```python
b"oh-my-vllm/checkpoint-identity/v1\0"
b"oh-my-vllm/tokenizer-identity/v1\0"
```

Changing file names, digests, or domains changes the identity.
Cache capacity does not change checkpoint or tokenizer identity.
Each loader verifies the files required by its declared startup identity.
Each model's guide specifies the files that its startup identity covers.

The `worker.checkpoint` import is used by the frozen Qwen FP64 generator.
It is not a shared runtime checkpoint interface.
Keep frozen source identities and numerical arrays unchanged during acceptance.
Delete unused migration paths and tests. Keep compatibility only when a current consumer uses it.
