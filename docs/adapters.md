# Model adapters

Each supported family owns its implementation and acceptance status in its [model directory](models).
Only implemented adapters can start a service.
The explicit `stub` backend supplies service-test simulations, not model inference.

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
Only the literal `stub` path selects the placeholder.

Configuration input has a 1 MiB limit.
Python rejects duplicate JSON keys and non-object configuration values.
Each family verifies its supported formats and its declared checkpoint identity.
DeepSeek uses config and index metadata for startup identity and does not hash all weight shards before Ready.
A selected adapter supplies defaults and validates explicit options.
A public served-model alias cannot change the selected family.

Init specifies `descriptor_version=1`.
Ready supplies a validated descriptor before scheduler construction.
The [Rust types](../crates/model-contract/src/lib.rs) and [Python validator](../python/oh_my_vllm/models/descriptor.py) use the same versioned named-map fields.
They reject unknown fields, unsupported versions, invalid values, duplicate groups, and invalid producer chains.
Python boolean values cannot substitute for integers.

Descriptors bind checkpoint, configuration, tokenizer, vocabulary, context, capabilities, formats, cache groups, parallel strategy, and speculation.
Shared validation checks structure. The family policy checks supported combinations and loaded capacities.
Metadata count and name limits bound validation work.
Transport also limits bytes before deserialization.

The current Qwen adapter uses TP1 and one rank.
Speculation keeps predictor stages separate from draft count and verification capacity.
Verification capacity is the maximum draft count plus one.
A descriptor does not enable an unimplemented execution mode.

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
DeepSeek reads the required weight tensors during load without hashing shard bytes.
Full shard identity stays in measurement artifacts.

The `worker.checkpoint` import is used by the frozen Qwen FP64 generator.
It is not a shared runtime checkpoint interface.
Keep frozen source identities and numerical arrays unchanged during acceptance.
Delete unused migration paths and tests. Keep compatibility only when a current consumer uses it.
