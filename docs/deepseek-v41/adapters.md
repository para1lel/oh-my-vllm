# Model adapter extraction

## Current scope

Phase 1 extraction uses the Python adapter entry.
The [registry](../../python/oh_my_vllm/models/registry.py) selects an implemented family from configuration architecture and model type.
Directory names do not select a family.
Only Qwen is registered.
DeepSeek configuration still causes a startup error, not execution through the Qwen backend.

The [adapter interface](../../python/oh_my_vllm/models/api.py) supplies `ready`, `handle`, and `close` methods.
The bridge owns ZMQ DEALER and msgpack transport.
The adapter does not own transport.
The `stub` path still selects the test placeholder directly.
All other paths use the registry.
The Qwen computation and accepted-state rules are unchanged.

Phase 1 extraction is complete.
The [nightly Qwen report](qwen-regression.md) records the full HTTP regression matrix.
Phase 2 supplies general cache transactions and four-rank control.
These interfaces do not enable distributed execution or DeepSeek capabilities.

## Python ownership

The [Qwen directory](../../python/oh_my_vllm/models/qwen38) owns weights, cache layouts, model computation, sampling, MTP4, request state, tokenizer, and media processing.
Its [offload implementation](../../python/oh_my_vllm/models/qwen38/offload.py) owns neural snapshots and restoration.
The [stub module](../../python/oh_my_vllm/models/stub.py) owns simulated offload for the explicit test backend.

The shared worker keeps transport, execution policy, metadata uploads, repetition checks, trace records, host-memory budgets, and constraint machinery.
Its transfer helpers still validate the existing paired-cache wire format.
These helpers do not supply model-independent state transactions.

The [Qwen tool grammar](../../python/oh_my_vllm/models/qwen38/tool_grammar.py) supplies model-specific token syntax to shared constraints.
Enabled tools must have an explicit grammar callback.
Missing callbacks cause an error, not selection of Qwen syntax.
A tool grammar callback is not necessary for JSON object and JSON schema constraints.

The old `worker.checkpoint` module only imports the Qwen checkpoint class.
This compatibility path keeps the frozen Qwen reference generator unchanged.
Other model modules use the new family paths.
The numerical fixtures and acceptance limits are unchanged.

## Rust output ownership

The [shared output interface](../../crates/zmq-worker/src/models/output.rs) supplies reasoning, text, and tool deltas.
The [Qwen parser](../../crates/zmq-worker/src/models/qwen38/output.rs) owns incremental tag scanning and tool argument parsing.
Readiness validation returns an internal model family only after all descriptor checks pass.
The worker client stores that family for its lifetime.
The service selects an output parser from this stored family, not from the public model name.

The explicit test stub uses Qwen output syntax for service simulations.
It cannot select a neural model after a startup error.
Unknown descriptors still cause a startup error.
This move does not change Qwen stream parsing or public response fields.

## Selection and startup

The [Rust policy selector](../../crates/zmq-worker/src/models/policy.rs) reads original configuration bytes before socket binding or worker startup.
It selects the family from architecture and model type, then verifies its supported format and configuration identity.
Only the literal `stub` path selects the test policy.
Unknown configurations cannot select that policy.

Rust and the Python loader each reject configuration inputs above 1 MiB.
The Python loader repeats selection before GPU backend imports.
It rejects duplicate JSON keys and configuration values that are not objects.

The architecture list must contain exactly the registered architecture.
The model type must also match.
Unknown architectures and incompatible formats cause an error before GPU backend imports.
Requested capabilities must be a subset of the selected adapter's capabilities.

The [Qwen entry](../../python/oh_my_vllm/models/qwen38/adapter.py) uses family key `qwen38`.
Its original checkpoint declares architecture `Qwen3_5ForConditionalGeneration` and model type `qwen3_5`.
These original identifiers stay unchanged.
The checkpoint must use dynamic E4M3 FP8 quantization with 128-by-128 weight blocks.
The Qwen entry compares the consumed configuration bytes with the frozen Qwen configuration SHA256 before it imports the backend.
The existing weight loader repeats its checkpoint check during loading.

Registry selection alone does not prove weight compatibility or numerical acceptance.
It does not examine weight payloads, allocate cache state, or change context limits.
The Qwen backend keeps its existing startup checks and numerical gates.
The [phase plan](plan.md) specifies the subsequent extraction and regression work.

## Rust model policy

The [Qwen policy](../../crates/zmq-worker/src/models/qwen38/policy.rs) owns model-specific defaults, vocabulary limits, speculation rules, and readiness checks.
The shared CLI uses the selected policy for options that the caller does not supply.
Explicit options must pass that policy's validation.
The paired-cache scheduler uses the resolved block unit.
This change keeps the current cache transaction format.

`WorkerConfig::default()` keeps the legacy Qwen preset for existing library callers.
It does not select defaults for another model family.
The CLI selects the model before it constructs the worker configuration.
Model selection and option errors occur before IPC binding or process startup.

`launch_with_model` repeats CPU selection before it starts the worker.
A different selection causes an error.
Ready must match the selected policy.
Worker recovery keeps the initial model policy and rejects a different family.

The validated policy supplies the default HTTP model name.
An explicit served model name changes only the public name, not model selection.
Qwen keeps its original defaults and accepted modes.
These checks do not read weight payloads or prove GPU capacity.

## Rust description contract

The [model-contract crate](../../crates/model-contract/src/lib.rs) supplies versioned types for readiness agreement.
It defines `ModelDescriptor`, `CacheSchema`, `ParallelPlan`, and `SpeculationSpec`.
Deserialization rejects unknown fields and enum values.
Call `validate()` before a consumer uses a descriptor.
Deserialization alone does not validate versions or structural relationships.

The descriptor binds checkpoint, configuration, and tokenizer SHA256 identities, plus vocabulary, context, capabilities, formats, cache groups, parallel strategy, and speculation.
Identity validation examines syntax only, not file content.
Cache groups identify storage, contributing layers, token units, capacities, and owned or shared producers.
The validator rejects duplicate groups, missing producers, self references, and cycles.
Metadata count and name-length limits bound structural validation.
Transport code must set an input byte limit before deserialization allocates storage.

Parallel plans use logical ranks, not physical GPU indices.
TP1 uses one rank, and TP4, TEP4, and DEP4 each use four ranks.
Only DEP4 partitions requests by rank.
Speculation keeps predictor stages separate from draft counts and verification rows.
MTP can have one stage with four drafts, and DSpark can have three stages with five drafts.
Verification capacity equals the maximum draft count plus one.

The family registry must validate supported layouts, dimensions, and family-specific combinations.
The Rust types do not enable execution, validate weight contents, or replace rank readiness and state transactions.
The Ready message contains the model descriptor.

## Python description contract

The [Python validator](../../python/oh_my_vllm/models/descriptor.py) uses the same versioned named-map fields as Rust.
It rejects unknown fields, invalid values, duplicate groups, invalid producer chains, and integers that do not fit their types.
Boolean values do not substitute for integers.
These checks use CPU data only.

The [Qwen builder](../../python/oh_my_vllm/models/qwen38/descriptor.py) uses verified original configuration bytes and loaded checkpoint and tokenizer identity maps.
It does not read checkpoint files or import a GPU backend.
The caller must verify file bytes before supplying their identities.
The parsed configuration must match the verified bytes.
The value types must also match.

Each identity map associates a file label with its lowercase SHA256 digest.
The checkpoint map uses `config` and `index` labels, plus the file names of all loaded shards.
The tokenizer map includes `tokenizer.json`, `generation_config.json`, and `chat_template.jinja`.
The helper sorts keys and encodes compact ASCII JSON without a final newline.
The SHA256 input starts with one versioned domain and ends with the JSON bytes:

```python
b"oh-my-vllm/checkpoint-identity/v1\0"
b"oh-my-vllm/tokenizer-identity/v1\0"
```

Changing a file name, digest, or domain changes the identity.
Changing cache capacity does not change checkpoint or tokenizer identity.
The helper hashes all supplied identity entries.
It does not prove file content or memory capacity.

Qwen supplies the existing 262144-token limit, TP1, and MTP limits of one stage, four drafts, and five verification rows.
The MTP limit does not enable MTP execution in ordinary-decode mode.
Configured cache membership includes the predictor FA array only for MTP4.

The two cache groups use versioned Qwen layout IDs.
FA storage uses BF16.
The GDN layout includes FP32 recurrent state, BF16 convolution history, and a BF16 boundary hidden-state array.
Their capacities are allocated scheduler pool counts.
Pool counts include null block or slot zero.

Private slots, scratch slots, host storage, and execution workspaces do not increase scheduler capacity.
The layouts keep the existing paired reservation and commit rules.
They are not independent state transactions or shared-producer stores.

The worker uses the builder after cache allocation.
The registry supplies the verified original configuration bytes without a new file read.
Before Ready, the checkpoint reader hashes all index shards and verifies the file bytes again.
This step prevents a later lazy vision load from changing the checkpoint file set.
The tokenizer also verifies its original asset bytes.
Ready returns a private descriptor copy.

After Ready, the reader verifies shard file states before and after each tensor read.
It verifies the frozen digest again before the first read of each shard after Ready.
After vision weight loading, a full identity check occurs before the model can use the new weights.
File timestamps alone do not prove unchanged file contents.

## Readiness agreement

Init must specify `descriptor_version=1`.
Python rejects a missing or unsupported version before adapter construction.
Ready must supply a descriptor.
An older message cannot use default descriptor values.
Rust validates its schema and implemented model family before scheduler construction.

It compares the cache layouts, layer membership, block units, and allocated capacities with the selected mode.
Legacy FA and GDN capacity fields must match their descriptor groups.
The request context limit must not exceed the supported model limit.
Qwen keeps TP1, ordinary decode, and MTP4.

The explicit `stub` backend has separate test identities and metadata-only cache layouts.
Its `f32` weight-format entry is a test wire sentinel, not loaded model weights.
Its capability entries specify service-test simulations, not neural inference or media decoding.
Rust accepts this descriptor only for the `stub` path.
Qwen cannot use it after a load error.

Readiness agreement does not supply numerical or GPU capacity acceptance.
The [phase plan](plan.md) specifies Qwen regression checks and the next steps for model-specific control extraction.
