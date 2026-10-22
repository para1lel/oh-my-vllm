# Model adapter extraction

## Current scope

Phase 1 extraction starts with the Python adapter entry.
The [registry](../../python/oh_my_vllm/models/registry.py) selects an implemented family from configuration architecture and model type.
Directory names do not select a family.
Only Qwen is registered.
DeepSeek configuration still causes a startup error, not execution through the Qwen backend.

The [adapter interface](../../python/oh_my_vllm/models/api.py) supplies `ready`, `handle`, and `close` methods.
The bridge owns ZMQ DEALER and msgpack transport.
The adapter does not own transport.
The `stub` path still selects the test placeholder directly.
All other paths use the registry.
The old Qwen execution, cache, sampling, and media implementations are unchanged.

This entry is not the full Phase 1 extraction.
Model descriptor agreement, shared state contracts, and the move of Qwen-specific control are subsequent steps.
These interfaces do not enable distributed execution or DeepSeek capabilities.

## Selection and startup

The loader rejects configuration inputs above 1 MiB.
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

## Rust description contract

The [model-contract crate](../../crates/model-contract/src/lib.rs) supplies versioned types for subsequent readiness agreement.
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
The current worker messages do not yet contain these descriptors.
