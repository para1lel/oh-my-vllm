# ADR-008: Semantic IR and compiled forward

Date: 2026-09-29. Status: accepted for the measured workset at `619c9d9`.

## Decision

Use `torch.library` operations as semantic nodes for owned CUDA and key FlashInfer operations.
Give each node an executable reference, fake shape/dtype contract, mutation schema, and ordered providers.
Select providers from static phase, shape, dtype, layout, and device metadata, never tensor values.
Keep registry configuration unchanged after selection.
The reference is an explicit debug provider. Provider failure raises an error.

Compile prefill, target decode, MTP draft, and four-step proposal with fullgraph `torch.compile`.
Lower semantic nodes through a small FX backend immediately before Inductor.
Keep Rust scheduling, host planning, and persistent allocation not in the compiled units.
Manual graphs capture compiled units after warmup. Inductor graphs are disabled.

Keep persistent writes ordered and prohibit persistent-cache donation.

## Consequences

The model names semantics instead of low-level entry points.
Eager and compiled paths share provider selection.
Ordinary PyTorch operations, with `F.linear`, are not in the operator inventory.
A checked call-site inventory records each low-level exception.
Bounded FakeTensor, FX metadata, and backend lookups use pinned PyTorch 2.14 internals with regression tests.

Planned attention keeps its CPU handle isolated from dynamic device metadata.
Graph capture restoration stays transactional.
A tested late rewrite fuses the single-use BF16 `silu_mul -> fp8_linear` chain with compatible production providers.
Dynamo limits are 4096 and warnings occur each 256 variants.
These limits are not a compiler-memory capacity guarantee.

No activation donation is currently enabled.

## Verification and alternatives

Target-model integration includes four compiled units and changed-metadata graph replay.
The full correctness/context suites,147 operator cases, and twelve framework rows passed on the identified source.
Three full prefix-batch 1 attempts failed spread. The fourth full collection passed.
Compilation success alone was not performance acceptance.

See [acceptance](../acceptance.md) and [open limits](../audit.md).

The project controls this bounded DSL as an alternative to a vLLM runtime or a general compiler.
The [vLLM IR design](https://docs.vllm.ai/en/latest/design/vllm_ir/) was a design reference.
