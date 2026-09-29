# ADR-008: Semantic operator IR and compiled GPU forward

Status: accepted for the tested workset on clean `619c9d9` (2026-09-29).

## Decision

Use PyTorch `torch.library` custom operations as opaque semantic nodes for
project-owned CUDA and key FlashInfer computation. An executable PyTorch
reference plus explicit fake shape/dtype and mutation schema define each
operation. Production providers register independently, with static metadata
capability checks and ordered priorities. The reference is an explicitly
selected debug provider, never an automatic fallback. Eager and compiled
execution share provider selection. Selection freezes registry configuration.

Compile prefill, target decode, MTP draft and four-step proposal with fullgraph
torch.compile and a small FX backend that lowers semantic nodes just before
Inductor. The existing manual CUDA Graph captures compiled GPU units. Rust
scheduling, Python host planning and persistent cache allocation stay outside
the compiled units. Keep a checked call-site inventory; ordinary PyTorch
operations, including `F.linear`, remain outside the semantic operator scope.

Use public `torch.library` and `torch.compile` interfaces where available.
Bounded FakeTensor, FX metadata and Inductor backend lookup calls use PyTorch
2.14.0 internals and require pinned-version regression tests. Keep cache writes
explicit and ordered; persistent KV/GDN buffers cannot be donated. Provider
selection depends on phase, shape, dtype, layout and device, never tensor
values. A failed compile or provider raises rather than falling back to eager.

## Consequences and verification

The model call path names semantics rather than CUDA/FlashInfer entry points.
Base FP8 quantization and fused FP8 linear/SiLU quantization both have semantic
names. Planned attention uses a live CPU handle for host plan state while
passing dynamic decode tables/lengths and native subpage metadata as explicit
graph inputs. Compiled target, draft and proposal closures warm up before
manual capture, and capture restore remains transactional. An exact-output
tested late FX rewrite fuses an adjacent, single-use BF16
`silu_mul -> fp8_linear` chain when the production providers agree. The
compiler adapter disables Inductor CUDA Graphs and raises both PyTorch 2.14.0
recompile limits to 4096. Per-unit compilation counts warn every 256 variants.
Compiler cache growth and eviction/recapture beyond the tested shape families
remain a long-lived-worker limit; 4096 is not a memory-capacity guarantee.
No activation donation is enabled without a proven temporary destination.

`tests/test_ir_*` check registration, contracts, provider selection, static
coverage and real-model fullgraph/CUDA Graph execution. The full existing
correctness and maximum-context suites passed, alongside the
[147-case operator gate](../../bench/baseline/2026-09-29-ir-operators.json)
and [twelve-row framework gate](../../bench/baseline/2026-09-29-ir-framework.json).
The three earlier complete prefix batch-1 collections with TTFT spread above
10% remain rejected in the framework summary; a fourth complete collection
passed. Torch.compile success alone was not treated as performance acceptance.

Design reference: [vLLM IR](https://docs.vllm.ai/en/latest/design/vllm_ir/).
This project implements its own bounded DSL without a vLLM runtime dependency.
