# Architecture — oh-my-vllm

Rust owns HTTP serving, request scheduling, accepted token history and logical KV
allocation. Python owns single-device GPU computation. Both processes communicate
through ZMQ DEALER and msgpack; there is no vLLM runtime, scheduler or model adapter.

## Execution flow

1. The Rust frontend normalizes Chat/Responses requests. Python's ServingAdapter
   applies the checkpoint chat template, tokenizer, sampling configuration and
   XGrammar constraints, returning prompt IDs.
2. Rust admits requests, resolves shared prefixes and allocates FA/Mamba tables.
   The scheduler batches prefill, decode and actual MTP draft IDs.
3. Python validates accepted history and physical state addresses, executes the
   Qwen model and samples from target distributions with per-draft grammar masks.
4. Python commits only retained outputs, selects accepted recurrent snapshots,
   saves crossed block checkpoints and generates new MTP proposals.
5. Rust updates accepted history, rolls back rejected speculative positions and
   frees finished requests. Finished-only notifications release Python state even
   when no requests remain scheduled.

See [design.md](design.md) for message fields and scheduler/KV data structures.

## Rust modules

- `crates/kv-cache`: shared logical block pool, chain-hashed prefixes, aligned
  Mamba checkpoints and speculative reservations. Two-phase allocation touches
  reused prefixes before allocating new pages.
- `crates/scheduler`: FCFS waiting/running queues, chunked prefill, token budgets,
  staggered admission, recompute preemption and accepted-draft accounting.
- `crates/zmq-worker`: CLI, OpenAI-compatible HTTP APIs, model process lifecycle,
  cancellation, ZMQ transport and correlated logs.

The target is one B200 and Qwen3.8-27B-FP8 at
`/data0/shared/Qwen3.8-27B-FP8`. Each logical FA page contains 784 tokens. There are
16 FA layers and 48 GDN layers. The CLI's existing `num_gpu_blocks` capacity unit
is preserved for frozen-baseline compatibility: ready reports floor(value/3)
logical blocks. Python now allocates each layer's tensors directly at that logical
capacity; it has no old three-way physical stride or mixed-layout storage.

## Python modules

- `worker/protocol.py`: framework-owned wire dataclasses.
- `worker/model_runner.py`: concrete Worker, persistent request/sampling state,
  cache tensors and execution orchestration.
- `worker/batch_plan.py`: host validation, accepted-state selection and checkpoint
  copies. It does not allocate logical pages.
- `models/qwen.py`: direct safetensors loading, fused projection weight mapping,
  64-layer target model and one-layer MTP head sharing embedding/output weights.
- `worker/sampler.py`: penalties, top-k/top-p, greedy/stochastic target sampling
  and exact verification of deterministic greedy proposals.
- `worker/serving.py`: Transformers/Tokenizers preparation, incremental decoding,
  XGrammar masks, EOS/stop/length handling and reasoning accounting.
- `worker/mtp.py`: shifted MTP cache, boundary features and proposal generation.
- `worker/decode_graph.py`: target and MTP CUDA graph buffers/capture/replay.
- `worker/runtime.py`: dependency identity and loaded-module/library audit.

## Hybrid cache and MTP

GDN state uses [slot,48,128,128], FP32 ordinarily and BF16 with MTP. Causal
convolution state is BF16 [slot,10240,3]. Decode reads the last committed physical
slot and writes candidate snapshots independently. A rejected candidate never
becomes a cached prefix. A token crossing a 784 boundary preserves that exact
snapshot in the corresponding checkpoint slot, even when later drafts also pass.
The zero source slot is immutable. Batch planning forbids write/write and foreign
write/source aliases, including checkpoint copies.

MTP row p combines target hidden[p-1] with input token[p]. A uniform +1 RoPE shift
preserves relative attention; row zero is absent. This makes every row match
Rust's prefix hash and avoids a shared boundary row depending on an uncached suffix.
Target hidden at each 784 boundary is stored with its FA page. If Rust has not yet
allocated the next page, the proposer defers the boundary row and shortens drafts.
It never allocates pages outside Rust. Preemption drops request-local progress;
resumption reconstructs it from the accepted prefix and retained boundary feature.

Greedy proposals are verified against samples from the target distribution for
successive draft prefixes. Matching proposals pass; the first mismatch or bonus
sample terminates verification. This preserves stochastic target sampling without
assuming draft probabilities. Grammar state advances only with retained outputs.

## Device kernels and graphs

Independent FlashInfer TRT-LLM FP8 GEMM handles <=32 rows, with column-major
activation scales and row-major checkpoint scales. Larger inputs use CUTLASS with
row-major scales. FlashInfer 0.6.18's CUTLASS SM100 17..32-row path was nondeterministic
at real model widths and is never selected. Actual-width FP64/repetition tests cover
16/17/20/24/32/33 rows. Quantization preserves per-row 128-element scaling.

Project-owned TileLang kernels implement GDN recurrence, causal convolution, paged
split-KV GQA decode, normalization, partial NeoX RoPE and pointwise fusions.
FlashInfer implements long GDN/FA prefill. GDN prefill explicitly normalizes q/k
because the selected release's advertised normalization flag is unused.

Decode graphs use fixed token/request shapes and dynamic positions, tables and
state addresses. Warmup/capture saves every FA/state destination and restores it
before formal replay. Target and proposer each retain at most 32 graph shapes;
prefill or uncached shapes after that limit execute eagerly. Graph outputs are
consumed before reuse, and recurrent MTP inputs are copied into separate buffers.
`OH_MY_VLLM_ENFORCE_EAGER=1` is diagnostic only. Final performance requires measured
coverage of the relevant shapes, not merely successful graph capture.

## Dependencies and future scope

All high-level dependencies and Rust tools are in conda `oh-my-vllm`; host driver,
CUDA/compiler tools are allowed. Independent FlashInfer/TileLang/third-party Triton cache roots avoid
reuse of old framework artifacts. Startup records exact runtime identity, and
shutdown checks imported modules and mapped libraries for legacy dependencies.

[ADR-006](decisions/ADR-006-independent-runtime.md) describes future model and NVIDIA
backend extensions, single-node multi-GPU collectives, and the local DSpark draft
checkpoint. These require concrete implementations when pursued; there are no empty
interfaces, PD-disaggregation paths or multi-node components in the current runtime.
Historical vLLM/V2 integration decisions remain in earlier ADRs and acceptance data.

### Grouped speculative verification attention

For graph target verification, up to five consecutive queries of a request share
one tiled KV read. Each query retains its own causal length; requests never share
a table or attention normalization. CUDA Graph replay copies ragged starts as
well as tables and lengths, so the same graph supports different per-request
counts. Ordinary one-query decode retains the single-query path; draft-head
verification uses the same grouped path, keyed by token count and request count. Grouped FP64 reference tests cover both split counts, page boundaries, the
shifted first valid position, and regrouping during replay.

Unmasked greedy batches without penalties transfer only selected tokens and row
validity, once for all requests. Each request then verifies its own draft prefix.
Other sampling configurations retain full target-distribution construction. Small
private CPU metadata tensors use nonblocking copies on the same stream as their
GPU consumers; this removes explicit per-copy waits without assuming that
pageable transfers necessarily overlap computation.

When every active MTP request has room for three further cache writes, a proposal
graph computes the initial greedy token and three autoregressive draft/head steps
without intermediate host synchronization. It returns all four proposals in one
transfer. Near an allocation/context boundary, the stepwise proposer still trims
the active set and proposal count. Warmup/capture restores all three speculative
FA destinations; persistent row indices, positions, tables, hidden inputs, model
and cache references remain alive for replay. Both proposer graph families share
the existing32-shape budget.

### Independent FA/GDN capacities (2026-09-22)

`--mamba-blocks N` gives the Rust coordinator an independent GDN pool. FA capacity
remains the worker-reported logical capacity; Python allocates recurrent tensors
at N slots and FA/MTP attention tensors at FA capacity. IDs are local to their
cache group and may have equal numeric values. Admission checks both pools, prefix
lookup reconciles hits across both, and the worker validates each address against
its own tensor capacity. Omission preserves the shared-capacity configuration.
See ADR-007 for rationale and the pending long-context acceptance protocol.

### Long prefill attention

For batches with a query span of at least 1024 tokens, Python gathers only active
FA KV tokens into temporary contiguous tensors and invokes independent FlashInfer
ragged TRT-LLM attention with FP32 softmax. Rust allocations and persistent
`[page,2,784,heads,dim]` tensors are unchanged. Query spans from 128 through 1023
with at least 4096 KV tokens use native paged context attention and the same
zero-copy subpage views as target decode, with FP32 softmax. Other small query
spans use paged FA2. Mixed batches
share this decision, so a long prefill also gathers active decode sequences.
The maximum-context batch4 correctness check includes the resulting temporary
memory; neither Mamba cache precision nor numerical tolerances change.

Long-prefill pointwise work uses eight-token convolution tiles and larger SiLU
blocks; short decode retains its original tile sizes. Large gate/up projections
use the independent CUTLASS dual-SM GEMM. BF16 rounding and per-token FP8 scales
are unchanged and checked against FP64 references.

Convolution, GDN recurrence and RMS normalization accept packed projection views
with explicit token/head strides. Outputs remain dense and recurrent source
snapshots retain their isolation guarantees. Host integer metadata is copied in
one buffer per target/draft group, whose device views retain the backing storage.
Graph inputs still copy into persistent buffers; GDN prefill starts are converted
once to int32 before layer execution.

Small-batch BF16 vocabulary projections use independent FlashInfer CuTe-DSL GEMM.
Residual addition and RMS normalization share one kernel, preserving the BF16
sum before FP32 normalization. MLP SiLU/multiplication and FP8 quantization share
a kernel while preserving both BF16 rounding points and the original scales.
Q/K RMS normalization and partial NeoX rotation share a TileLang kernel, retaining
the intermediate BF16 rounding and packed projection strides. Its fixed256-wide
heads,64 rotary dimensions and theta10000000 match the validated Qwen checkpoint.
It computes and reduces the phase in FP64 before FP32 sin/cos, avoiding amplified
frequency/angle rounding error near the maximum context.
GDN recurrence uses 32-value tiles for at least four sequences, 16 otherwise.

Target decode graphs expose existing 784-token pages to native TRT-LLM attention
as 49 sixteen-token subpages. K/V offset views and `page * 98 + subpage` tables
avoid KV copies and retain Rust page ownership. Table expansion and query/KV
length selection occur once per model execution inside graph capture, so replay
uses current request metadata. Draft attention retains the project kernel to
exclude its absent position zero during decode. Draft prefill with at least 128
query tokens gathers valid KV starting at position one and uses independent
ragged TRT-LLM attention with FP32 softmax; intermediate query spans use FA2.
Planning validates nonempty contiguous queries and KV lengths before skipping
the native operator's redundant active-row check. GDN prefill normalizes Q/K
in one strided kernel with FP32 norms, epsilon 1e-6 and BF16 outputs, avoiding
several large temporary tensors. No cache precision or tolerance changes.
