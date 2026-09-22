# Handoff — 2026-09-22

## Model convolution layout specialization

The CUDA model convolution specializes 10240 channels with token stride16384.
It loads four BF16 weights together, uses a stable sigmoid for FP32 accumulations,
and chooses four rows per block for medium inputs and eight for large inputs.
Weight alignment and conservative disjoint pool/input spans guard the optimized
path; other layouts or possible aliases retain the generic CUDA implementation.
Source states are still snapshotted before candidate writes, including writes
that overwrite another sequence's original source slot.

The13-case dirty-source convolution subset passes all speed decisions. Full
correctness passes169 tests plus28 subtests; the existing ragged convolution test
now includes the real16384 stride alongside10240/20480 without changing references
or tolerances. Extra FP64 checks cover127/128 and4095/4096 rows, weight offsets,
pool/weight aliases, cancellation, near-zero and large signed values, and negative
values near exponential underflow. Written snapshots and untouched slots remain
exact. The initial temporary checker had a CPU/GPU reference-device mismatch;
that harness issue was corrected before the successful supplemental run.

Ordinary/MTP4 eager model FP64 probes pass. Forced-length text with control tokens
and repetition does not establish agentic acceptance. Independent review and
separate TileFoundry/Nsight observations are summarized in
`bench/baseline/2026-09-22-cuda-convolution-progress.json`.
Full operator/framework acceptance remains pending; default stays TileLang.

## Vector GDN recurrence

The clean c40cd5c matrix passes 129/147 cases, with no detected GPU interference.
The remaining 18 failures are recurrent (7), Q/K normalization (6) and convolution
(5). This complete result is separate from subsequent subset improvements.

The model recurrence now assigns eight contiguous key values to each of 16 lanes,
uses vector state/Q/K loads and paired BF16 snapshot conversions, and keeps the
persistent state in FP32 throughout verification. Model heads, base/row alignment
and conservative disjoint storage spans guard restricted pointers. Generic heads,
unaligned views and possible pool/input overlap use the existing generic CUDA
path. Own-source in-place updates remain valid; no TileLang fallback is introduced.

The first two formal subsets pass 7/8; FP32 batch3 fails. Four warps with two
value rows per half warp resolve that case, and the final subset passes 8/8.
Full correctness passes 168 tests plus 28 subtests. Extra FP64 checks cover generic
heads, input/pool offsets, odd token strides, source aliasing, padding-span overlap,
mixed metadata widths, three sequences and same-source writes, without changing
original tolerances. Ordinary/MTP4 eager model probe results and independent
TileFoundry/Nsight observations are recorded in
`bench/baseline/2026-09-22-cuda-recurrent-progress.json`.

Independent review passes. This is partial migration evidence: full operator and
framework acceptance remain pending, and the default backend stays TileLang.

## Vector FP8 quantization and reciprocal refinement

Plain quantization now uses four-element input/output vectors. Host dispatch
checks input alignment once, retaining scalar loads for storage-offset views.
Rows4–127 flatten independent scaling groups into four-warp blocks. BF16 finite
maxima use an approximate reciprocal plus one FMA residual correction; scales
still use exact division. FP16/FP32 and nonfinite maxima retain exact division.
Both BF16 SiLU rounding boundaries remain. This is not FP32 division equivalence.

Temporary SM100/CUDA13.1 exhaustive checks over finite BF16 input/max pairs
with abs(input)<=maximum find
zero FP8 differences after refinement for both signs, whereas direct reciprocal
multiplication differs. Entry checks cover all BF16 encodings in groups, all
supported dtypes, both scale layouts, aligned/offset views, generic/model widths,
3/4/127/128-row boundaries and NaN/Inf groups. FP8 bits and scales match the old
native entry exactly in those checks; they do not prove every possible tensor.

Both the original and final formatted39-case quantization/SiLU subsets pass.
Final results and separate TileFoundry/Nsight observations are recorded in
`bench/baseline/2026-09-22-cuda-quant-vector-progress.json`. Full correctness and
ordinary/MTP4 eager FP64 probes pass without tolerance changes. Forced64-token
text includes control tokens/repetition and does not establish agentic acceptance.
Full147-case and framework acceptance remain pending; default stays TileLang.

## Stable GDN beta and complete operator matrix

The clean087bb51 collection completes147 operator cases:113 pass and34 fail,
without detected GPU interference. Remaining failures are quantization12,
recurrent7, Q/K normalization6, convolution5 and gates4. This is not acceptance.

GDN beta now uses a bounded exponential and fast division with denominator in
[1,2]; decay and all existing tolerances remain unchanged. The13-case dirty-source
gates subset passes every speed decision. Full CUDA correctness passes168 tests
plus28 subtests. A temporary exhaustive check of all65536 BF16 beta encodings
passes the original1e-7 absolute/1e-6 relative tolerance against FP64 sigmoid,
including signed zero, infinities and NaNs (maximum finite absolute error
9.1063e-8). This does not claim bitwise equivalence.

Independent review passes. Separate TileFoundry estimates and paired Nsight
observations are retained with the formal summaries in
`bench/baseline/2026-09-22-cuda-gates-progress.json`. Subset wins are not added to
the previous full-matrix count. Full framework performance, context and service
acceptance remain outstanding; the default backend remains TileLang.

## Full-attention preparation fusion

Target and MTP now share a production preparation entry that fuses Q/K RMS/RoPE,
V layout conversion and KV append on CUDA. It returns Q and writes the same K/V
cache rows; packed gate values remain untouched. Both BF16 rounding points and
FP64 phase reduction remain. Negative slots skip only KV writes. The TileLang
backend executes the original complete frozen chain, preserving V.contiguous()
as a no-op for already-contiguous views rather than forcing a copy.

Static cases are re-derived from the changed production call graph:13 fused
configurations replace26 independent RMS/RoPE and13 append invocations, giving
147 cases. Target Batch and MTP/DraftGraph positions/slots are int64; attention's
int32 tables/lengths are unrelated. Existing standalone APIs remain covered by
correctness tests and are explicitly unused in the performance workloads. The
old173-case111-pass/62-fail collection is retained, not reclassified as passing.

The dirty-source fused subset passes13/13 speed decisions. Full CUDA tests pass
168 plus28 subtests; eight new checks also pass with frozen TileLang. They cover
single/five tokens, all positions/slots dtype combinations, near262144 positions,
page boundaries/scattered slots, exact V/untouched cache, both BF16 roundings,
nondefault stream and graph replay after changing inputs and write destinations.
Actual-model ordinary/MTP4 eager FP64 probes pass. Their forced-length text does
not establish semantic/agentic acceptance. TileFoundry's representative logical
chain check and separate Nsight collection pass; HIR cache concatenation and
FP32 phase are estimates, not native traffic or precise phase validation.

See `bench/baseline/2026-09-22-cuda-attention-prepare-progress.json`. Independent
review and all-file checks pass, and all owned GPU programs exited. Full147-case
and framework acceptance remain outstanding; default stays TileLang.

## Medium residual RMS cache policy

Only2048–4095-row residual RMS now uses a streaming store cache hint for its two
outputs. This preserves the stored BF16 bits and stream visibility while reducing
input eviction in the operator workload. The26-case dirty-source RMS diagnostic
passes all decisions; the previously failing2496-row case saves about1.30us in
paired mean latency. This is not complete operator/framework acceptance, and
possible downstream cache misses still require the final end-to-end gates.

Full CUDA correctness passes160 tests plus28 subtests, and FP64 boundary checks
at2047/2048/4095/4096 rows pass with exact residual sums. A compiler incompatibility
in the first packing intrinsic was fixed before these successful reruns; its test
process was stopped. Independent review, separate TileFoundry/Nsight observations
and all-file checks pass. Evidence is in
`bench/baseline/2026-09-22-cuda-rms-stream-progress.json`.
All owned GPU programs exited. Default remains TileLang; migration continues.

## Gated RMS specialization and complete-matrix update

Clean d45c4f2 completes all173 cases:111 pass,62 fail, with no detected GPU
interference. All16 attention cases pass. The2496-row residual RMS case fails
this complete collection despite earlier subset wins; its advantage is not yet
reliable. Remaining failures include append, small quantization/rotary, Q/K
normalization, recurrent updates, gates and convolution. Full acceptance is pending.

A new48-head/128-wide gated RMS path retains values in registers and uses a stable
sigmoid with denominator in[1,2]. It keeps FP32 gating without intermediate BF16
rounding and supports the original strides/epsilon. The dirty-source gated RMS
subset passes13/13 cases. Full CUDA correctness passes160 tests plus28 subtests.
Independent review, extra FP64 packed/nonunit-stride and large-weight/negative-gate
checks pass. All finite BF16 gate encodings were also checked with fixed input1
and weight0.25; this is not exhaustive over input/weight combinations or bitwise
proof. No original tolerances change. Separate TileFoundry/Nsight observations,
complete-matrix decisions and subset evidence are summarized in
`bench/baseline/2026-09-22-cuda-gated-rms-progress.json`.

All-file checks pass and all task-owned GPU programs exited. Default stays
TileLang; further kernel tuning and complete framework acceptance remain.

## RMS launch and vector-width tuning

Model-width RMS now selects 128 threads for medium batches, 160 for large plain
RMS and 320 for large residual RMS. Large residuals use eight-element vectors
only when input/residual are16-byte and weights32-byte aligned; otherwise the
four-element or generic path remains. Compile-time divisibility checks ensure
complete row coverage. Reduction order changes; existing tolerances are retained.

The final unchanged-source diagnostic passes26/26 RMS cases. Earlier diagnostics
pass24/26 then25/26; the latter has one unusually slow CUDA sample and fails the
confidence bound despite faster medians in all three rounds. No samples were
excluded and no interference was detected; its exact cause remains unresolved.
All attempts are summarized in `bench/baseline/2026-09-22-cuda-rms-progress.json`,
with separate TileFoundry/Nsight observations. These are dirty-source subsets,
not full operator or framework acceptance.

The full CUDA suite passes160 tests plus28 subtests. Temporary FP64 checks cover
2047/2048/4095/4096-row dispatch boundaries, nondefault epsilon, and both alignment
fallbacks at4096 rows; residual sums remain bitwise equal to the reference.
Independent review and all-file checks pass. All owned GPU programs exited.
Default remains TileLang; remaining operator tuning and full acceptance continue.

## Attention fragment reuse and merge optimization

The attention diagnostic now passes the speed decision for all 16 static cases,
with three rounds of 20 interleaved pairs and no detected GPU interference.
This dirty-source subset is not full acceptance. Q/K and probability/value MMA
fragments are reused across output tiles; static register indices eliminate
local-memory spills. The merge uses shared split weights and vector output.
Ungrouped KV double buffering is limited to four queries to preserve occupancy
for larger eager batches. The sampled final eager profile reports zero local
loads/stores; TileFoundry estimates remain separate from measured counters.

The unchanged full CUDA suite passes 160 tests plus 28 subtests. Actual-model
ordinary and MTP4 eager FP64 probes pass, including recurrent states and draft
attention. Forced-length probe text is not semantic or agentic acceptance.
Merge reduction order changes, so numerical tolerance results do not establish
bitwise equivalence. Independent review and all-file checks pass. Summarized
source identities, timings, hardware observations and probe coverage are in
`bench/baseline/2026-09-22-cuda-attention-progress.json`.

RMS, Q/K normalization, KV append and other failed cases still need tuning.
The latest complete clean matrix remains 80/173; do not add subset pass counts.
Full operator and framework performance acceptance remain outstanding; default
backend stays TileLang. All task-owned GPU programs have exited.

## Packed SiLU follow-up

Cleancb867d4 completed all173 operator cases without detected GPU interference:
80 pass,93 fail. The next diagnostic uses width-specialized FP8 kernels and
paired BF16 SiLU multiplication. All13 fused-SiLU cases satisfy the existing
three-round speed decision, including the previously failing medium shape;
this dirty-source subset is not full acceptance. Full CUDA correctness remains
160 tests plus28 subtests. Independent review, layout/dtype boundary checks and
all-file hooks pass. Separate TileFoundry/Nsight summaries and source identities
are recorded in `bench/baseline/2026-09-22-cuda-silu-progress.json`.

The fast exponential and paired multiply preserve both BF16 rounding boundaries.
Temporary SM100/CUDA13.1 exhaustive checks found zero differences for rounded
SiLU over every finite BF16 input and for BF16-pair multiplication versus FP32
multiplication followed by BF16 rounding over all finite BF16 operand pairs.
These checks include signed zero, subnormals, overflow and underflow; they do not
establish equivalence for other architectures/toolchains or arbitrary FP32 SiLU.
No temporary tuning scripts or raw traces are retained in the repository.
Attention, Q/K normalization, KV append and other failed cases remain to be tuned;
no final CUDA model/performance/service acceptance is claimed. Default remains
TileLang. All task-owned GPU programs have exited.


## Vector-kernel optimization progress

Clean7437e0f completed the full173-case operator matrix without detected GPU
interference:48 pass and125 fail. The subsequent vector-kernel diagnostic covers
91 normalization/quantization cases:55 satisfy the speed decision, versus23 of
those same cases before this change. Dirty source and partial coverage make the
new collection diagnostic only. Per-source results and paired hardware summaries
are in `bench/baseline/2026-09-22-cuda-vector-progress.json`; no complete CUDA
operator or framework acceptance is claimed.

Native quantization now tiles rows and scaling groups, uses unsigned warp REDUX
on nonnegative FP32 magnitudes, and packs fused SiLU loads/FP8 stores. Small fused
rows and widths exceeding CUDA grid.y capacity use a flat grid. Exact division
and both BF16 rounding boundaries remain. Model-width5120 RMS retains values in
registers and uses aligned vector loads with guarded scalar/general-layout paths;
residual addition rounds BF16 pairs before normalization. Fixed fused-RoPE
frequencies use FP64 read-only values; phase reduction remains FP64. Fresh-output
kernels declare nonaliasing pointers; in-place state/cache kernels do not.

Full CUDA correctness passes160 tests plus28 subtests. Supplementary boundary
checks cover nondefault RMS epsilon, unaligned input/weight storage, FP16/FP32
quantization, dispatch boundaries and extremely wide quantization. Existing
long-position FP64/stream/graph checks and all-file hooks pass. Independent
review found and fixed the quantization grid.y width limit. Large RMS, small
quantization, fused SiLU and attention still need tuning; the default stays
TileLang. Test/profiler workers exited and no task-owned GPU process remains.


## Active: CUDA/PTX custom kernels

User confirmed REQ-KERNEL-002; CUDA migration is in progress, not accepted.
All owned entries now have native CUDA implementations. After shared attention
merge-weight optimization, full unchanged GPU suite passes153 tests plus24
subtests (no skips). Production-entry FP64 packed Q/K, near262144 positions,
nondefault stream and changed-input graph replay pass. An earlier missing
convolution factory binding was fixed before these successful full reruns.

Formal static matrix contains173 deduplicated cases from12 workloads, including
three-sequence dispatch, graph metadata dtypes and eager fallback. Whole frozen
operations retain snapshot/merge costs. Subsets and dirty-source runs cannot
pass full acceptance; collector checks reference and dependency-lock hashes.
TileFoundry twins now explicitly bind frozen TileLang. Native and matrix review
findings have been addressed. The initial performance diagnostic was invalidated
by an external GPU entrant; no formal result is claimed. Attention and other
small shapes still need tuning. Remaining: per-case stable wins, complete metrics,
actual-model correctness, all12 frozen-vLLM rows and boundary/service/agentic
acceptance. Default stays TileLang until complete. All test GPU programs exited.
See cuda-development.md. The following TileLang results are the accepted reference.

User-requested cleanup removes13 duplicate TileLang factory bodies from the
seven production modules. Only frozen TileLang contains the DSL; public wrappers
are AST-identical and bind factories by explicit names. Reference hashes pass.
Native attention now uses register accumulators, inline PTX MMA/ldmatrix,
swizzled shared memory and shape/position-type specialization. Both full backend
reruns pass154 tests plus24 subtests, no skips; independent reviews pass. These
are still correctness milestones: attention timing diagnostics remain slower
than TileLang, so no performance acceptance is claimed. Owned GPU runs exited;
an unrelated GPU process remains and was not stopped.


Latest attention tuning keeps score/softmax state in registers, shares grouped
32-query/64-key tiles, and overlaps KV prefetch for larger ungrouped batches.
Review found unaligned contiguous BF16 storage-offset inputs: native Q now has
safe scalar loads; the production wrapper aligns KV for both backends and Q for
frozen TileLang. The frozen source remains unchanged. CUDA full suite passes160
tests plus28 subtests; TileLang attention passes16, including all three new
alignment cases. Hardware-observation CLI now profiles both complete operations
with TileFoundry estimates separate, validates counter/source identity and reaps
owned process groups on cancellation. Real paired Nsight collection and CPU
validation pass. Performance diagnostics still fail TileLang-relative gates;
formal per-case and full-framework CUDA acceptance remain outstanding.



## Current: TileLang acceptance and revised stability

All seven project-owned custom-kernel modules now use TileLang. Clean measured
implementation da75c02 passes all twelve frozen EngineCore throughput/TTFT rows
under the user-approved10% spread/median limit (revised from5% on2026-09-22).
The same complete raw measurement sets were re-evaluated; original decisions
are preserved. Prefix batch1 has5.97% TTFT spread and failed only the old gate.
Minimum throughput ratio is95.3098%; maximum TTFT ratio is101.4735%.
The comparator emits its stability limit and CPU tests cover both engines and
metrics below, at and above10%. No GPU inference source changed for this revision.

The unchanged full GPU suite passes146 tests +12 subtests, with no skips. Actual
ordinary/MTP FP64 probes, six262144-total-token capacity rows (no OOM/preemption),
text/preemption/prefix, twelve MTP4 constraint cases, lifecycle and long-service
checks pass. Both real oh-my-pi APIs completed nine successful tool calls each,
with positive MTP activity. Their answers predate this final acceptance update;
raw answers and grounding caveats are retained. Final document readback on83d77ff
also passes: Chat7 successful tools in22.51s, Responses13 in34.78s; all7 model
requests have positive proposed/accepted drafts. Final request throughput is
252.31/234.24tok/s and TTFT279/351ms; the first cold request includes compilation.
All owned GPU programs exited; ports18025/18026/18027 are released. Independent
review caveats are recorded in acceptance.md; no pending performance blocker.
See acceptance.md and bench/baseline/2026-09-22-tilelang-*.json for full evidence.

TileFoundry is development-only, pinned to the personal fork6b1b149. Offline
TileLang AutoTuner and cost/memory analysis inform production dispatch. The HIR
cannot represent FP64 phase reduction; independent FP64/model tests remain the
correctness authority. All25 owned Markdown documents have Chinese companions.
Earlier sections below are historical progress, not current blockers.


## Bilingual documentation (user-requested)

All 25 project-owned Markdown sources now have same-directory `.zh.md`
companions. Agents continue to read the English originals as authoritative;
every future Markdown change must update its Chinese companion in the same
change. AGENTS.md and CONTRIBUTING.md record this rule. Third-party submodules
and generated dependencies are excluded. Independent translation review, complete companion coverage, local links and
code-block checks pass. This documentation work does not replace kernel acceptance.

Latest kernel status: clean 874a54a reaches94.9688% on formal MTP batch4, still
below95%. The Q/K RMS+RoPE fusion preserves intermediate BF16 rounding.
Static review passes. Its full GPU suite rerun passes146 tests +12 subtests;
the first run failed the process-timeout test because another process entered
the selected GPU, not because of numerical failure. Actual-model ordinary/MTP probes pass, but a production-entry FP64 check found
RoPE phase error near position262144. The production fix computes/reduces phase in FP64 and passes packed Q/K,
int32/int64 positions and graph replay at the original tolerances. After this
fix, the complete146+12 suite and actual ordinary/MTP FP64 coverage pass.
TileFoundry analysis and the representative twin check pass; its lack of FP64
phase representation is explicitly documented and cannot validate this boundary. Final twelve-row and feature acceptance
remain required before completion.

## Active task: TileLang migration and TileFoundry development workflow

User confirmed implementation. Replace all 15 project-owned Triton kernels in
seven modules, preserving public behavior and existing correctness tests/tolerances.
Third-party internal Triton is outside scope. TileFoundry is development-only,
forked from upstream main into para1lel/TileFoundry on oh-my-vllm-integration,
and pinned as 3rdparty/TileFoundry. Main project stays on main; no upstream PR.
Remove the fork's Transformers upper bound and source-install in oh-my-vllm.
User permits compatible TVM FFI/protobuf downgrades, but no TileLang/OR-Tools fork.
Simple TileFoundry fixes are authorized; substantial fixes require discussion.
Temporary operator experiments and records stay outside the repository. Final
tests/evidence are limited to user-required and already-documented acceptance.
Keep vLLM ratio gates; do not introduce a Triton-relative gate. Profile, optimize
and resolve performance failures. The user strengthened the goal: all twelve rows
must pass throughput, TTFT and stability gates; measured explanations alone are
not completion. Tune across required shapes, informed by TileFoundry analysis.
Current stage: all seven production kernel modules use TileLang, with no owned
Triton imports or fallback. Token dimensions are dynamic where they do not choose
an algorithm/layout; existing CUDA graph shape keys remain unchanged.
TileFoundry fork at 6b1b149 has the dependency and HIR sin/cos adaptations, reviewed
and pushed. Development-only complete-operator HIR/twins and installation/workflow
docs are in development/kernels and docs/tilelang-development.md.

Verified: full existing GPU pytest 146 tests +12 subtests, no skips, after dynamic
length fixes; Rust workspace tests, fmt, hard line-width and clippy pass. Dependency
closure check and development install dry-run pass. Actual ordinary FP64 probe
passes. MTP FP64, twelve constraints, lifecycle and long-context prefix service
passed before the dynamic-length change; final model/agentic/boundary checks and
formal twelve-row comparison are in progress. No final performance claim yet.
A diagnostic run overlapped a source edit and failed deferred JIT inspection;
discarded and rerunning against fixed sources. A temporary MTP probe driver set
the coverage flag to 4 instead of 1, so its shutdown coverage check failed;
the corrected driver is rerunning. No numerical tolerance or test was changed.
Operator experiments and their records remain outside the repository.

The migration optimization keeps dynamic token/batch dimensions and computes
attention split extents on device. KV gather uses explicit coalesced placement;
one- and two-sequence GDN uses warp-local reductions. Review found no correctness
blockers. Optimized full suite again passes146+12 with no skips, and both actual
ordinary/MTP FP64 probes pass. Six boundary rows passed before this optimization;
their optimized reruns plus service/agentic subsequently passed on 6a6e6a5.
Its first formal matrix passed seven of twelve rows: MTP throughput failed at
batch 1/2/4, and prefix TTFT stability failed at batch 1/2. These are incomplete
results, not final acceptance. The subsequent two-sequence GDN layout passes the
full 146-test suite plus 12 subtests; final model and performance reruns remain
pending while attention shape tuning continues.
The next attention milestone uses predicated asynchronous global-to-shared
copies, 128 threads and 64 splits. Logical positions use int32 only with enough
ceildiv headroom; physical cache offsets remain int64. Review verified shared
memory barriers and tail zero-fill. All 146 GPU tests plus 12 subtests pass on
the rerun; an earlier run had a GPU-contention failure in the process-timeout
test, not a numerical failure. All-file formatting, Rust checks and hooks pass.
TileLang AutoTuner is now exercised offline with explicit legal metadata, CUDA
Graph timing and unchanged FP64 checks. Workflow documentation requires complete
operator and model revalidation, with no runtime search or repository microtests.
Actual-model ordinary/MTP FP64 probes pass with complete required coverage.
The full performance matrix remains in progress.
Formal MTP on f52a2d4 passes batch1/2 (97.386%/95.351% of baseline), but batch4
is still92.248%; TTFT and stability pass. Further offline tuning adjusts tile1
FP8 quantization threads, removes residual-width padding at5120, reduces GDN
threads for larger batches and reduces verification splits for short extents.
The full suite again passes146+12; actual batch4 GDN FP64 output/state checks
pass. Final model/service/boundary and twelve-row acceptance remain required.
The b868c34 formal MTP batch4 row reaches94.6674%, with TTFT/stability passing;
it remains below95%. The following milestone streams attention merge accumulation
to avoid a large weighted fragment, enabling128 splits/BK32 for longer ungrouped
decode. Short-row RMS uses one warp. All146+12 GPU tests and all-file hooks pass;
temporary long-extent FP64 checks cover absent position0, unequal lengths and
graph replay with changed lengths/page tables. Final acceptance remains pending.
Historical performance figures below describe the pre-TileLang implementation,
not acceptance for this migration.

## Previous investigation: TTFT and long context

User approved a new 12-row throughput/TTFT baseline on latest official vLLM main,
plus 262144-total-token ordinary/MTP4 boundary checks at batch 1/2/4. See current
requirements. New acceptance is pending; the results below describe the prior task.
Frozen upstream SHA: e9f169d16b9408bb9ae44f75072b91a5521d733c.
The prior local vLLM telemetry commit was preserved in
/data0/shared/dongwu.chen/vllm-before-ttft-20260922.bundle before updating main.
Baseline environment adaptation is complete. The exact cu130 wheel metadata
was initially unavailable, so a source build began. Official wheel publication
completed at 02:13 UTC; rechecking metadata succeeded and the owned source build
was stopped before compilation. Installed the official frozen-commit wheel via upstream editable mode: version
0.29.1rc1.dev505+ge9f169d16.precompiled, Torch2.13.0+cu130. pip check and
vllm._custom_ops imports pass. See /tmp/oh-my-vllm-baseline-install-official.log
and /tmp/oh-my-vllm-ttft-final/baseline-environment.json for extension hashes.

Implemented, not yet accepted: per-request TTFT, optional independent GDN pool,
formal measurement/provenance checks, and corrected Qwen3.8 naming. 77 Rust tests
pass. Final complete GPU pytest passes 100 tests and 12 subtests without skips,
after rebuilding the debug binary for the new model ID. Formatting, clippy and
all pre-commit checks pass.
131072-input batch4 short-output diagnostic completed without OOM/preemption.
The first full 262144 boundary attempt exposed signed-int32 decode page-offset
overflow; widened before multiplication and added FP64 high-page tests (9 pass).
All six ordinary/MTP4 batch 1/2/4 boundary checks finished: input 258048 +
output 4096, zero preemptions. Peak reserved memory is 134951731200 bytes ordinary
and 140033130496 bytes MTP4. These cold diagnostic runs are not formal performance
measurements. Actual-model ordinary/MTP FP64 probes retain original tolerances.
Twelve MTP constraint cases, service lifecycle and both APIs with 131099 input
tokens pass; repeat requests reuse 130928 cached tokens. All owned boundary/probe/
service workers exited and port 18015 is released. See
bench/baseline/2026-09-22-ttft-stage1.json for raw boundary results and log hashes.
Both oh-my-pi APIs reran with Qwen3.8 model ID and MTP4: Chat/Responses made
10/8 successful reads, including both required source files, with zero tool errors.
Elapsed times were 28.02/22.16 seconds. Their GB/GiB unit error and other answer
caveats are retained in bench/baseline/2026-09-22-ttft-agentic.json. Server/worker
exited and port 18016 is released; unrelated GPU processes remain untouched.
Prefill profiling found inefficient per-row FP8 quantization. A 16-row tile
reduces observed quantization GPU time from 295 to 51 ms in a diagnostic model
run. Arithmetic is unchanged; small/decode shapes retain the old kernel. Complete
suite before final threshold tuning: 103 tests + 12 subtests; final boundary/FP8
tests: 14 pass. See 2026-09-22-prefill-quantization.json for limitations and hashes.
Post-optimization ordinary/MTP actual-model FP64 checks also pass. Four initial
baseline attempts were aborted because an external TP4 job entered their GPUs;
see 2026-09-22-ttft-discarded.json. Those results are excluded. The collector now
streams to disk so interference/timeout retains child output. Eleven baseline rows have passed collection audits; remaining
rows are collecting/retrying in /tmp/oh-my-vllm-ttft-final. Prefix4 attempt103
was rejected for TTFT spread 13.94% despite stable throughput. User confirmed to
keep waiting/retrying when external GPU interference occurs. No complete 12-row
comparison exists yet. Candidate MTP attempt0 was canceled during first warmup
because copying FlashInfer caches caused expensive recompilation; no measured
result from that attempt is used. Future candidates use the existing independent
FlashInfer cache and auditable per-run Triton cache.

Long-prefill attention now compacts active KV for independent ragged TRT-LLM
attention, retaining persistent 784-token pages and small-query FA2. Full suite
110 tests +12 subtests and both actual-model FP64 paths pass. Ordinary/MTP4 bs4
maximum-context reruns pass without preemption, with unchanged peak reserved
memory. See 2026-09-22-ragged-prefill.json. Final numerical checks, including
mixed long-decode isolation, pass (3 tests). Finish baseline
collection, performance fixes and reviews before claiming completion.

Prefill convolution/SiLU tiling and dual-SM large-projection GEMM reduce a
diagnostic GPU profile from 1.816 to 1.721 seconds. Full GPU suite passes
113 tests +12 subtests; actual-model MTP target/draft FP64 checks pass.
See 2026-09-22-prefill-tiles.json. First candidate rows exposed shared Triton
cache audit contamination, so subsequent runs must use separate Triton roots.
The 131072-input batch1 candidate on 0ad2bb4 passed both gates; other first
rows are diagnostic only. MTP throughput remains below threshold. Isolated
baseline step-count diagnostic found 898 steps versus our 899, pointing to
per-step cost rather than acceptance rate. Alternate private GEMM tactics
triggered illegal access in a temporary prototype and are not used in runtime.

Packed GDN/convolution/RMS inputs now retain their token/head strides; target
and draft metadata transfers are grouped. Review requested a single int32 starts
conversion per batch to avoid repeated FlashInfer casts; implemented. GPU suite
120 tests +12 subtests, actual MTP FP64, all 12 service constraints, lifecycle
and 131099-input prefix checks pass. Service18017 exited/released. Diagnostics
improved MTP bs1/2 to305.86/450.65 tok/s, still below their new baseline gates.
See 2026-09-22-strided-metadata.json. Prefix2 baseline continues to retry after
variance failures and repeated unrelated GPU entrants; user authorized retries.

Residual RMS/SiLU quantization fusion, independent CuTe-DSL vocabulary projection
and native target decode via zero-copy 16-token subpage views are implemented.
The 784-token persistent layout is unchanged; draft position-zero exclusion
retains its custom kernel. Final GPU suite:138 tests +12 subtests. Actual
batch2 target/draft graph-warmup FP64 probes pass (separate from the eager GDN
probe, which also passes). Twelve service constraints, lifecycle and long-prefix
checks pass; service18018 and its worker exited. Initial MTP diagnostics at
batch1/2/4 are319.66/473.69/672.44 tok/s; these are NOT formal acceptance, and
batch4 remains below95%. Larger GDN tiles for four sequences and eight-warp
residual RMS pass final tests and review; short MTP4 diagnostic674.03 tok/s
remains below target. All six maximum-context reruns pass with zero preemptions.
The launch tuning follows boundary worker start; see explicit evidence limits in
bench/baseline/2026-09-22-native-decode-fusions.json. All owned diagnostic,
probe, boundary and service workers exited; port18018 is released. Formatting,
Rust tests/clippy and all hooks pass. Formal collection remains in progress. Review found no blocking static correctness issue and
explicitly required distinguishing graph-warmup FP64 from eager coverage.

Prefix2 baseline retries110+ temporarily use CPU96 alone to investigate TTFT
variance. Attempt112 still failed spread37.94%; preserve all attempts. Accepted
candidates must match the affinity of whichever baseline is finally accepted.
User again confirmed no exclusive GPU can be reserved: continue waiting/retrying.

All 12 refreshed baseline rows are now collected and frozen in
bench/baseline/2026-09-22-refreshed-enginecore.json. Prefix2 attempt114 passed
with CPU96 alone; its candidates must use that affinity. Duplicate attempt201
was canceled after acceptance and its worker exited. On 5166885, six ordinary
and MTP batch1/2 rows pass both formal gates. Three prefix rows fail TTFT;
MTP batch4 attempts were interrupted by external GPU arrivals. No complete
final acceptance is claimed. Retained attempts remain in /tmp/oh-my-vllm-ttft-final.

The requested implementation/performance comparison is documented in
docs/performance-gap-2026-09-22.md with frozen upstream source links and profile
hashes. Draft attention is the clearest isolated unfavorable kernel difference;
FP8 GEMM, vocabulary projection and target attention are competitive. CPU
sampling annotations include GPU waits and must not be labeled ZMQ overhead.
Fused GDN prefill Q/K normalization, native short-prefix target context attention
and independent TRT ragged MTP prefill are implemented and reviewed. Latest
MTP batch4 diagnostic is 692.012 tok/s versus the 693.140 gate, still short and
not a formal result. GPU suite passes 146 tests +12 subtests, no skips. Actual
model FP64, all 12 constraints, lifecycle and long-prefix service checks pass.
All six maximum-context boundary rows pass without preemption. Ordinary boundary
checks precede the final MTP-only change; MTP boundaries were rerun afterward.
See 2026-09-22-prefill-research-verification.json. All owned GPU workers exited,
nvidia-smi is empty and port18020 is released. Milestone ee2fb63 is pushed.

Formal follow-up on clean ee2fb63 completed all12 rows:10 pass. MTP batch4
median692.1899 tok/s is94.8698% of baseline versus95% required; stable and TTFT
passes. It needs0.1373% additional throughput (32.45ms per complete batch).
Prefix batch1 passes both ratio gates but fails TTFT stability twice; latest
TTFT44.72–51.16ms (14.20% spread). Prefix batch2/4 and all ordinary/MTP batch1/2
rows pass.
See 2026-09-22-performance-gap-candidates.json for complete data and excluded
external-GPU-interference attempts. All owned GPU workers exited; nvidia-smi is
empty. Investigation is complete, but overall performance acceptance is not.
Next implementation priorities are masked draft attention, then GPU-resident
accepted-token bookkeeping, and a separate investigation of short-prefix timing
variance. Preserve cache position-zero exclusion and all correctness tolerances.

## Previous task: independent runtime

The independent runtime is implemented and all nine performance rows pass the
original frozen EngineCore baseline at 95% or better. Both final oh-my-pi
workflows completed with MTP4 against the reconciled documentation.

Rust owns HTTP serving, scheduling and logical KV. Python loads and runs Qwen
using project-owned code, PyTorch/Triton/FlashInfer and XGrammar. There is no
installed/imported/linked vLLM, old source/environment dependency or legacy runner.
Use scripts/with-env.sh and scripts/with-gpu.sh. Target model, single B200,
block 784 and the conda oh-my-vllm environment remain unchanged. Dependencies are
pinned in requirements/runtime.txt; extension scope is documented in ADR-006.

## Previous task verification

- GPU pytest 92 tests and 12 subtests pass, with no skips; Rust 75 tests pass.
- fmt, hard Rust line width, clippy, ruff and all pre-commit checks pass.
- Actual-model ordinary/MTP FP64 checks retain the original tolerances; grouped
  target/draft attention and graph buffer lifetime/cache restoration also pass.
- Distinct-city text tests pass through prefix reuse, staggered arrivals and
  recompute preemption. Twelve real MTP constraint cases and service lifecycle
  (mixed batch, continuation/delete, disconnect and release) pass.
- Six ordinary/prefix rows use clean ca5a200. Three MTP rows use clean 5c0848e;
  only MTP execution changed between them. Existing graph definitions are
  AST-identical and other non-MTP source files are unchanged. Per-row identities,
  raw repetitions and hashes are in the final independent acceptance artifact.
- Source/runtime/file-access audits found no old vLLM dependency. Kernel caches
  belong to this project. No original EngineCore baseline was rerun or replaced.

See acceptance.md and bench/baseline/2026-09-21-independent-acceptance.json for
numbers and evidence. Intermediate artifacts retain failed performance rows,
external-GPU-contaminated attempts, two CPU-overlap exclusions and an isolated
prototype buffer-lifetime failure fixed before production. Diagnostics are never
substituted for formal repetitions.

## Previous task agentic verification and cleanup

Chat / Responses completed 7 / 18 successful read calls, including both required
source files, with zero tool errors and real follow-up requests. End-to-end times
were 26.27 / 36.68 seconds. Both answers identify the project and completion state;
minor Responses wording/count inaccuracies are explicitly retained in acceptance.md
and the JSON evidence. Do not claim perfect model-answer grounding.

No work remained in the previous independent-runtime scope. All owned
GPU programs exited; nvidia-smi has no compute processes and ports 18013/18014 are
released. No service was requested to remain running.

Develop and commit only on main, stage intentional files, keep hooks enabled and
retain the required attribution. Never leave task-owned GPU programs running.
