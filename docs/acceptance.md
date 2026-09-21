# Acceptance evidence — 2026-09-19

## Matched performance

All nine required rows passed the >=95% throughput gate on single B200 GPUs.
Input/output lengths are 32768/4096 per request. Each pair uses identical input
IDs, GPU UUID, inherited single-core CPU affinity, physical KV capacity (1024),
block size (784), greedy sampling and fixed output counts. MTP uses four drafts
and BF16 SSM on both sides. Async scheduling is disabled.

| Mode | Batch | vLLM tok/s | Framework tok/s | Ratio |
|---|---:|---:|---:|---:|
| Ordinary | 1 | 92.21 | 93.74 | 101.66% |
| Ordinary | 2 | 167.87 | 170.03 | 101.29% |
| Ordinary | 4 | 250.96 | 254.64 | 101.46% |
| MTP | 1 | 269.16 | 276.27 | 102.64% |
| MTP | 2 | 418.53 | 434.33 | 103.78% |
| MTP | 4 | 446.61 | 450.04 | 100.77% |
| Prefix | 1 | 95.16 | 96.76 | 101.68% |
| Prefix | 2 | 179.44 | 182.22 | 101.55% |
| Prefix | 4 | 333.32 | 323.75 | 97.13% |

These are medians of three measurements after two warmups per engine.
[Raw data and exact commands](../bench/baseline/2026-09-19-acceptance.json)
preserve clean source commit `12d227d`, executable SHA256, GPU/CPU identity,
vLLM version, configuration and every repetition. Later cancellation and CLI
changes are not attributed to this measured executable.

The baseline uses `LLM.generate` to drive native EngineCore with detokenize=False.
Both timers include request submission through completion; Rust includes
registration, scheduling, transport and final cleanup. Loading, compilation,
warmup and prefix seeding are outside the timer. Each prefix request reports
exactly 32144 cached tokens in both engines. Every MTP measurement proposes and
accepts real drafts. Cold rows report zero prefix hits.

Cases ran concurrently on different idle GPUs, with one CPU core assigned per
pair. These results apply to that recorded configuration, not arbitrary CPU
placement. GPU ownership was monitored during each engine run; no external
same-GPU process was observed. Polling cannot exclude arbitrarily short activity
between samples. Even comparing each row's slowest framework repetition with
its fastest vLLM repetition exceeds 95% (minimum 97.10%).

MTP batch 2 had approximately 2.6% within-engine timing spread, so an additional
[pair with three warmups and five measurements](../bench/baseline/2026-09-19-mtp-bs2-repeat.json)
passed at 98.85% (vLLM 427.32 tok/s, framework 422.42 tok/s). Within-engine
spread fell to 0.67% for vLLM and 0.13% for the framework. The original three
measurements remain intact. This follow-up supports the >=95% gate, not a stable
3.78% speedup claim. It uses the same measured executable with unrelated dirty
source edits recorded; Python worker sources were frozen until it completed.

## Correctness and feature coverage

Actual-path probes instrument real GQA and GDN calls during inference and compare
their rounded inputs/outputs with independent CPU FP64 references. Coverage
includes prefill, decode, crossing 784-token boundaries and all speculative
state prefixes. BF16 outputs use atol=rtol=0.03; recurrent states require NRMSE
<=1% and maximum absolute error <=2% of reference peak. These are eager,
single-request kernel checks, not full-model FP8 token-equality claims.
See [testing](testing.md) and the historical probe paths in [handoff](handoff.md).

Ordinary, MTP and prefix-hit real Chinese text checks passed. The additional
[two-request text evidence](../bench/baseline/2026-09-19-batch-text.json) uses
distinct Beijing/Tokyo prompts, staggered arrivals, 2048 input tokens and 1024
output tokens each. Both beginnings and tails remain coherent and on topic.

- Ordinary: 2 recompute preemptions, zero initial prefix hits.
- MTP plus seeded prefix: 4 preemptions, 3136 initial prefix-hit tokens,
  1143 accepted drafts.

Initial prefix hits exclude hits on re-entry after preemption. The aggregate
hit counter also reports those recompute hits. Fixed output counts deliberately
ignore EOS, matching the performance workload. Text checks test request isolation
and recovery; they do not establish exact output equality with vLLM.

Rust tests cover cache allocation/release, prefix ownership, arrivals, preemption,
speculative state migration/rollback and cancellation. Cancellation is allowed
between completed steps and must flush finished IDs to the worker. CPU adapter
and bridge checks cover cleanup of Python and native Worker request state.

## Operational scope

Timestamped multilevel logs correlate Rust/Python runs and steps; debug host
timers help locate scheduler, IPC and worker overhead. They are not CUDA timings.
GPU test scripts wait for any idle B200, and baseline/worker use conda vllm while
framework tools use conda oh-my-vllm. VSCode's ignored local settings select the
framework environment for Python analysis.

Multi-GPU execution, CPU KV swap, multimodal inputs, LoRA and production
deployment remain outside the agreed scope. OpenAI-compatible HTTP serving was
accepted as future scope on 2026-09-21 (see serving.md), but is not implemented or
covered by this acceptance evidence. These measurements remain EngineCore-level.
