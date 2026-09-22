# Handoff — 2026-09-23

## Current state: CUDA migration complete

CUDA is the default custom-kernel backend on B200. Set
`OH_MY_VLLM_KERNEL_BACKEND=tilelang` before Python starts for the single frozen
comparison implementation. Runtime identity reports the actual process selection;
there is no silent fallback. Rust still owns serving/scheduling/logical KV and
Python owns GPU computation. No vLLM runtime/source/environment dependency returns.

Clean `c36d1c9`, explicitly selecting CUDA, passes all **147 operator cases** and
all **12 framework performance rows**. Each operator passes three rounds of20
interleaved pairs and a positive one-sided95% bootstrap gain bound. Framework
throughput is97.52–125.09% and TTFT76.01–97.50% of the frozen vLLM baseline; every
row meets the unchanged10% stability rule. Preserve earlier failed stability
sets and three physical-CPU-overlap exclusions. No baseline was rerun.
The historical173-case inventory predates attention-preparation fusion.

The12 workloads are ordinary/MTP/prefix input32768 at batch1/2/4, plus
ordinary-only input131072 at batch1/2/4, all with output4096. Six ordinary/MTP
input258048/output4096 boundary runs also complete without OOM/preemption.
Peak PyTorch reserved memory is134.69GB; this is not total-device memory.

Final default-selection correctness passes **174 tests plus31 subtests** without
relaxed numerical references/tolerances. The measured CUDA kernel/model/dispatch
implementation is unchanged by the default switch. Actual-model FP64 probes,
ordinary/MTP real-text isolation, prefix reuse and forced preemption pass.
MTP4 constraints, lifecycle and long strict-JSON prefix reuse pass. A final
service run with backend/CUDA_HOME/TVM architecture variables unset verifies the
actual default and repeats twelve constraints plus lifecycle successfully.

Updated-document oh-my-pi Chat/Responses readback passes the core task with nine
successful read calls each,5/3 real model requests and nonzero MTP acceptance.
Both read the required source files and distinguish current147 cases from history.
Retain answer limitations: Chat workload wording is imprecise; Responses mixes
Rust scheduling with Python prefill ordering and includes a historical performance
command; both repeat an earlier cleanup snapshot while their own service runs.
Do not claim perfect model grounding or use model answers as cleanup evidence.

## Evidence and provenance

See `acceptance.md` and the2026-09-22-cuda-{operators,framework,features}.json
artifacts in `bench/baseline` (collection date is UTC). Full147/12 timing is from
clean c36d1c9 with explicit CUDA; readback is from clean ef07b2b. The final change
only selects CUDA by default, unifies identity and adds selection tests. Default
suite/service evidence records its working-tree provenance and unchanged CUDA hash.
Older counts in `cuda-development.md` are historical, not current status.

All-file Rust fmt/width/clippy/tests and Ruff checks pass. Independent reviews
cover numerical code, statistics, feature evidence and final selection behavior.
No implementation or acceptance work remains in this migration scope.

## Cleanup and operating rules

The feature artifact records a timestamped cleanup snapshot after tests: no GPU
compute processes, ports18030/18031/18032 released, no requested running service.
This is a snapshot, not a live system-status promise.

Use `scripts/with-env.sh` and idle UUID selection via `scripts/with-gpu.sh`.
Develop/commit only on main, stage intentional files, keep hooks enabled, maintain
Chinese Markdown companions and the required commit attribution. Stop every
owned GPU program promptly. Temporary tuning and raw GPU traces stay outside
the repository; formal summarized acceptance evidence belongs in `bench/baseline`.
