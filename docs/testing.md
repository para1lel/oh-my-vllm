# Testing Guide — oh-my-vllm

## Environments and CPU checks

All commands use scripts/with-env.sh to set PYTHONPATH and CARGO_TARGET_DIR.
The framework uses conda oh-my-vllm; GPUWorker and baseline use conda vllm.

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python -m unittest discover -s tests -p test_runtime_tools.py
scripts/with-env.sh python -m unittest discover -s tests -p test_benchmark_tools.py
scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python -m unittest discover -s tests -p test_scheduler_adapter.py
scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python -m unittest discover -s tests -p test_bridge_logging.py
```

Rust tests cover pool/free/hash/prefix accounting, chunked prefill, arrivals,
recompute preemption, MTP acceptance/rejection, private prefix-hit states,
speculative slot migration over multiple blocks, and complete release.
CPU benchmark tests check invalid MTP configuration and descendant cleanup.

## GPU selection and real text

Every single-GPU command waits for any idle B200 using scripts/with-gpu.sh.
Use a unique socket per run. Do not terminate unrelated GPU processes.

```bash
scripts/with-env.sh cargo build --release -p oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

Add --num-speculative-tokens 4 for MTP; add --context-repeats 100 to cross a
784-token boundary. Both ordinary and MTP paths have produced coherent Chinese.
Fixed output limits deliberately ignore EOS, as does the throughput baseline.

## Actual-path FP64 reference probe

Add these environment variables to the real-text command (after with-env.sh):

```bash
env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON=/data0/shared/dongwu.chen/oh-my-vllm/tests/probe_worker.py
```

For MTP also set OH_MY_VLLM_PROBE_MTP=1 and pass --num-speculative-tokens 4.
The probe instruments a selected real FlashInfer GQA layer and target GDN
prefill/packed decode/fused MTP calls. It never substitutes production kernels.
It checks outputs and recurrent states, including each speculative state.
Required coverage categories must appear before shutdown can succeed. This is
single-request, eager diagnostic coverage; never enable it in throughput runs.

References use actual rounded inputs in CPU FP64. BF16 output checks use
atol=rtol=0.03. Recurrent state checks require normalized RMS error <=1% and
maximum absolute error <=2% of the reference peak; both metrics are logged.
Near-zero elementwise state relative errors are unstable. These checks diagnose
individual kernel/state paths, not full-model FP8 token equality. The historical
standalone SDPA test is supplementary and does not establish actual-path coverage.

## Feature combinations and preemption

The Rust bench command accepts --arrival-interval (steps between arrivals),
--prefix-hit (seed before timed batch), --warmup and --repetitions. Global
--scheduler-blocks can shrink the Rust pool for deterministic memory pressure
without changing physical GPU allocation. Verify preemptions >0 in its result;
a constrained pool alone does not prove that preemption happened.

Examples with 256 physical blocks and max-model-len 8192:

- Ordinary: scheduler-blocks 10, batch-size 2, input-len 2048, output-len 1024.
- MTP/prefix/arrivals: num-speculative-tokens 4, batch-size 2, input-len 2048,
  output-len 128, prefix-hit, arrival-interval 3.

Results report exact output counts, cache hits, preemptions, proposed and accepted
drafts. Feature smoke runs may use warmup 0/repetitions 1; they are not performance
acceptance measurements.

## Matched performance acceptance

```bash
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

Repeat with --mode mtp and --mode prefix. Defaults are input32768/output4096,
1024 physical blocks, one warmup and three measurements. Each engine runs in its
own process on the same selected GPU with matching input IDs, sampling, cache
state and model configuration. Prefix seeding occurs outside the timer; both
engines must report exactly batch_size*32144 cached tokens for the default input.
MTP uses four drafts and BF16 SSM in both engines (ADR-003), and both must report
actual drafts. The timer includes request submission through completion.

Results preserve starting HEAD, dirty status/diff hash, binary hash, vLLM version,
configuration and raw repetitions. A dirty source is never labeled as clean HEAD.
Timeout cleanup terminates the entire owned engine process group. Median framework
throughput must reach >=95% in every mode/batch, with additional repetitions when
variance is material. See handoff.md for current results; no overall pass yet.
