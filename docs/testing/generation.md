# Generation and numerical checks

## Shared service sampling acceptance

Repetition tests compare bounded cycle detection with an independent definition and check speculative prefix distributions.
CPU cases cover joint cycles, forced grammar continuations, prompt isolation, explicit disable, and invalid logits.
GPU cases check filtering order, forced grammar continuations, and speculative prefix distributions.
HTTP regressions check shared defaults, null values, and explicit booleans through Rust, msgpack, and Python.

Tokenizer regressions check canonical reasoning closure, tool opener constraints, schemas, function choice, parallel calls, and speculative rollback.
They keep literal markup, plain whitespace, disabled tools, explicit non-strict arguments, stop strings, and total length.
These CPU cases do not prove that model repetition is resolved or that long-duration agent acceptance passed.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/integration/test_serving_worker.py tests/unit/test_thinking_budget.py -q
```

Test HTTP services use production defaults.
The two APIs and each draft mode use the same validation and precedence.
The default presence penalty is `0.5`. The separate reasoning budget default is `-1`.

Persistent acceptance must use these defaults. Record service defaults and request sampling fields after service defaults are added from the original diagnostics and metrics.
Maximum effort and output/context limits stay explicit workload settings.

Offline phase/operator measurements and budget boundary tests keep their specified explicit parameters.

`tests/integration/test_serving_http.py` checks Rust HTTP, ZMQ, Python sampler, and checkpoint merge without GPU inference.
It checks explicit zero, null omission, service changes, and rejection before worker launch.
Rust tests use the same validation function for CLI and request fields.

## Native tool description compatibility

Use the original pinned OMP Bash tool definition through the two APIs.
Its `name` parameter has `maxLength=48`. Keep this description in the model prompt.
The request adapter must accept the description without adding a string-length mask.
OMP must reject a generated name with `49` characters, return that error in subsequent model input, and complete the next valid command.
Explicit `strict=true` must keep unsupported-schema rejection.

The CPU fixture uses deterministic API responses and a sandboxed OMP executable process.
It does not prove target-model output or one-hour stability.
With `OH_MY_VLLM_MODEL` and the release normalizer available, it also uses the production CPU preflight.

```bash
scripts/with-env.sh python -m pytest tests/integration/test_serving_worker.py tests/integration/test_agentic_goal_integration.py -q
```

## Reasoning budget acceptance

Build the release binary and start each service with `serve --thinking-token-budget 32`.
Use modes `none`, `mtp`, and `dspark`, with an idle UUID-pinned B200 and the corresponding startup mode.
The script checks zero, small, default, unlimited, and off budgets through the two APIs.
It checks streams, JSON, strict tools, total length, effective request metrics, and unchanged source identities.
Stop the owned service and worker after each run. Keep original request and response records.

```bash
scripts/with-env.sh python scripts/thinking-budget-acceptance.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/thinking-budget.json"
```

## Accuracy

Use independent FP64 equations on stored inference inputs.
Production and previous implementations supply no numerical reference values.
Full-model numerical experiments use short sequences and small batches.
Use individual operators for large numerical comparisons.

The independent [model reference](../../development/qwen_fp64_reference.py) keeps all arithmetic, intermediate activations, caches, and logits in FP64.
The independent [operator references](../../development/kernels/oracles.py) use explicit casts for prescribed storage rounding between FP64 calculations.
Normalize logits and calculate numerical-error and distribution metrics in FP64.

Tests must include prefill, decode, boundary writes, strided views, grouped MTP verification, and speculative rollback.
BF16 operator output uses `atol=rtol=0.03`.
Recurrent state must have NRMSE at most 1% and maximum absolute error at most 2% of reference peak.
Examine each written slot and unchanged protected state.

For fused chains, include all rounding, scale, copy, and reduction behavior.
See `tests/probe_worker.py` and the applicable kernel tests.

`tests/numerical/test_ir_normalized_linear.py` checks small-row fusion at rows `1/2/4/6/12/24/32`.
Graph replay reads changed inputs and weight scales, keeps compact output shapes, and leaves input bytes unchanged.
Storage-offset cases check each of the three normalization inputs at rows `2/17`.
The previous exact checks and FP8 operator tolerances stay active.

## Offline execution and numerical probes

Build the release binary before these commands. Set the environment and model from [development](../development.md).

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

Use `--num-speculative-tokens 4` for MTP4.
Use `--context-repeats 100 --prefix-hit` to cross a 784-token boundary. The second execution must have a prefix hit.
Use `--binary` to select a different build.
Offline fixed output ignores EOS.

`run --prompt-file PATH` reads whitespace-delimited token IDs, one request per line, with at most 32 requests.
`--arrival-interval N` gives an interval of N scheduling steps between admissions.
`--prefix-hit` first seeds the prompts and then must have positive cache hits.
The global `--scheduler-blocks` limits the Rust pool. Its value must be at most the reported worker capacity.
The `bench` subcommand also has `--warmup` and `--repetitions`.

These two-request probes use 2048 input and 1024 output tokens per request:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

They must have positive preemption counts and the correct city at the start and end of each output.
MTP4 must accept drafts. Prefix mode must have positive initial prefix hits.
Examine the printed text as a semantic smoke test.
A small pool alone does not show preemption.

For inference-path numerical diagnosis, use the executable probe as the worker interpreter:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" python scripts/smoke-text.py --socket /tmp/probe-text.ipc --max-tokens 32
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" OH_MY_VLLM_PROBE_MTP=1 python scripts/smoke-text.py --socket /tmp/probe-mtp.ipc --max-tokens 32 --num-speculative-tokens 4
scripts/with-gpu.sh scripts/with-env.sh python scripts/probe-independent-model.py --max-tokens 32
```

The worker probe instruments model GQA, GDN prefill/recurrent verification, and MTP attention calls without replacement kernels.
It compares rounded inference inputs with independent references through CPU FP64 computation.
Eager mode disables capture so diagnostic copies observe inference requests.
The direct model probe supplies a short eager model diagnosis without Rust scheduling.
Keep these probes and eager mode disabled for performance acceptance.
