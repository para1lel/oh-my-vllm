# HTTP performance measurement

Use four B200 GPUs and the full Rust service.
Set the selected GPUs apart from GPU 0.
Keep raw artifacts in a separate location.
Set `TMPDIR` to a directory with write permission.
The directory must have a short path for Unix sockets.

The approved matrix has 38 cases.

| Mode | Strategy | Batch | Input tokens | Output tokens | Cases |
|---|---|---:|---:|---:|---:|
| DSpark5 | TP4, TEP4, DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 36 |
| Ordinary | TP4 | 4 | 400000 | 64, 8192 | 2 |

Input and output lengths apply to each request.
Use uncached prefixes and exact tokenizer lengths.
Use greedy text execution until the model sampling contract is extended.
The runner sets `ignore_eos` and checks the full usage record.

Run the matrix with:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_benchmark \
  --model "$MODEL_ROOT" \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

Use `--case` with the identifier from `development/serving/deepseek_matrix.py` for a subset.
Set `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` through the runner's `--cache-context` option.
Set `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS=32768` for long-context throughput.
The runner records source, executable, native modules, checkpoint, tokenizer, topology, and configuration identities.
It also records the harness Python, PyTorch, CUDA, NCCL, and numerical settings.
These settings do not independently identify each worker's runtime.

The full timer includes request submission, prompt preparation, queueing, model execution, and response parsing.
It excludes model loading and initialization.
The CSV records mode, strategy, batch, exact lengths, request count, full seconds, and output rate.

No complete approved matrix result is recorded yet.
The HTTP result must use a DeepSeek-specific reference calculation before efficiency can be reported.
The session-only SOTA target does not become a repository performance gate.
