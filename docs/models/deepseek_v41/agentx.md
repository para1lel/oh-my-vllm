# AgentX measurement

Use `agentx-harness==1.0.7` and the existing `inferencex-agentx-mvp` scenario.
Use the pinned dataset, seed `42`, and at least 900 seconds of profiling.
Use four GPUs apart from GPU 0.
The runner uses the session-tree concurrency as the default service sequence capacity.
Use `--max-num-seqs` to override it.

The default dataset, `semianalysis_cc_traces_weka_062126_256k`, has a 256K context range.
The context option selects an upper limit. It does not extend the dataset requests.
A 1048576-token limit does not prove execution at that length.

Use `--dataset semianalysis_cc_traces_weka_062126` for an additional full-context run.
Set `--cache-context 1048576` for that run.
Keep its results separate from the approved `_256k` matrix.
The [full dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729) permits input lengths up to 990016 proxy tokens.
Report the observed maximum input and total lengths, not only the configured limit.

The approved cases are:

| Session-tree concurrency | Strategy |
|---:|---|
| 4 | TP4 |
| 16 | TEP4 |
| 64 | DEP4 |

The concurrency counts session trees.
Observed HTTP concurrency can be higher.
The service limit is 64 active requests.
Keep queueing, failed, cancelled, incomplete, overflow, and completed counts separate.

Run one case with:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_agentx \
  --model "$MODEL_ROOT" \
  --agentx-python "$AGENTX_ENV/bin/python" \
  --strategy tep4 \
  --concurrency 16 \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/agentx-tep4-16.csv \
  --figure docs/models/deepseek_v41/assets/agentx-tep4-16.png
```

The runner records source, executable, native modules, checkpoint, tokenizer, topology, strategy, and concurrency.
Raw harness records and service logs stay in a separate location.
The CSV and dark-background scatter plot are approved result artifacts.

## Functional record

The latest functional record uses TEP4 with 16 session trees on GPUs 1, 2, 3, and 4.
It uses `max_num_seqs=1` to fit the model and cache in the available device memory.

| Setting | Value |
|---|---:|
| AgentX version | `1.0.7` |
| Profiling duration | 900 seconds |
| Cache context | 65536 tokens |
| GPU blocks | 512 |
| Mamba blocks | 64 |
| Maximum batched tokens | 16384 |
| Warmup requests | 16 |
| Profiling requests sent | 14 |
| Completed requests | 2 |
| Failed requests | 0 |
| Cancelled requests | 12 |
| Unaccounted requests | 0 |
| Drain duration | 120 seconds |
| Drain timeout | Yes |

The [functional CSV](assets/agentx-tep4-16-functional.csv) and
[functional dark scatter plot](assets/agentx-tep4-16-functional.png) record this run.
The two completed rows show end-to-end service execution and artifact export.
They do not represent TEP4 throughput at concurrency 16 because the service used one resident sequence.
The harness phase counters include 12 requests cancelled after the drain timeout.
The raw request export does not include these cancelled requests.
Thus, counts from raw request records alone are not complete.

The current DEP4 command uses the TEP4 executor fallback.
Do not treat a DEP4 result as DEP4 performance evidence until request ownership and expert dispatch are implemented.
No complete approved AgentX result set is recorded yet.
