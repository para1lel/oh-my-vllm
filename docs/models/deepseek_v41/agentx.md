# AgentX measurement

Use `agentx-harness==1.0.7` and the existing `inferencex-agentx-mvp` scenario.
Use the pinned dataset, seed `42`, and at least 900 seconds of profiling.
Use four GPUs that have no locked clock.
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

Run one case with the repository `oh-my-vllm` Conda environment. Set
`OH_MY_VLLM_CONDA_PREFIX` to that environment and keep `ARTIFACT_ROOT` external
to the repository.

```bash
export OH_MY_VLLM_CONDA_PREFIX="$CONDA_PREFIX"
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_agentx \
  --model "$MODEL_ROOT" \
  --agentx-python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" \
  --strategy tep4 \
  --concurrency 16 \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/agentx-tep4-16.csv \
  --figure docs/models/deepseek_v41/assets/agentx-tep4-16.png
```

The runner records source, executable, native modules, checkpoint, tokenizer, topology, strategy, and concurrency.
Raw harness records and service logs stay in a separate location.
The CSV and dark-background scatter plot are approved result artifacts.

No complete approved AgentX result set is recorded.
