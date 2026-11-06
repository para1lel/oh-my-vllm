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

## Request and tree status

The [DeepSeek exporter](../../../development/serving/deepseek_agentx.py) keeps the existing strict request checks and successful HTTP CSV rows.
`accounting_complete` gives the accounting status for issued HTTP requests.
The final output and `summary.json` add `request_accounting_scope=issued_http_requests` and a separate `session_trees` object.
The same object is in `session-tree-status.json`, next to `request-counts.json`.

The runner also saves available tree evidence when the harness or request checks fail.
A report write error does not replace the original failure.
Failed request checks do not produce successful CSV rows.
Missing or invalid tree inputs give evidence limits and `null` counts where the inputs are necessary.

The [tree report](../../../development/serving/agentx_completion.py) has these fields:

- `status` is `incomplete` when evidence shows cancellation, errors, truncation, suppressed joins, pending work, or forced completion.
- `status` is `unknown` when no counterexample is known and the evidence cannot prove that all trees completed.
- `reasons` gives the evidence limits and the conditions that prevent a pass.
- `scope` identifies the benchmark, profiling phase, timestamps, and selected log lines.
- `observed_root_sessions` gives the sent, completed, and cancelled root session counters, plus distinct root IDs in profiling raw records.
- `tree_counts` keeps admitted and completed tree counts as `null`. Root session counters and distinct IDs are not complete tree counts.
- `pending_children` and `pending_joins` give observed cleanup counts. Missing counts are `null`, not zero.
- `truncated_children`, `suppressed_joins`, and `forced_tree_slots` give explicit counts. Missing counts are `null`.
- `facts` keeps source lines or JSON field locations for the reported conditions.
- `source_digests` gives source paths and SHA256 digests. A missing source has a `null` digest.
- `strict_target_passed` is `false` for each state.

The log must match the summary timestamps and the single profiling phase with index zero.
The report selects the PhaseRunner profiling teardown after phase completion, with a timestamp at most one second after completion.
The report uses this teardown for slot counts and branch cleanup.
Branch cleanup must be after the completion line and before the teardown line.
Its timestamp must be between the completion and teardown timestamps.

The report cannot identify the phase of later cleanup.
Each log fact must have a timestamp in the selected interval.
Unbound cleanup counts are `null`, with a reason in `reasons`.

Raw observations must have explicit `phase_index=0` and `profiling_index=0`.
Each raw request must start and end in the summary interval, with the start before or at the end.
If a profiling raw record cannot meet these conditions, the distinct root count is `null`.
These tree checks do not change the shared request checks.

Warmup records and other runs cannot supply profiling cleanup evidence.
Missing or ambiguous boundaries cannot prove completion.
The current exports lack the full tree inventory and natural terminal evidence, so this report cannot return `complete`.

The CLI exit contract stays unchanged.
A valid request measurement can exit zero when the tree state is `incomplete` or `unknown`.
Callers must check `session_trees` for the strict all-tree target.
The approved harness, scenario, dataset, seed, concurrency, and request deadline stay unchanged.
The 900-second deadline stops all new requests and the later turns of existing sessions.
The report does not extend that deadline or resume cancelled work.

No complete approved AgentX result set is recorded.
