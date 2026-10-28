# AgentX measurement

Use `agentx-harness==1.0.7` and the existing `inferencex-agentx-mvp` scenario.
Use the pinned dataset, seed `42`, and at least 900 seconds of profiling.
Use four GPUs apart from GPU 0.
The runner uses the session-tree concurrency as the default service sequence capacity.
Use `--max-num-seqs` to override it.

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
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_agentx \
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

The current DEP4 command uses the TEP4 executor fallback.
Do not treat a DEP4 result as DEP4 performance evidence until request ownership and expert dispatch are implemented.
No complete approved AgentX result set is recorded yet.
