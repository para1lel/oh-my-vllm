# AgentX 测量

使用 `agentx-harness==1.0.7` 和现有的 `inferencex-agentx-mvp` scenario.
使用 pinned dataset, seed `42` 和至少 900 秒 profiling.
使用 GPU 0 以外的四张显卡.

approved case 如下:

| Session-tree concurrency | Strategy |
|---:|---|
| 4 | TP4 |
| 16 | TEP4 |
| 64 | DEP4 |

concurrency 表示 session tree 数量.
observed HTTP concurrency 可能更高.
service limit 是 64 个 active request.
queueing, failed, cancelled, incomplete, overflow 和 completed count 分开记录.

使用以下命令运行一个 case:

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

runner 记录 source, executable, native module, checkpoint, tokenizer, topology, strategy 和 concurrency.
raw harness record 和 service log 保留在 repository 外.
CSV 和暗色背景散点图是 approved result artifact.

当前 DEP4 command 使用 TEP4 executor fallback.
在 request ownership 和 expert dispatch 实现前, 不要把 DEP4 result 作为 DEP4 performance evidence.
当前还没有记录完整的 approved AgentX result set.
