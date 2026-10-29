# AgentX 测量

使用 `agentx-harness==1.0.7` 和现有的 `inferencex-agentx-mvp` scenario.
使用 pinned dataset, seed `42` 和至少 900 秒 profiling.
使用 GPU 0 以外的四张显卡.
runner 默认使用 session-tree concurrency 作为 service sequence capacity.
使用 `--max-num-seqs` 可以覆盖这个值.

默认 dataset `semianalysis_cc_traces_weka_062126_256k` 的 context 范围为 256K.
context 选项选择上限. 它不会延长 dataset 中的 request.
1048576-token limit 不能证明在该长度下的执行.

额外的 full-context run 使用 `--dataset semianalysis_cc_traces_weka_062126`.
该 run 设置 `--cache-context 1048576`.
其 result 与批准的 `_256k` matrix 分开记录.
[完整 dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729) 允许 input length 达到 990016 个 proxy token.
报告观测到的最大 input length 和 total length, 不要只报告配置的上限.

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
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_agentx \
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

## 功能性记录

最新功能性记录在 GPU 1, 2, 3 和 4 上使用 TEP4 和 16 个 session tree.
它使用 `max_num_seqs=1`, 以适配可用 device memory 中的 model 和 cache.

| Setting | Value |
|---|---:|
| AgentX version | `1.0.7` |
| Profiling duration | 900 秒 |
| Cache context | 65536 token |
| GPU blocks | 512 |
| Mamba blocks | 64 |
| Maximum batched tokens | 16384 |
| Warmup requests | 16 |
| Profiling requests sent | 14 |
| Completed requests | 2 |
| Failed requests | 0 |
| Cancelled requests | 12 |
| Unaccounted requests | 0 |
| Drain duration | 120 秒 |
| Drain timeout | 是 |

[functional CSV](assets/agentx-tep4-16-functional.csv) 和
[functional dark scatter plot](assets/agentx-tep4-16-functional.png) 记录这次 run.
这两个 completed row 证明 end-to-end service execution 和 artifact export.
由于 service 使用一个 resident sequence, 它们不代表 concurrency 16 下的 TEP4 throughput.
harness phase counter 包含 drain timeout 后取消的 12 个 request.
raw request export 不包含这些 cancelled request.
因此, 只根据 raw request record 得到的 count 不完整.

当前 DEP4 command 使用 TEP4 executor fallback.
在 request ownership 和 expert dispatch 实现前, 不要把 DEP4 result 作为 DEP4 performance evidence.
当前还没有记录完整的 approved AgentX result set.
