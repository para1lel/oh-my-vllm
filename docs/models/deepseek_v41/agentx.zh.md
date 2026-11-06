# AgentX 测量

使用 `agentx-harness==1.0.7` 和现有的 `inferencex-agentx-mvp` scenario.
使用 pinned dataset, seed `42` 和至少 900 秒 profiling.
使用四张没有锁频的显卡.
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

使用 repository 的 `oh-my-vllm` Conda environment 运行一个 case. 将
`OH_MY_VLLM_CONDA_PREFIX` 设为该 environment, 并将 `ARTIFACT_ROOT` 保持在 repository 外.

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

runner 记录 source, executable, native module, checkpoint, tokenizer, topology, strategy 和 concurrency.
raw harness record 和 service log 保留在 repository 外.
CSV 和暗色背景散点图是 approved result artifact.

## 请求和树状态

[DeepSeek 导出器](../../../development/serving/deepseek_agentx.py) 保留现有严格请求检查和成功 HTTP 请求的 CSV 行.
`accounting_complete` 给出已发出 HTTP 请求的对账状态.
最终输出和 `summary.json` 新增 `request_accounting_scope=issued_http_requests` 以及独立的 `session_trees` 对象.
相同对象保存在 `session-tree-status.json`, 与 `request-counts.json` 位于同一目录.

harness 或请求检查失败时, runner 也保存已有的树证据.
报告写入错误不会替代原失败.
请求检查失败时不会生成成功 CSV 行.
树输入缺失或无效时, 报告给出证据限制, 并将依赖这些输入的计数设为 `null`.

[树报告](../../../development/serving/agentx_completion.py) 包含以下字段:

- 证据显示取消, 错误, 截断, join 被禁止, 工作遗留或强制结束时, `status` 为 `incomplete`.
- 没有已知反例, 且证据不能证明全部树完成时, `status` 为 `unknown`.
- `reasons` 给出证据限制和阻止通过的条件.
- `scope` 标识 benchmark, profiling 阶段, 时间戳和选定日志行.
- `observed_root_sessions` 给出根会话的发送, 完成和取消计数, 以及 profiling 原始记录中的不同 root ID 数量.
- `tree_counts` 将准入和完成树数量保留为 `null`. 根会话计数和不同 ID 数量不是完整树数量.
- `pending_children` 和 `pending_joins` 给出观测到的清理时计数. 缺失数量为 `null`, 不是零.
- `truncated_children`, `suppressed_joins` 和 `forced_tree_slots` 给出明确计数. 缺失数量为 `null`.
- `facts` 为报告中的条件保留来源行号或 JSON 字段位置.
- `source_digests` 给出来源路径和 SHA256 摘要. 缺失来源的摘要为 `null`.
- 每种状态的 `strict_target_passed` 均为 `false`.

日志必须匹配 summary 时间戳以及唯一的索引为零的 profiling 阶段.
报告选择阶段完成后的 PhaseRunner profiling teardown, 其时间戳不得晚于完成时间一秒.
报告将同一 teardown 用于槽计数和分支清理.
分支清理必须位于完成行之后, teardown 行之前.
其时间戳必须位于完成和 teardown 时间戳之间.

本报告不能确认更晚清理的阶段归属.
每条日志事实的时间戳必须位于选定区间内.
未绑定的清理计数为 `null`, 并在 `reasons` 中说明原因.

原始记录观测必须有显式的 `phase_index=0` 和 `profiling_index=0`.
每个原始请求必须在 summary 区间内开始和结束, 且开始时间不晚于结束时间.
profiling 原始记录无法满足这些条件时, 不同根 ID 计数保持 `null`.
这些树检查不改变共享请求检查.

Warmup 记录和其他运行不能提供本次 profiling 的清理证据.
边界缺失或有歧义时不能证明完成.
当前导出缺少完整树库存和自然终止证据, 因此报告不能返回 `complete`.

CLI 退出合同保持不变.
请求测量有效且树状态为 `incomplete` 或 `unknown` 时, 可以退出零.
调用者必须检查 `session_trees` 来判断严格的全树目标.
已批准的 harness, scenario, dataset, seed, concurrency 和请求截止时间保持不变.
900 秒截止时间会停止全部新请求和已有会话的后续轮次.
报告不会延长此截止时间, 也不会恢复已取消的工作.

当前没有记录完整的批准 AgentX 结果集.
