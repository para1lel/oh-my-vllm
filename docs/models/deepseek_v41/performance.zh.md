# HTTP 性能测量

使用四张 B200 显卡和完整 Rust service.
选择没有锁频的显卡.
原始 artifact 保留在 repository 外.
将 `TMPDIR` 设为具有写权限的目录.
该目录必须使用适用于 Unix socket 的短路径.

approved matrix 有 38 个 case.

| 模式 | Strategy | Batch | Input tokens | Output tokens | Case 数 |
|---|---|---:|---:|---:|---:|
| DSpark5 | TP4, TEP4, DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 36 |
| Ordinary | TP4 | 4 | 400000 | 64, 8192 | 2 |

Input 和 output length 适用于每个 request.
使用 uncached prefix 和 exact tokenizer length.
在 model sampling contract 扩展前使用 greedy text execution.
runner 设置 `ignore_eos` 并检查完整 usage record.

使用以下命令运行 matrix:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_benchmark \
  --model "$MODEL_ROOT" \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

使用 `development/serving/deepseek_matrix.py` 中的 identifier 传给 `--case` 选择子集.
使用 runner 的 `--cache-context` option 设置 `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT`.
对于 long-context throughput, 设置 `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS=32768`.

使用 `--request-timeout-seconds` 为 HTTP client 和 service 设置相同的超时.
默认值为 7200 秒. Measurement identity 记录所选值.
runner 记录 source, executable, native module, checkpoint, tokenizer, topology 和 configuration identity.
它还记录测试程序的 Python, PyTorch, CUDA, NCCL 和数值设置.
这些设置不独立标识每个工作进程的运行时.

完整 timer 包含 request submission, prompt preparation, queueing, model execution 和 response parsing.
它不包含 model loading 和 initialization.
CSV 记录 mode, strategy, batch, exact length, request count, full seconds 和 output rate.
它还记录 `effective_strategy`, `reference_seconds`, `reference_acceptance` 和 `efficiency_percent`.

## 可选的 v4 计量

默认值为 `deepseek-http-reference-v3.1`.
添加 `--reference-version deepseek-http-resource-floor-v4` 可选择已审查的 v4 计算器.
其结果是 **有条件的部分资源下界 (conditional partial resource floor)**.
它不证明完整 roofline, resident 容量, 或所有相同输出算法的下界.

目标矩阵算术, 数值次序, 格式, 参数放置和各 rank 上复制执行的投影保持固定.
MoE wire 信用还要求以下条件:

- 输出分片保持固定, 不增加权重副本或重新计算投影.
- 每次计入的交付使用完整 BF16 行.
- 不使用未建模的数值压缩或跨行数值复用.

对于量化传输, 复用或重新分片, 获取新的审查和身份.
HBM 和 host 严格信用为零.
保留容量和必要新传输的证明仍待补齐.
Embedding, `wo_b`, vocabulary-head, index, Engram 及 DEP4 dispatch/combine wire 的严格信用为零.
MTP proposal, 拒绝或重复工作, padding, 状态复制, scheduler 开销, 经验 latency, `S/2` 和非线性标量估算均不计入信用.
权重并集及 `direct_protocol_*` 流量仅作诊断.

harness 在 v4 计算前验证四 rank runtime 身份和 HTTP usage.
`source_trace_version` 和 `source_trace_rate_profile` 标识 v3.1 源绑定.
`accounting_version`, `rate_profile` 和 `assumptions_sha256` 标识所选计算器.
身份 manifest, 结果 JSON, CSV 和 reference event 记录其范围, 条件, 零信用项和诊断项.
CSV 中的结构化字段包含 JSON.
结果 JSON 还保留每个 rank 的资源数值.

`reference_case` 只提供静态规模.
HTTP 路径使用 `reference_trace`.
该选项不证明 frozen 数值, GPU, HTTP 或 33% 验收.
19 条长输出行的逐行 33% 目标保持不变.

## Packed 项目参考与 19 行检查

选择 `--reference-version deepseek-http-protocol-estimate-v5` 可使用适配 packed 的项目参考估算.
默认仍是 v3.1.
旧版本的参考数值不变.
v5 参考使用 BF16 window 行, 带 G16 scale 的 FP4 压缩行, 以及带 G32 scale 的 FP4 index 行.
共享物理历史与各使用方流量有独立计数.
Index workspace 解码按每个物理 source 和 call 计一次.

Collective 阶段遵循数据依赖.
TP4 activation 交付在 expert up 和 down 投影之间.
DEP4 dispatch 在 expert 计算之前, 反向 FP32 combine 在其之后.
各方向分别计数.

路由 metadata 和 count exchange 只作协议诊断, 不增加参考信用.
参考保留已声明的资源和两微批松弛.
串行协议秒数与未改动的 v4 部分下界是独立诊断项.

该项目参考不是普适物理最小值.
Cache 保留, padding, launch latency, overlap 和 MTP 工作不构成完整执行时间模型.
运行身份记录每个 rank 的 batch decode 和 Graph 设置.
Graph 设置包含 graph 数量上限.
这些设置不标识每一步的 capture 或 Eager 回退.

使用以下命令检查完成的结果 JSON 文件:

```bash
scripts/with-env.sh python -m development.serving.deepseek_acceptance "$ARTIFACT_ROOT"/*-8192.json
```

检查从已绑定的 runtime trace 重算所选参考和百分比.
检查使用完整 19 条长输出 case, 同一测量身份, 以及每行至少 33%.
缺行, 重复, 失败, 混合身份, 静态或参考变化的记录不能通过.
v4 部分下界不具备该完整项目参考检查的资格.
旧 v3.1 历史格式不具备当前 packed 结果的资格.

退出码 0 表示通过, 1 表示记录无效或有失败行, 2 表示参考不具备资格.
通过不能替代数值, 容量或 AgentX 验收.

当前还没有记录完整的 approved matrix result.
理想计划计入全部五个 draft. 它不是实测 efficiency 结果.
策略回退没有参考值.
使用 [参考计量](accounting.zh.md) 了解 trace 契约和缺少的计数.
当前会话的 SOTA 目标不加入 repository performance gate.
