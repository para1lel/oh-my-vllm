# HTTP 性能测量

使用四张 B200 GPU 和完整 Rust 服务.
通过 [GPU wrapper](../../distributed.zh.md#gpu-租约) 选择 GPU.
性能测量不得使用锁频 GPU.
将原始 artifact 保存在仓库外.
将 `TMPDIR` 设为可写目录, 使用适合 Unix socket 的短路径.

## 批准的工作负载

批准的矩阵有 38 个用例:

| 模式 | Strategy | Batch | 输入 tokens | 输出 tokens | 用例数 |
|---|---|---:|---:|---:|---:|
| DSpark5 | TP4, TEP4, DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 36 |
| Ordinary | TP4 | 4 | 400000 | 64, 8192 | 2 |

输入和输出长度适用于每个请求.
使用 uncached prefix 和精确 tokenizer 长度.
使用贪心文本执行.
Runner 设置 `ignore_eos` 并检查完整 usage record.
当前没有记录完整的批准矩阵结果.

## 运行矩阵

按[服务配置](../../serving.zh.md#启动)构建服务和原生模块.
将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `ARTIFACT_ROOT` 设为新的仓库外目录.
运行矩阵:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_benchmark \
  --model "$MODEL_ROOT" \
  --reference-version deepseek-http-roofline-v6 \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

将[矩阵标识符](../../../development/serving/deepseek_matrix.py)传给 `--case` 以选择子集.
使用 `--cache-context` 设置物理缓存上下文容量.
使用 `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS=32768` 测量长上下文吞吐.
使用 `--request-timeout-seconds` 设置 HTTP client 和服务超时.
默认超时为 7200 秒.

计时包括提交, prompt 准备, 排队, 模型执行和响应解析.
它不包括模型加载和初始化.
CSV 标识 mode, strategy, batch, 精确长度, request count 和完整 HTTP 秒数.
它还记录 `effective_strategy`, `reference_seconds`, `reference_acceptance` 和 `efficiency_percent`.

Artifact 绑定 source, executable, native module, checkpoint, tokenizer, topology, configuration, sampling 和 command 身份.
Harness runtime 版本只标识 harness, 不标识每个 worker.
参考计算还验证四个 rank 的 runtime 身份和 HTTP usage.
结果 JSON 保留各 rank 资源, 参考范围, 假设和诊断项.
结构化 CSV 字段包含 JSON.

## 选择参考

Harness 默认值仍为历史版本 `deepseek-http-reference-v3.1`.
使用 `--reference-version deepseek-http-roofline-v6` 进行当前效率验收.
V6 是固定 target 算术, 放置方式, wire 格式和最大重叠条件下的资源下界.
将相同已接受工作拆成更多调用时, 其整次 HTTP 资源量不会增加.
使用 `--reference-version deepseek-http-protocol-estimate-v5` 查看历史 packed 协议估算.
使用 `--reference-version deepseek-http-resource-floor-v4` 选择有条件部分下界.
[参考契约](accounting.zh.md)规定格式, 必要性论证, 零计入项和身份.

`reference_case` 只提供静态规模.
HTTP 结果通过 `reference_trace` 使用绑定的运行时 trace.
策略回退没有参考值.
理想 draft 接受假设不是实测效率.

## 检查长输出行

当前效率验收要求使用 `deepseek-http-roofline-v6` 及其固定算法资源 roofline 范围.
检查覆盖全部 19 个批准的 8192 输出用例, 使用同一测量身份, 并要求每行效率至少 33%.
对完成的结果 JSON 文件运行检查:

```bash
scripts/with-env.sh python -m development.serving.deepseek_acceptance "$ARTIFACT_ROOT"/*-8192.json
```

检查从各绑定的运行时 trace 重新计算所选参考和效率.
缺行, 重复, 失败, 混合身份, 静态或参考变化的记录不能通过.
V3.1, v4 和 v5 不具备当前效率验收资格.
其重新计算的阈值诊断仍可查看, 但不能产生 target 通过结果.

旧计算器身份保持不变.
旧记录不能静默获得 v6 资格.
完整协议阶段的字节是诊断值, 不为 v6 下界增加串行秒数.

退出码 0 表示通过, 1 表示记录无效或存在失败行, 2 表示参考不具备资格.
通过不能替代[数值验收](evaluation.zh.md), 容量检查或 [AgentX 工作负载](agentx.zh.md).
