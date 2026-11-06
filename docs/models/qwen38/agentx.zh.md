# AgentX 测量

使用 AgentX 在单张 NVIDIA B200 上, 按记录的 Agent 请求模式测量 MTP4.
测试工具根据 trace 重建匿名文本.
此测量检查服务性能, 不检查 Agent 任务质量.

## 配置

使用 `agentx-harness==1.0.7` 和场景 `inferencex-agentx-mvp`.
使用数据集 `semianalysis_cc_traces_weka_062126_256k`, 种子 `42` 和 `900` 秒的正式采样时间.
轨迹起点比例为 `0.0` 和 `1.0`.

最大上下文长度为 `262144` 个 token.
测试工具按此上限筛选 trace, 不截断请求.
服务检查准备后的提示词长度和请求的输出长度.

并发 `4` 表示四个活跃的会话树.
子 Agent 可以使活跃 HTTP 请求超过四个.
服务最多调度 `4` 个请求, token 预算为 `32768`.
缓存使用[运行时张量预算](cache.zh.md#cache-内存).

测试工具使用流式响应和 `ignore_eos=true`.
请求种子为 `42`, 关闭推理, 使用服务采样默认值.
EOS token 计入输出上限, 不停止生成.
停止字符串和显式最小输出长度保持原有行为.

场景保留 trace 延迟, 并使用 `first_turn_prefix` 隔离缓存.
系统空闲间隔上限为 `10` 秒.
快照请求在正式采样前预热缓存.
预热后开始正式采样. 最后的排空阶段允许活跃请求完成.

预热和排空时间不计入这 `900` 秒.
默认的正式采样排空上限为正式采样截止后的 `120` 秒.
使用 `--grace-period SECONDS` 更改此上限.
该值必须是有限数且至少为零. 没有固定的上限.
零表示正式采样截止后不再提供额外响应时间.

独立的 Agent 预热宽限上限保持为 `1800` 秒.

正式采样排空期间完成的响应仍计入正式采样结果.
所有 credit 返回且没有剩余会话树任务时, 测试工具可以提前结束排空.
仅有零个活跃 HTTP 请求不能保证提前结束.
排空截止时, 测试工具取消未返回的 credit.
取消清理和结果导出可能在此截止时间后继续耗时.

启动器在 `7200` 秒后停止测试工具, 该时间包含设置和预热.

场景规则见[上游流程](https://github.com/SemiAnalysisAI/agentx-harness/blob/agentx-v1.0.7/docs/tutorials/agentx-mvp.md).

## 运行

按[服务指南](../../serving.zh.md#启动)构建服务和原生模块.
测试工具使用 repository 的 `oh-my-vllm` Conda environment.
将 `OH_MY_VLLM_CONDA_PREFIX` 设为该 environment, 将 `ARTIFACT_ROOT` 设为仓库外的新目录.
将 `MODEL_ROOT` 设为本地 checkpoint 目录.
测量时使用 [GPU 包装脚本](performance.zh.md#测量命令).

```bash
export OH_MY_VLLM_CONDA_PREFIX="$CONDA_PREFIX"
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.agentx \
  --model "$MODEL_ROOT" \
  --agentx-python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/qwen38/assets/agentx-mtp4.csv \
  --figure docs/models/qwen38/assets/agentx-mtp4.png
```

如果指定文件已存在, 使用新的 CSV 和图片路径.
命令启动并停止 MTP4 服务.
它记录源码, 可执行文件, 原生模块, checkpoint, tokenizer, 设备和命令的身份.
身份文件中的 `agentx` 项和 `command.json` 给出工作负载配置.
共享的 HTTP 性能核算字段不适用于这些结果.
执行 trace 的身份记录给出实际 cache 容量和内存计算.

[缓存内存契约](cache.zh.md#cache-内存)规定当前 FP8 page 和 GDN checkpoint 容量.
选定容量必须满足 `P = 64 * C` 和完整字节预算.
新布局的 AgentX 行为和物理容量均未验证.
成功 HTTP 请求不能证明全部会话树已完成.

原始记录, 服务日志, 执行 trace 和计数报告保留在仓库外.
获准保存的 AgentX CSV 和图片是此次测量的仓库产物.

## 请求数据

每行 CSV 和每个点表示一个成功的正式采样请求.
整理工具使用完整响应 ID, 将流式 usage 与服务指标关联.
它拒绝缺失的缓存计数, 重复 ID, 不一致的长度和无效计时.
完成的请求数必须与测试工具的汇总一致.
失败, 取消, 未完成和上下文溢出请求在外部报告中分别计数.
没有有效场景结果的运行会被整理工具拒绝.

更改排空上限不会将这些请求变成成功的 CSV 行.
更改上限时应比较这些计数. 更短的上限可能使活跃请求无法完成.

| 字段 | 定义 |
|---|---|
| `processing_start_s` | 服务端开始处理时间减去正式采样开始时间, 单位秒 |
| `output_tokens_per_second` | 输出 token 数除以服务端处理秒数 |
| `cache_hit_ratio` | 缓存命中的提示词 token 数除以提示词 token 数 |
| `client_seconds` | 客户端从请求开始到最后一个流式事件的单调时钟耗时 |
| `queue_seconds` | 服务端开始处理前的接入排队时间 |

服务端处理时间包含请求准备, 后续调度等待和生成.
它不包含此前的接入排队和完整网络传输.
此输出速率不是纯 decode 速率.
颜色使用实测前缀复用率, 范围固定为 0% 到 100%.
缺失缓存数据属于错误, 不记为零.

从 CSV 重新生成图片:

```bash
scripts/with-env.sh python -m development.serving.agentx_plot \
  --csv docs/models/qwen38/assets/agentx-mtp4.csv \
  --figure docs/models/qwen38/assets/agentx-mtp4.png
```
