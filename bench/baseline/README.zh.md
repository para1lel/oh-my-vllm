# 基准测试证据

[English](README.md)。agent 以英文原文为准。

本目录存放正式的汇总验收证据。不要放入 GPU profiling trace 或临时调优记录。各产物的解读和
按日期排列的索引见 [docs/acceptance.zh.md](../../docs/acceptance.zh.md)。

## 当前

- `2026-09-28-audit-mnt03-recurrent-operators.json` 记录干净 `0394727` 的受影响
  正式证据：8 项递归用例在固定 UUID 的 B200 上均通过输出和冻结 TileLang 门槛。
  这不是新的完整 147 项结论。
- `2026-09-28-audit-mnt02-krn04-operators.json` 记录干净提交 `312c54b` 的
  60 项选中 `quant`、`silu_quant`、`gates` 与 `recurrent` 正式用例。每项都通过
  输出验证和冻结 TileLang 对照。这是子集，不是新的 147 项完整判定。
- `2026-09-28-audit-gates-confirm-operators.json` 在第二张固定 UUID 的 B200
  上确认全部 13 项 `gates` 用例。此前一次 59/60 的失败尝试在验收索引中说明，原因未查明。
- `2026-09-28-audit-p1-operators.json` 是干净提交 `96e4ecc` 的受影响正式算子
  子集：在固定 UUID 的 B200 上，选中的 29 项 `prepare_attention` 与 `attention`
  用例全部通过。它不是新的 147 项完整判定。热路径修复后的当前 12 组框架性能尚未测量。
- `2026-09-22-refreshed-enginecore.json` 是冻结的 12 组官方 vLLM EngineCore 基线，采集自上游
  `e9f169d16b9408bb9ae44f75072b91a5521d733c`。
  - 每个 `rows[].artifact` 正是 `benchmarks/ttft.py` 需要的输入。
  - 把某一行序列化为 `json.dumps(artifact, indent=2)` 再加一个换行，即可得到它的
    `rows[].sha256`。
  - 未经用户明确授权，不要重新生成此文件。
- `2026-09-22-cuda-framework.json` 包含干净提交 `c36d1c9` 下通过验收的 12 组 CUDA 结果。
  - 每组都含候选方的原始重复测量，以及把它关联到冻结基线行的 `baseline_sha256`。
  - 它也保留了此前所有失败和被排除的尝试。
- `2026-09-22-cuda-operators.json` 包含全部 147 项正式算子判定。
- `2026-09-22-cuda-features.json` 包含：
  - 默认后端选择的正确性
  - 边界运行
  - 服务约束和生命周期检查
  - agentic 读回，附生成答案的局限
  - 清理快照

## 历史

其余 JSON 文件记录的是较早的阶段，均列在验收索引中：

- 2026-09-19 的适配器
- 2026-09-21 的服务、V2 和独立运行时阶段
- 2026-09-22 的 TTFT、性能差距和 TileLang 阶段
- CUDA 各里程碑子集

这些结果对当时测量的实现是正确的。2026-09-22 之前的各组使用原始基线和三次重复的协议，
因此不属于当前验收。

`baseline_*`、`clean_*`、`mtp_*` 文件和 `logs/` 是早期从 vLLM HTTP 服务端点采集的数据，
其设置、输入和计时边界都与 EngineCore 协议不同。不要把它们用作分母。

部分较早的产物使用了错误的 “Qwen3.5” 标签或 `qwen3.5-27b-fp8` 服务 ID。它们记录的是实际执行
情况，所以原样保留。模型是 Qwen3.8-27B-FP8。
