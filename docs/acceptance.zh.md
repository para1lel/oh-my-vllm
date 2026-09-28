# 验收证据

[English](acceptance.md)。agent 以英文原文为准。

## 当前修复证据 — 2026-09-28

干净提交 `40e3e57` 在 B200 UUID `GPU-1b174534-ebba-826f-b452-8e7f3c05c301`
上通过全部 66 项选中的 KRN-08 受影响正式用例：`norm`、`add_norm`、
`gated_norm`、`convolution` 各 13 项，`recurrent` 8 项，`qk` 6 项。每项均通过
输出验证，记录 CUDA 主机分派快路径恰好一次、通用路径零次，并在三轮、每轮 20 对
交替测量中快于冻结 TileLang。最小的单侧 95% 正收益下界为
0.000000213335 ms，三轮最大 spread/median 为 1.695%。
[正式子集](../bench/baseline/2026-09-28-audit-krn08-operators.json) 源码身份干净，
`selected_passed=true`；`passed=false` 表示未覆盖全部 147 项。自有 GPU 进程
已退出。当前完整 GPU 和 12 行框架门槛仍待执行。

干净提交 `96e4ecc` 在固定 UUID 的 B200 上通过全部 29 项选中的受影响正式算子
用例：13 项 `prepare_attention` 和 16 项 `attention`。最小的单侧 95% 正收益
下界为 0.000796 ms。[汇总产物](../bench/baseline/2026-09-28-audit-p1-operators.json)
保留源码哈希、冻结参考哈希、协议、选中用例和原始采集哈希。这个子集不能证明新的
147 项完整矩阵结果。

干净提交 `312c54b` 修正 KRN-04 两个入口后，通过全部 60 项选中的
`quant`、`silu_quant`、`gates` 与 `recurrent` 用例。每项都有三轮热身后正式测量、
每轮 20 对交替样本、每样本 100 次 graph 重复，通过输出验证且单侧 95% bootstrap
收益下界为正。最小收益下界为 0.00000699734 ms，两端三轮的最大 spread/median 为
0.846%。[正式子集](../bench/baseline/2026-09-28-audit-mnt02-krn04-operators.json)
记录 B200 UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`、干净源码身份及
实时 nvcc/module 哈希。`selected_passed=true` 只适用于这 60/60 项；`passed=false`
表示 147 项尚未全覆盖。干净提交 `6bd2119` 上的首次尝试因历史 KRN-04 误拒而中止，
不算验收证据。后来干净提交 `312c54b` 的一次尝试通过 59/60：1248-token `gates`
用例的 bootstrap 收益下界为负，两端延迟均抬升，但未检测到 compute 进程干扰。
原因未证实，因此该次尝试作废。同一 UUID 的完整重测通过 60/60；另一 UUID
`GPU-80cebbaf-a106-2605-b902-f5f9a08645ea` 上的独立
[13/13 gates 确认](../bench/baseline/2026-09-28-audit-gates-confirm-operators.json)
也通过。确认测试的一次初始启动因外部 GPU 进程出现，在测量前被停止。当前完整 GPU
和 12 行框架门槛仍待执行。

干净提交 `0394727` 将 MNT-03 状态池对齐检查写为向量字节对齐后，通过全部 8 项选中
的递归用例。[受影响正式产物](../bench/baseline/2026-09-28-audit-mnt03-recurrent-operators.json)
记录 B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`、输出验证、3 × 20
交替配对和单侧 95% 正收益下界；最小下界为 0.000064624 ms。源码干净，
`selected_passed=true`；`passed=false` 仅因未覆盖完整 147 项。自有 GPU 进程已退出。

`96e4ecc` 后的完整 B200 pytest 通过 211 项测试及 32 个子测试，六项 context
边界占位测试跳过。后续 CPU/协议修复 `d33b844` 通过 31 项聚焦 Python 测试和
26 个子测试，以及 Rust workspace 测试、格式、Ruff 与 Clippy；该提交未重跑完整
GPU 套件。热路径改动后的当前 12 组吞吐和 TTFT 仍未验证。下文 2026-09-22
的结果只适用于 `c36d1c9`。

## 历史 CUDA 验收基线 — 2026-09-22/23

**性能。** 干净提交 `c36d1c9` 显式选择 CUDA，通过全部 12 组冻结 vLLM 对比。

- **工作负载：** 输入 32768 的 ordinary、MTP4 和 prefix，各 batch 1/2/4；外加仅 ordinary
  的输入 131072，batch 1/2/4。输出均为 4096 token。
- **协议：**
  - `benchmarks/ttft.py`，2 次完整预热和 5 次正式重复测量。
  - 冻结基线的 CPU affinity。
  - UUID 固定的 B200，并监控进程。
  - 稳态编译/捕获审计。

| Workload | 基线 tok/s | CUDA tok/s | 吞吐比 | TTFT 比 |
|---|---:|---:|---:|---:|
| ordinary-32768-1 | 90.10 | 105.30 | 116.86% | 93.14% |
| ordinary-32768-2 | 163.50 | 193.08 | 118.09% | 95.60% |
| ordinary-32768-4 | 288.41 | 331.64 | 114.99% | 96.37% |
| mtp-32768-1 | 329.36 | 329.24 | 99.96% | 93.92% |
| mtp-32768-2 | 497.37 | 490.52 | 98.62% | 94.81% |
| mtp-32768-4 | 729.62 | 711.53 | 97.52% | 97.50% |
| prefix-32768-1 | 93.01 | 109.10 | 117.30% | 82.96% |
| prefix-32768-2 | 173.52 | 217.06 | 125.09% | 85.02% |
| prefix-32768-4 | 325.31 | 380.50 | 116.97% | 76.01% |
| ordinary-131072-1 | 72.35 | 82.68 | 114.29% | 92.95% |
| ordinary-131072-2 | 114.06 | 129.45 | 113.49% | 91.54% |
| ordinary-131072-4 | 162.85 | 178.82 | 109.80% | 93.82% |

- **稳定性：** 每组都满足吞吐 ≥ 95%、TTFT ≤ 110%，两个引擎在两项指标上的
  `(max-min)/median` 都 ≤ 10%。进程轮询无法排除样本之间任意短暂的干扰。
- **此前的尝试：**
  - prefix batch 1 两次、prefix batch 2 一次未满足稳定性；在稳定性规则不变的情况下完整重跑后通过。
  - 有三组通过的数据与其他运行共用了物理 CPU 核，保守起见已用完整重跑替换。
  - prefix TTFT 抖动的原因尚未查明；一次临时的 GC 诊断在测量区间内未发现回收，该诊断不作为验收依据。
  - 所有尝试都保留。
- **基线：** 没有重跑 vLLM 基线。
- **对比性质：** 这是不同运行、不同框架版本之间的对比，不是同源配对实验。
- **余量：** MTP 各组余量最小；MTP batch 4 比门槛只高 2.52 个百分点。

**算子（REQ-KERNEL-002）。** 全部 147 项正式静态用例都快于冻结的 TileLang 实现。

- **判定规则：** 三轮热身后测量、每轮 20 对交替测量；每轮 CUDA 中位数都更快；节省时间的单侧
  95% bootstrap 下界为正。
- **余量：** 较小的收益可能只有纳秒级，不声称任何最低余量。
- **与延迟分开：** TileFoundry HIR 估计值和 Nsight 计数器都与验收延迟分开记录。

**正确性和功能。**

- **测试套件：** 在默认 CUDA 下、不设置任何 backend/CUDA_HOME/TVM 覆盖，套件通过 174 项测试
  及 31 个子测试，没有跳过，也没有放宽容差。
- **FP64 探针：** 实际模型的 ordinary 和 MTP4 探针通过。
- **真实文本：**
  - ordinary 和 MTP4 真实文本检查通过，包括强制抢占和前缀复用。
  - 十二组 MTP4 约束用例、各档思考强度和生命周期检查通过。
  - 四次长 strict-JSON 调用通过。
- **边界运行：** 六组输入 258048、输出 4096 的运行完成，无 OOM、无重算抢占：

  | 模式 | Batch | 输出 tok/s | 提议 / 接受的草稿 |
  |---|---:|---:|---|
  | ordinary | 1 | 56.25 | — |
  | ordinary | 2 | 78.58 | — |
  | ordinary | 4 | 96.65 | — |
  | MTP4 | 1 | 89.40 | 5010 / 2842 |
  | MTP4 | 2 | 111.66 | 8989 / 5939 |
  | MTP4 | 4 | 126.06 | 16272 / 12306 |

  PyTorch reserved 峰值为 134.69 GB，这不是整卡用量；这些运行是容量诊断，不是吞吐门槛。
- **oh-my-pi 读回（干净提交 `ef07b2b`）：** 两种 API 都通过核心任务。
  - Chat 和 Responses 各成功读取九次，分别发出 5 次和 3 次模型请求。
  - 提议/接受的草稿分别为 3297/1943（Chat）和 3687/2164（Responses）。
  - 部分答案不准确之处已记录在功能产物中，不声称回答完全基于文件。
- **清理：** 清理结果不依赖模型答案，是独立验证的。

**来源。**

- 147/12 的计时来自 `c36d1c9`。
- `030f60f` 只修改默认选择和身份报告，kernel、模型和派发实现均未改变。
- 默认选择的测试套件和服务运行使用的是 `ef07b2b` 之后的工作树（其 diff 哈希记录在 feature
  artifact 中），该工作树后来成为 `030f60f`；没有在该提交本身上重跑。

**证据文件**（位于 `bench/baseline/`）：

- `2026-09-22-cuda-operators.json`
- `2026-09-22-cuda-framework.json`，其各行的 `baseline_sha256` 与
  `2026-09-22-refreshed-enginecore.json` 中内嵌的产物一致
- `2026-09-22-cuda-features.json`

**范围限制。**

- 2026-09-23 的[代码审计](audit-2026-09-23.zh.md)在这条路径上没有发现产生错误 token 的缺陷。
- 审计确实记录了服务可用性和证据完整性方面的缺口，其中两项尤其相关：
  - 用于比较的候选方配置是由测试工具设定的，而不是观测到的（EVD-02）。
  - 边界运行没有自动化测试覆盖（EVD-07）。
- 上述限制适用于以上证据。

## 历史证据索引

较早的阶段已被取代。原始产物仍保留在 `bench/baseline/` 中，作为当时测量内容的记录。不要把
它们当作当前状态引用，也不要把 2026-09-22 之前的基线用作当前的分母。

| 日期 | 阶段 | 当时的结果 | 产物 |
|---|---|---|---|
| 2026-09-19 | 原始 GPUWorker 适配器，9 组对比原始 EngineCore 基线（2 次预热/3 次重复） | 9/9 ≥ 95% | `2026-09-19-acceptance.json`、`2026-09-19-mtp-bs2-repeat.json`、`2026-09-19-batch-text.json`、`2026-09-19-paired-investigation.json` |
| 2026-09-21 | 服务扩展（真实 MTP4 约束、生命周期、oh-my-pi） | 通过 | `2026-09-21-serving-acceptance.json` |
| 2026-09-21 | V2 Model Runner，9 组对比原始基线 | 9/9 ≥ 95% | `2026-09-21-v2-acceptance.json` |
| 2026-09-21 | 独立运行时（移除 vLLM），9 组对比原始基线 | 9/9 ≥ 95% | `2026-09-21-independent-{stage1,stage2,stage3,acceptance,batched-sampling,grouped-decode,proposal-graph}.json` |
| 2026-09-22 | 刷新的官方 vLLM 基线 `e9f169d1…`，12 组 | 冻结的分母 | `2026-09-22-refreshed-enginecore.json` |
| 2026-09-22 | TTFT/长上下文调查及 MTP 差距诊断 | 10/12；MTP bs4 为 94.87% | `2026-09-22-ttft-{stage1,agentic,discarded}.json`、`2026-09-22-performance-gap{,-candidates}.json`、`2026-09-22-{native-decode-fusions,prefill-*,ragged-prefill,strided-metadata}.json` |
| 2026-09-22 | TileLang 迁移（`da75c02`），12 组 | 在 10% 稳定性下 12/12 | `2026-09-22-tilelang-{acceptance,correctness}.json` |
| 2026-09-22 | CUDA 里程碑子集（未提交的源码，诊断用） | 部分 | `2026-09-22-cuda-*-progress.json` |

`baseline_*`、`clean_*`、`mtp_*` 文件和 `logs/` 是早期从 vLLM HTTP 服务采集的数据，其设置、
输入、计时边界和 MTP 配置都与 EngineCore 协议不同，不能与之比较。部分较早的产物带有错误的
“Qwen3.5” 标签，模型一直是 Qwen3.8-27B-FP8。

## 运行范围

多卡、CPU KV swap、多模态输入、LoRA 和生产部署不在范围内。OpenAI 兼容的 HTTP 服务已实现，
其验收由上面的功能和 agentic 检查组成，没有 HTTP 吞吐门槛。EngineCore 测量不能验证 HTTP
路径的性能。
