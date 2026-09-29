# 交接记录 — 2026-09-29

[English](handoff.md)。agent 以英文原文为准。

## 语义 IR 已验收检查点（2026-09-29）

REQ-IR-001 现通过 Python `torch.library` 语义算子管理 Qwen/MTP 模型调用的项目自有 CUDA 和关键 FlashInfer 入口。生产默认以 fullgraph `torch.compile` 编译 prefill、target decode、MTP draft 与 proposal，并由现有手工 CUDA Graph 包住后三类单元。provider 只按静态元数据选择；缓存修改具有显式 schema，PyTorch 参考实现须显式选为调试 provider。实现包含受检查的 18 个调用点清单，以及保守的 BF16 SiLU/FP8 图改写。真实 27B 的 eager/compiled 测试比较 prefill 接续 decode 的输出和写入缓存，并验证变化的 MTP 重放元数据。不自动回退到 eager 或参考实现。

干净提交 `619c9d9` 通过 `scripts/test.sh full`：**544 项测试、70 个子测试**，包含全部六组 258048+4096 边界运行；B200 UUID 为 `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`（日志 `/tmp/oh-my-vllm-ir-full-gpu-green.log`）。`scripts/test.sh cpu` 通过 **303 项测试、70 个子测试**（`/tmp/oh-my-vllm-ir-cpu-verified.log`）。Ruff、Rust workspace 测试、fmt、行宽及 Clippy 均通过。只读子代理复审未发现剩余 P1/P2 正确性问题。第一次完整 GPU 运行因所选 GPU 上的无关进程而无效；第二次通过 543/544 项并暴露旧 mock 返回非 Tensor 的问题；修复后定向测试和最终完整套件均通过。任务启动的 GPU 进程已退出。

[正式算子证据](../bench/baseline/2026-09-29-ir-operators.json)显示干净源码 `619c9d9` 的 147 项 CUDA/冻结 TileLang 对照全部通过；最小单侧 95% 正收益下界为 `0.0000036639670530955112 ms`。[框架证据](../bench/baseline/2026-09-29-ir-framework.json)显示同一源码的 12 行全部合格：吞吐为冻结基线的 99.27–125.63%，TTFT 为 76.92–95.91%。每个合格行均预热两次、测量五次，离散度至多 10%，无预抢占，稳态日志和缓存审计通过。九行使用一块 B200，三个 prefix 行使用另一块；每行均为单 GPU。此前三次完整的 prefix batch 1 测量因 TTFT 离散度 17.378%、20.976% 和 20.177% 被拒收；另有六次尝试因无关 GPU 进程进入而中断。框架摘要保留原始身份、哈希和拒收记录。孤立 TTFT 尖峰原因未证实，且早于本轮 IR。现有审计未见计量窗口内的编译或捕获日志及磁盘缓存证据，但无法排除静默的内存中重编译；在计量边界记录编译计数属于后续增强。

REQ-IR-001 在已测工作集内已验收。4096 的 Dynamo 重编译上限及按单元的编译警告不证明长期形状切换的缓存占用有界；已测工作集之外的逐出/重捕获压力仍是开放容量限制。没有证明安全的临时目的地时不启用 activation 捐赠。未加入 vLLM 运行时依赖。

## 以前已验收的状态

CUDA 迁移在 `c36d1c9` 的原始验收属于历史结果。整库审计修复的干净源码
`e3c42e0` 现已通过完整 B200 套件、147 项正式算子矩阵，以及经一次高离散度
单行重测后的 12 个合格框架行。PY-04、PY-06、KRN-06 保留下文明确的未闭合
范围。最新检查点之后按日期保存的段落反映各自提交时仍待完成的工作。

- **Kernel 后端：** B200 上默认使用 CUDA。在 Python 启动前设置
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 则改用冻结的 TileLang 对照。运行时身份报告实际使用的
  后端，不会静默回退到 TileLang。
- **职责划分：** Rust 负责服务、调度和逻辑 KV 缓存；Python 负责 GPU 计算。不使用任何 vLLM
  运行时、源码或环境依赖。
- **当前算子（干净提交 `e3c42e0`，显式选择 CUDA）：** 全部 147 项正式用例的
  输出和冻结 TileLang 对照通过，每项三轮、每轮 20 对交替测量；最小单侧
  95% 正收益下界为 `0.000005722664296627035 ms`。
- **当前框架（同一源码）：** 首轮完整采集通过 11 行；prefix batch 1 因
  23.924% TTFT 离散度被拒，在同 GPU 独立重测以 7.825% 通过。合并后的
  12 个合格行吞吐为冻结基线的 99.43–124.58%，TTFT 为 75.70–96.40%，
  每行均通过 10% 离散度和稳态审计门槛。
- **默认选择变更（`030f60f`）：** 只修改后端选择和身份报告。174 项测试加 31 个子测试的
  套件以及 MTP4 服务约束和生命周期检查，是在 `ef07b2b` 之后、后来成为该提交的工作树上
  通过的（CUDA 源码未变），而不是在 `030f60f` 上单独运行的。
- **其他已通过的检查：**
  - 六组 258048+4096 边界运行完成，无 OOM、无抢占。
  - 真实文本的抢占/前缀检查。
  - 两种 oh-my-pi API。

当前证据见 [acceptance.zh.md](acceptance.zh.md) 和
`bench/baseline/2026-09-29-audit-final-{operators,framework}.json` 汇总。
`2026-09-22-cuda-{operators,framework,features}.json` 仍是历史产物。

## 最新验收检查点（2026-09-29）

- 干净源码 `e3c42e0` 的完整 B200 pytest：**491 项通过、70 个子测试**，
  42 条第三方 warning/编译提示；日志
  `/tmp/oh-my-vllm-final-full-gpu-e3c42e0.log`。
- [完整正式算子汇总](../bench/baseline/2026-09-29-audit-final-operators.json)：
  147/147 通过，UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`，
  无干扰、源码干净；已加载 CUDA 模块 SHA-256 为
  `24bec3a7208ac09629d0a594940266dec2dfbb272c87e02e27c6a8196cd84318`。
  原始 `/tmp/oh-my-vllm-final-formal-e3c42e0.json` 的 SHA-256 为
  `526c99df52ded4cf73630d08ed5f0861a383da5644bc937f55bb3a4725989c33`。
- [框架汇总](../bench/baseline/2026-09-29-audit-final-framework.json)：
  同源码、同 UUID；release 二进制 SHA-256 为
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`。
  首轮 owner `/tmp/oh-my-vllm-final-framework-e3c42e0.log` 产出 12 份
  结构/来源有效的结果，但只验收 11 行：prefix batch 1 TTFT 离散度
  23.924% 而被拒收，故 owner 以状态 1 退出；独立重测
  owner `/tmp/oh-my-vllm-final-framework-prefix1-retry-e3c42e0.log` 以
  7.825% 离散度、状态 0 退出。11+1 个合格行均有两次预热、五次测量、
  实报池容量、干净稳态审计；吞吐至少为基线的 99.426%，TTFT 至多 96.402%。
- 后续 `scripts/test.sh cpu` 通过 266 项测试，排除 225 个 GPU 用例，
  70 个子测试通过（`/tmp/oh-my-vllm-final-cpu-e3c42e0.log`）。Rust
  workspace、fmt、行宽、Ruff format/check 和 Clippy 均通过。无自有
  server/worker/TTFT 进程或服务监听残留；所选 GPU 为 0 MiB/0%，无计算
  进程。已清理本次运行遗留的 13 个无监听 IPC socket 文件。工作树后续只
  为本批已复审文档/证据修改。
- PY-03 的有界 graph 策略在所测 Serve 工作集变形与框架行上关闭。较早干净
  源码的六项 258048+4096 边界用例已完成，但未覆盖 262144 处反复 shape
  churn。物理 <4 GiB 的首次 miss 和后续已有图的 36k→40k 变形诊断均通过，
  但 PY-04 因跨形状 262k 逐出/再捕获仍未实测，继续为
  **partial/open**；PY-06 在 pinned DtoH 未见实质完整调用收益的诊断后，整步
  host/GPU overlap 净收益仍未证实，仍为
  **partial/open**；KRN-06 的安全候选没有稳定完整调用收益，仍为
  **open/no-go**。详见当前[审计索引](audit-2026-09-23.zh.md)。

## 物理低显存余量检查点（2026-09-29）

- 经独立复核的仓库外 shim 在干净 HEAD `8111c19`、B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上运行；release 二进制
  SHA-256 为 `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`。
  CLI FA 容量 7000 映射为实际 FA 2333 块，GDN 容量 128。cache 初始化后
  保留 68 个各不超过 256 MiB 的分配，真实空闲显存从 22,479,568,896 降至
  4,225,957,888 字节（3.936 GiB），贯穿 MTP4 batch 1、输入 32、输出
  32 的 bench。target、draft、proposal 的 `headroom_eager` 分别为
  26/110/28，捕获均为 0。运行完成 32 个输出 token，MTP proposed/accepted
  为 56/18，无抢占。
- 原始 `/tmp/oh-my-vllm-py04-physical/probe-141183.log` 的 SHA-256 为
  `09c2513114c4f2a5f7700023d2d49e07ca236bfbc2a4856a880110b2d810d17a`；
  结果 `/tmp/oh-my-vllm-py04-physical/probe-result-141183.json` 的 SHA-256
  为 `ce0fb9ed75ddad08b5b4647f3a41796eb41c75bbba73263e4ae68a835ecb2a49`。
  owner PID 141183、子 PGID 141218、worker PID 141220 均退出，所选
  GPU 恢复 0 MiB/0%，自有 IPC socket/监听不存在。首轮因真实空闲 20.94 GiB
  超过旧的 20 GiB 前置上限而在分配前拒收。该诊断只证明本配置中低于 4 GiB
  时首次 graph miss 转 eager；不覆盖已有 graph 后的晚期新形状或 262144
  token 处反复变形在此检查点尚未验证。下方较新的已有图诊断缩小了第一个
  缺口；PY-04 仍为 **partial/open**。

## 已有图与重复 262k 检查点（2026-09-29）

- 经独立复核的仓库外探针在干净 HEAD `70a5e5e` 上使用 B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`；release 二进制 SHA-256
  仍为 `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`。
  未修改生产源码。两项均为零次预热的行为诊断，并非正式性能测量。
- FA/GDN 2333/128 的 MTP4 batch-1、36832+256 运行先确认
  target/draft/proposal 的精确 36864 图驻留，40960 图均不存在。保留
  67 个不超过 256 MiB 的分配后，实报空闲显存从 22,055,944,192 降至
  4,070,768,640 字节。三家首次 40960 miss 依次为 draft→proposal→target：
  每次空闲低于 4 GiB，`headroom_eager` 各增 1，旧图仍驻留且捕获不增。
  最终捕获数仍为 2/2/1，无逐出。输出 256 token，MTP proposed/accepted
  为 225/199，零抢占。原始日志
  `/tmp/oh-my-vllm-py04-resident/probe-283459.log` 的 SHA-256 为
  `9f5113d42622635639dbb1e278f0b68fc8c1681e4233939ff115de9ea3a8f847`；
  结果 `/tmp/oh-my-vllm-py04-resident/probe-result-283459.json` 的 SHA-256 为
  `f5fae3f04eb916ce6f358c0a15d74af50f573528edf3559b8925760b3ebbbc7d`。
- 另一 FA/GDN 1400/128 的 MTP4 batch-4 worker 在同进程完成两轮
  258048+4096。每轮输出 16,384 token，零前缀命中、零抢占，MTP
  proposed/accepted 为 16,219/12,318。第一轮捕获 45 个图，第二轮 0 个；
  最终 target/draft/proposal 驻留为 13/28/4，逐出、近期再捕获和 churn
  冷却计数均为 0。因此同形 262144-token 重复运行完成，
  `repeated_churn_verified=false`。原始日志
  `/tmp/oh-my-vllm-py04-262k/probe-308672.log` 的 SHA-256 为
  `10ab638c78e4ac8ca91c45af1d279bba211a20594b717b799efb844fc5514f61`；
  结果 `/tmp/oh-my-vllm-py04-262k/probe-result-308672.json` 的 SHA-256 为
  `1df3e7f27be02e8b5326ce9ca0b32dc40ed171aa193e8f28bbc4d4db08507bea`。
- 两位 owner 退出后均无自有 worker、进程组或 IPC 监听；所选 GPU 回到
  0 MiB 且无计算进程，源码身份保持干净。Rust Bench 在重复运行之间保持
  固定形状。验证 262k 跨形状逐出/再捕获仍需另经复核的有界混合形状 Serve
  会话，精确控制 token 长度并逐波记录 graph key/计数；该路径尚未验证，
  PY-04 继续为 **partial/open**。PY-06、KRN-06 保持此前状态。

## 整步重叠检查点（2026-09-29）

- 干净 HEAD `fa3b8e6` 使用未变的 release 二进制 SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`
  和 B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`。仓库外
  profiler 采集一次已预热的 MTP4 batch 4、输入 32768、输出 256 的
  decode step：CPU step 21.891 ms，1251 个 kernel 总计 13.624 ms，
  GPU 工作合并时长 13.535 ms。target greedy 读回的 host
  `cudaMemcpyAsync` 耗时 15.044 ms，而 320 字节 GPU DtoH 仅约 3.4 微秒。
  target token 是验证、提交及 MTP 的输入，proposal token 是 Rust 下一步
  调度更新的输入。target DtoH 后 4.417 ms 空窗无法由单条 profiler trace
  归因。原始 `/tmp/oh-my-vllm-py06-profile/trace-173909.json` 的 SHA-256
  为 `00767f55e0b04e12886385c703ee0d5d0b7e714addd4beb7b7e0eb5dc694357c`；
  owner 结果 `/tmp/oh-my-vllm-py06-profile/result-173804.json` 的 SHA-256
  为 `6a0bdc5e05a21d42d6c90383b6019f3ddb910a8c5e157e23bec3787980e5473d`。
  包装层 157 ms wall time 包含 profiler 进入/退出开销，不是吞吐计时。
- 经独立复核的仓库外 pinned-D2H shim 通过 12 项聚焦检查：九项 CUDA
  精确输出对照（FP16/BF16/FP32、并列及非有限值）、一项 CPU 回退和两项
  CUDA 无效输入拒绝；
  `/tmp/oh-my-vllm-py06-ab/equivalence.log` 的 SHA-256 为
  `a7283cce1be0067b1ad6527d9855409143a05f34c0fe0217c29b8e65d3a8745a`。
  无 profiler 的 ABBA 对照在该 UUID 上串行运行四个 worker，每个一次预热、
  三次测量，工作负载同为 MTP4 batch 4、32768→256。baseline/pinned
  各六次中位数为 137.031/137.328 token/s（+0.216%），相对各自中位数
  最大偏离为 0.885/0.314%。两组 baseline 自身漂移约 0.287%，超过候选
  收益。每行均有 1024 个输出、69 步、MTP proposed/accepted 961/783、
  零抢占。结果 `/tmp/oh-my-vllm-py06-ab/result-211290.json` 的 SHA-256
  为 `bedccc3e3c2ef52c545755249250810d50dac7a4994a832905e718704e949f03`；
  内含四份原始日志哈希和清理结果。此 1+3 诊断不是正式框架的 2+5 协议，
  也没有 token 序列哈希。本轮未将 pinned DtoH 加入生产。仓库源码未变，
  先前完整 GPU、147 项算子及 12 行框架门槛仍有效。PY-06 维持
  **partial/open**。trace owner PID 173804 及 worker/IPC、ABBA owner
  PID 211290 及四组 worker/IPC、聚焦测试进程均退出；所选 GPU 恢复
  0 MiB/0%。

## 较早修复检查点（2026-09-28）

- `96e4ecc` 修复余下的 P0/P1 请求隔离、RPC 取消、注册和 FA kernel 问题
  （SRV-01/05/06/07 与 KRN-01/02/03）。取消后的 prepare 回复不会移位到下一次
  RPC；失败或被拒的准备会释放 Python 状态。请求局部故障不再停止其他活跃流；
  CUDA/设备故障仍会停止引擎。
- `d33b844` 补完 SRV-03 的错误分类：明确的输入验证错误返回 HTTP 400，未知
  Python prepare 故障返回 500，第 65 个活跃请求返回 503。Axum 的 413/415
  状态码保持原样。定向 bridge、HTTP、serving 测试通过 31 项及 26 个子测试，
  包括畸形 schema 和 sampling 参数。
- 干净提交 `96e4ecc` 的[受影响算子证据](../bench/baseline/2026-09-28-audit-p1-operators.json)
  在 B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过全部 29 项
  选中正式用例（13 项 prepare-attention、16 项 attention）；最小的单侧 95% 正
  收益下界为 0.000796 ms。这是受影响子集，而非新的 147 项完整矩阵结果。
- `96e4ecc` 后的完整 B200 pytest 通过 211 项测试及 32 个子测试，六项 context
  边界占位测试跳过。Rust workspace 测试、格式、行宽、Ruff、Clippy 与提交 hooks
  通过。SRV-03 的聚焦 Rust/Python 测试和静态检查也通过。
- 热路径改动后的当前 12 组框架吞吐和 TTFT **尚未验证**。下文的 2026-09-22
  数值只适用于干净提交 `c36d1c9`。P2 性能决策结束后需重跑全部 12 组。
  EVD-07 在 `bd8e21e` 的干净六行边界与 `8dfc97b` 的真实 MTP HTTP smoke 均已通过。
  审计中原先四项 “Not verified” 风险现均有明确范围的证据：Rust MTP GDN 槽位由
  `7e93c18` 验证，所测 FlashInfer workspace/陈旧尾部路径由 `6b378d5`
  验证，有限 BF16 倒数上界由下文 `32cdc2b` 验证。
- 后续 fixed 状态复核重新打开 SRV-02、KRN-05、PY-01、SCH-01/02/04、
  EVD-01/02/11/12/13 和 MNT-04。已复审的提交 `487f8de` 修复 SRV-02、KRN-05、PY-01、
  SCH-01/02/04 及 EVD-01/02/11。EVD-01 拒绝三次重复或高离散度证据；
  EVD-02 从已分配 FA/GDN 张量报告容量，拒绝 CLI/日志不符；EVD-11 检查完整
  缓存树。聚焦 CPU 回归通过 73 项及 37 个子测试；Rust workspace 测试通过
  56/26/14 项，格式、行宽、Ruff、Clippy 和提交 hooks 均通过。当前 GPU 套件和
  完整框架测量仍待完成。
- EVD-12 由 `f34b661` 加 9648012 修复：记录准确加载的 `.so`
  SHA-256 和 TVM FFI nvcc 路径/版本；绑定源码、参数、SM100a、ABI 的 sidecar
  允许后续进程无法查询 nvcc 时经哈希验证复用。两个正式采集器都会拒绝不完整身份。
  独立源码复审和聚焦 CPU 测试通过。真实 B200 qk 在 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过 6/6 用例，来源记录与
  sidecar 已核验；该运行因工作树未清洁而是诊断，不能算干净的完整算子验收。
  EVD-13 的源码 finding 已由 `895d57b` + `487f8de` 关闭：TVM FFI 缓存已纳入审计，
  且测量窗口内写入回归通过。当前 12 行框架验收仍须独立重跑。
- MNT-04 运行时代码清理由 `ff1f944` 提交（`96e4ecc` 已新增活跃的 Abort 路径）：
  删除未使用的 scheduler 状态和清零提示，协议注释及中英文设计样例与 wire 一致，
  grammar mask 直接使用 draft ID，FA/GDN 容量分别校验，初始化只使用一个期限，
  Ctrl-C/SIGTERM 在共用期限内请求 Python 正常关闭，到期则强制结束。两项真实 HTTP fixture 回归
  分别覆盖活跃 SSE 与停读客户端。聚焦 CPU 通过 46 项测试和 23 个子测试；Rust
  workspace 通过 56/26/17 项，格式、行宽、Ruff、Clippy 与 hooks 通过。
  独立复审通过。三个未调用的冻结 TileLang 包装器因其哈希定义历史对照而有意保留；
  生产和正式 harness 均未调用。此变更尚未运行当前 GPU 套件或 12 行框架测试。
- `6da6427` 通过已复审源码关闭 EVD-03/08/10：`scripts/test.sh` 是有文档说明的
  CPU/full pytest 统一入口，GPU 用例有显式 marker；脚本化 worker 跟踪 FA/GDN
  归属，HTTP 取消回归要求替代请求跨越位置 784、实际占满四块可分配共享 Rust 池并
  完成，以证明真实池容量可复用；正式 fixture 使用旧种子 784 的局部 CUDA generator。
  `scripts/test.sh cpu` 通过 143 项测试和 67 个子测试（排除 112 项 GPU 用例）；
  Rust workspace 通过 56/26/17 项，fmt、行宽、Ruff、Clippy 和 hooks 均清洁。
  B200 聚焦 RNG 测试在 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上
  1/1 通过；日志：`/tmp/oh-my-vllm-evd10-fixture-smoke.log`。自有 GPU 进程已退出。
  当前完整 GPU 套件、正式算子矩阵和 12 行框架采集仍待运行。
- `4ed62ab` 以独立复审的测试关闭 EVD-04/05。自有 CUDA 分页 GQA decode
  现与 CPU FP64 参考比较，覆盖 batch 1/2、784/785、反序物理页表；负向对照
  证明漏读第二页会超出容差。分组 draft 和 proposal graph 除原有一致性检查外，
  也有独立 CPU FP64 参考。B200 聚焦套件在 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上 9/9 通过；自有进程已退出。
  `scripts/test.sh cpu` 通过 143 项测试和 67 个子测试（排除 114 项 GPU 用例）；
  Rust workspace 通过 56/26/17 项，fmt、行宽、Ruff、Clippy 和 hooks 均清洁。
  当前完整 GPU、正式算子矩阵和 12 行验收仍待运行。
- `ea0aa4e` 经独立复审关闭 EVD-09 的源码缺口：每项正式用例在计时前校验两端
  返回值和实际写入的 cache/state 槽，包括精确复制值、FP8 scale stride 和逐槽
  递归状态边界。不匹配会使采集失败；计时 callable 与默认 fixture API 未改变。
  六项对抗性 CPU 比较器用例及六项代表性 B200 fixture 在 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过；无自有 GPU 进程
  残留。`scripts/test.sh cpu` 通过 149 项测试和 67 个子测试（排除 120 项 GPU
  用例）；Rust workspace 通过 56/26/17 项，fmt、行宽、Ruff、Clippy 和 hooks
  通过。干净完整算子矩阵与 12 行框架采集仍待运行。
- `04540bd` 经独立复审实现 MNT-02 列出的直接 CUDA FFI 契约。量化、gates 和递归在启动前
  检查精确 dtype、设备、形状与 stride；融合 SiLU 只接受 BF16，空 gates 不再启动零 grid，
  空量化和递归会拒绝。固定的 `1e-6` RMS epsilon 与唯一受支持 Qwen checkpoint 一致，
  loader 在加载权重前验证。递归索引值仍由调用方保证：`starts` 从零覆盖全部 token 行，
  `reads` 位于池内，`writes` 为 `-1` 或池内。CPU batch plan 检查槽位容量，runner 构造
  `starts`，没有增加 GPU→CPU 索引读回。B200 直接 FFI 测试 45/45，另有聚焦
  `writes=-1` 用例 1/1，通过的 UUID 为
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`；自有 GPU 进程已退出。
  `scripts/test.sh cpu` 通过 153 项测试和 67 个 subtest（排除 166 个 GPU 用例）；Rust
  workspace 通过 56/26/17 项，fmt、行宽、Ruff、Clippy 与 hooks 均通过。干净提交
  `6bd2119` 的首次受影响正式采集在融合 SiLU 大形状处遇到 KRN-04 误拒，之前两项 PASS
  不构成该次验收。已独立复审的 `952a01d` 修正重复计算的 packed FP8 宽度，允许恰好
  `2**31` 个输入元素，并在直接 CUDA FFI 启动前检查同一边界。已独立复审的
  `312c54b` 将相同边界应用到独立 `silu_mul` Python 包装层和直接 CUDA FFI。
  CPU 测试覆盖上限以下、恰好上限、超过上限、正式形状及普通/融合宽度。B200 直接
  FFI 55/55 通过；当前 `scripts/test.sh cpu` 通过 156 项测试和 67 个 subtest
  （排除 175 个 GPU 用例），Rust workspace 56/26/17，fmt、行宽、Ruff、Clippy
  与 hooks 均通过。干净提交 `312c54b` 的
  [受影响正式子集](../bench/baseline/2026-09-28-audit-mnt02-krn04-operators.json)
  在 UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5` 上通过全部 60 项选中的
  `quant`/`silu_quant`/`gates`/`recurrent` 用例。每项均通过输出验证、3 × 20 交替
  配对和单侧 95% bootstrap 正收益下界，最小下界为 0.00000699734 ms。首次干净
  `312c54b` 运行仅通过 59/60；失败的 gates 用例中两个后端计时均偏高，原因尚未证实，
  因此该次运行仅是被拒绝的诊断证据。独立的
  [gates 产物](../bench/baseline/2026-09-28-audit-gates-confirm-operators.json)
  在 UUID `GPU-80cebbaf-a106-2605-b902-f5f9a08645ea` 上通过 13/13，先前失败形状的
  下界为 0.0000465867 ms。自有 GPU 进程已退出。这关闭了 KRN-04/MNT-02 列出的
  源码问题；当前完整 GPU 套件、147 项正式矩阵和 12 行框架采集仍须另行执行。
- 已独立复审的 `0394727` 关闭 MNT-03 的对齐与分派契约。递归按实际 BF16/FP32
  八元素向量对齐检查（16/32 字节），别名区间分析拒绝负 stride。已记录的 RMS 与
  gated-RMS 分派公式在现有 BF16 容差内与 FP64 参考比较，不承诺跨 batch 逐位一致。
  B200 聚焦测试 6/6、CPU 156 项测试和 67 个 subtest（排除 181 项 GPU 用例）、
  Rust workspace 56/26/17、fmt、行宽、Ruff、Clippy 和 hooks 均通过。干净 `0394727`
  的[受影响递归证据](../bench/baseline/2026-09-28-audit-mnt03-recurrent-operators.json)
  在 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过全部 8 项选中用例，
  输出检查与 bootstrap 正收益下界均通过；最小下界为 0.000064624 ms。自有 GPU
  进程已退出。完整 147 项、GPU 套件和 12 行框架验收仍待执行。
- 已独立复审的 `be84176` 关闭 MNT-01 的 Python 调参值被丢弃问题。CUDA 私有工厂
  现在只接收实际生效的设置，冻结 TileLang 签名和公开 API 保持不变。decode 仍传递
  split、首个位置、分组和位置宽度；其 block 计算保留为位置宽度的保守上界，启动前
  检查 split 缓冲区大小。聚焦 CPU mock 套件 9/9 通过，完整 CPU 入口通过 165 项
  测试和 67 个 subtest（排除 181 项 GPU 用例）。在 B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上，公开输出用例通过 25/25，
  增加守卫后的 decode 重跑通过 2/2。Rust workspace 56/26/17、fmt、行宽、Ruff、
  Clippy 和 hooks 均通过；自有 GPU 进程已退出。当前完整 GPU、147 项正式矩阵和
  12 行框架验收仍待执行。
- 已独立复审的 `f161b5e` 将 EVD-07 的六项跳过占位测试改为真实的普通/MTP4 GPU
  batch 1/2/4 用例，并为长上下文 HTTP 脚本增加显式 MTP draft 门槛。CPU 入口通过
  201 项测试和 67 个 subtest（排除 181 项 GPU 用例）；Rust 56/26/17、fmt、行宽、
  Ruff、Clippy 和 hooks 均通过。三次干净源码 GPU 收集均被 GPU 排他门槛拒绝：前两次
  在 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 遇到他人的短 GEMM 作业；
  第三次在 UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5` 的 MTP4 batch 4
  前遇到外部多 GPU 作业。第三次完成的五行仅属诊断，不是六行验收。原始日志保留在
  `/tmp/oh-my-vllm-context-boundary-f161b5e*`，自有 GPU 进程均已退出。随后干净
  提交 `bd8e21e` 的 [六行产物](../bench/baseline/2026-09-28-audit-evd07-context-boundary.json)
  在固定 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过全部普通/MTP4
  batch 1/2/4、258048+4096 边界行。每行生成 `batch_size * 4096` 个输出 token，
  零抢占、无 OOM；MTP4 提议/接受草稿数为 5010/2842、9075/5917、
  16219/12318；worker 最高峰值 reserved 为 132441440256 字节。前后源码、
  release 二进制哈希与每行 UUID 一致，Git 状态干净。原始日志在
  `/tmp/oh-my-vllm-evd07-context-bd8e21e*`；所有自有 GPU worker 均已退出，
  后续 `nvidia-smi` 没有计算进程。随后干净提交 `8dfc97b` 的
  [汇总证据](../bench/baseline/2026-09-28-audit-evd07-long-context-http.json)
  记录 chat/completions 与 responses 各首次/重复两条、四个已校验 strict-JSON
  答案、每请求 131099 prompt token、重复时 130928 个缓存 token、每请求 20 个
  MTP 草稿提议。前后干净源码、二进制 SHA-256 与 UUID 一致；专用服务退出码 0，
  listener/IPC 已关闭，所选 GPU 没有计算进程。日志/来源记录在
  `/tmp/oh-my-vllm-evd07-http*`。EVD-07 的要求证据已修复；当前完整 GPU、
  147 项正式算子和 12 行框架门槛仍待验证。
- 已独立复审的 `5a3b703` 关闭 SCH-07 的全历史重复 SHA-256 计算。只追加的哈希链
  在 262144 token 内与全量重算一致，生成输出填满块后下一请求仍能真实命中前缀。
  100 次 CPU 微基准中，旧路径 42/334 满块的中位数为 98.373/786.007 微秒，
  增量路径为 2.313/2.300 微秒；这是局部 CPU 开销，不是框架结果。Rust workspace
  58/29/17、CPU pytest 201 项和 67 个 subtest（排除 181 项 GPU 用例）、fmt、
  行宽、Ruff、Clippy 和 hooks 均通过。当前 12 行框架验收仍待完成。
- 已独立复审的 `c082937` 关闭 SRV-09 在 serving 线程上的整段 tool-call 重扫。
  解析器跨 chunk 保留各工具调用的字节进度，非字符串 JSON 在参数闭合或按长度结束
  丢弃未完成调用前会校验，最终的 `parse_tool` 校验仍保留。对实际 `Parser::feed`
  路径的字节计数回归覆盖七种
  分块大小，以及畸形和未完成的 XML/JSON。仓库外临时 CPU 基准使用 16 字节分块、
  5 次预热、10 次测量：参数长度为 8,192/32,768 字节时，旧版 `e7e6d42`
  中位数为 151/1,668 微秒，新版为 22/89 微秒。这只是解析器局部开销。
  Rust workspace 58/29/21、CPU pytest 201 项通过及 67 个 subtest（排除
  181 项 GPU 用例）、fmt、行宽、Ruff、Clippy 和 hooks 均通过。当前 12 行
  框架验收仍待完成。
- 已独立复审的 `67deb85` 关闭 SCH-06 在池容量不足时的优先级倒置。旧版 8-block
  回归抢占较早的请求 1；新版抢占较晚的请求 2 并重试请求 1。独立池 MTP4 用例
  抢占 `[3, 2]`，保留等待顺序 `[2, 3, 4]`、已接受历史并清空 draft。测试还覆盖
  自抢占兜底、同一步抢占和重新接纳输出；CPU fake Python worker 测试证明先清状态
  再规划，但不声称已覆盖生产 block 大小下的完整重新接纳执行。中英文需求和架构
  已同步实现后的优先级策略。Rust workspace 58/32/21、CPU pytest 202 项通过及
  67 个 subtest（排除 181 项 GPU 用例）、fmt、行宽、Ruff、Clippy 和 hooks 均通过。
  该策略修复不声称局部速度提升；当前 12 行框架验收仍待完成。
- 已独立复审的 `b1761e9` 关闭 SRV-10 按步数丢失 SSE 的问题。256 事件
  channel 满时只暂停对应请求并启动 30 秒宽限；1,024 事件/8 MiB 的 FIFO 保留待发
  及最终事件。待发队列清空且 channel 至少空出 128 个槽位后恢复生成。真实 CPU
  HTTP 测试停读后观察到 worker 超过 256 步并在 750 步之前稳定暂停，停读期间
  第二请求完成，恢复后按序收到 1–750 的全部内容序号及 `[DONE]`。Rust 测试覆盖
  宽限计时、低水位、容量、断连、完成态排空和独立排空任务退出。在该检查点，单次在途
  prepare/execute RPC 可按自身超时延后 30 秒检查。`efdef09` 后，后台 CPU
  准备不再阻塞该检查，但 execute 往返或阻塞的 prepare/Abort 发送仍可能延迟。
  Rust workspace 58/38/26、`scripts/test.sh cpu` 203 项通过及 67 个 subtest
  （排除 181 项 GPU 用例），fmt、行宽、Ruff、Clippy 和 hooks 均通过。当前完整
  GPU 与 12 行框架门槛仍待验证。
- 已独立复审的 `f4aacca` 通过线程安全的进程内累计计数、附带复制字节数的 2 的幂次
  限频警告，关闭 KRN-07 未对齐 FA cache 克隆不可见的问题。保留未对齐 BF16 view
  支持，克隆仍在，不声称提速。修改前在 B200 UUID
  `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5` 上对 4,383,375,360 字节
  clone 实测中位数 2.736096 ms（5 次预热、10 次测量），诊断日志为
  `/tmp/oh-my-vllm-krn07-clone-baseline.log`。同一 UUID 上四项聚焦 B200 用例
  通过，覆盖对齐/未对齐数值输出、累计次数及警告限频；无自有 GPU 进程残留。
  CPU pytest 通过 203 项测试及 67 个 subtest（排除 182 项 GPU 用例）；Rust
  workspace 58/38/26，fmt、行宽、Ruff、Clippy 和 hooks 均通过。当前完整 GPU/
  正式算子及 12 行框架门槛仍待验证。
- 已独立复审的 `40e3e57` 关闭 KRN-08 的 CUDA 快/通用分派选择不可观测问题。
  线程安全的主机提交计数器覆盖 `norm`、`add_norm`、`gated_norm`、`qk`、
  `recurrent`、`append`、`convolution`；正式采集器在输出验证阶段要求矩阵内
  六种操作各发生快路径一次、通用路径零次。聚焦 B200 测试覆盖全部七组的两个
  变体。干净[受影响正式证据](../bench/baseline/2026-09-28-audit-krn08-operators.json)
  在 UUID `GPU-1b174534-ebba-826f-b452-8e7f3c05c301` 上通过 66/66 项选中
  用例（每项 3 × 20 对交替测量）；每行记录快路径=1、通用路径=0，且单侧 95%
  收益下界为正。最小下界为 0.000000213335 ms；三轮最大 spread/median 为
  1.695%。`scripts/test.sh cpu` 通过 210 项及 67 个 subtest（排除 183 项 GPU
  用例）；Rust workspace、fmt、行宽、Ruff、Clippy 和 hooks 均通过。自有 GPU
  进程已退出。这是受影响子集，不是完整 147 项结果或框架提速实测；完整 GPU
  套件和当前 12 行框架门槛仍待验证。
- 已独立复审的 `94c3473` 关闭 KRN-09 消费式 CUDA 启动检查，并增加仅限 eager
  的故障诊断。自有 kernel 的 19 处启动检查改用 `cudaPeekAtLastError`；显式
  `OH_MY_VLLM_CUDA_DEBUG_SYNC=1` 要求 `OH_MY_VLLM_ENFORCE_EAGER=1`，在每次
  启动前后检查同一 stream，并先拒绝 graph capture。诊断标记观察阶段，不声称
  精确的故障指令。七项独立 B200 用例在 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过，另外五次先前故障
  子进程重复通过。CPU pytest 通过 211 项及 67 个 subtest（排除 190 项 GPU
  用例）；Rust workspace、fmt、行宽、Ruff、Clippy 与 hooks 通过。自有 GPU
  进程已退出。同卡旧/新 eager RMS 主机调用的三轮诊断中位数接近，但孤立离群
  超过 10% spread 门槛，原因未证实，故不声称提速。日志和原始样本保留在 Git
  仓库外的 `/tmp/oh-my-vllm-krn09-*`。当前完整 GPU/正式算子及 12 行框架
  门槛仍待验证。
- 已独立复审的 `37f8cc9` 实现 KRN-10：对每个 kernel 特化仅设置一次
  `cudaFuncSetAttribute`，每个线程对每个 BK tile 缓存一到两个 FA 页表项。
  适用范围是进程内单 B200 CUDA context。修改前干净提交 `6ba9046` 的
  [before 汇总](../bench/baseline/2026-09-28-audit-krn10-attention-before.json)
  在 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上通过 16/16 项
  选中的 attention 用例。干净提交 `231f066` 的
  [after 汇总](../bench/baseline/2026-09-28-audit-krn10-attention-after.json)
  在同一 UUID 上通过相同 16/16 项：源码干净、无干扰、全部输出与冻结 TileLang
  对照通过，最小单侧 95% 收益下界 +0.0028209709 ms，三轮最大 spread/median
  为 0.173%。`selected_passed=true`；顶层 `passed=false` 表示其余 131 项未选。
  跨两个源码提交，全部 16 项 CUDA 三轮中位数降低，中位相对差异 7.26%；
  这不能隔离个别改动或证明框架提速。修改后聚焦 B200 attention 测试另通过
  原有 29 项及四项新的 direct-FFI 边界、Position、分组、graph 用例。同卡 eager
  FFI A/B 的旧/新每次调用中位数为 3.4138/3.0865 微秒，spread 为 3.51%/5.33%，
  仅属主机诊断。此前三次完整修改后 16 项正式尝试因外部 GPU 进程（PID 849461、
  897317、913070）而拒收，均不算验收。`scripts/test.sh cpu` 通过 211 项及
  67 个 subtest（排除 194 项 GPU 用例），Rust workspace、fmt、行宽、Ruff、
  Clippy 与 hooks 也通过。干净 after16 后自有 GPU 进程已退出。受影响子集
  已验证；最终验收仍待当前完整 147 项/GPU 套件及 12 组框架门槛。
- 已独立复审的 `17413e5` 是 PY-05 候选实现：MTP eager/draft/proposal
  元数据按当前 extent 定宽；target decode 保持原有 extent 宽度，同时改为每个
  请求上传一行页表。设备端 `index_select` 展开 token 行。target/draft graph
  replay 仍复制变更后的页表；回归覆盖位置 783/784/262143 和请求重新分组。
  在最大上下文 262144、extent 36864、四个各 42 页且每请求五个 token 行时，
  draft 表由 20×335 缩到 20×48。同卡 host 组件 A/B（5 次预热、20 组各
  256 次调用）的中位数（微秒）为 draft graph 259.69→32.13、eager
  265.86→32.22、proposal 57.39→15.52、target 46.51→32.06。部分 draft
  样本的离散度超过 10%，这些只是组件诊断计时，不是框架吞吐证据。原始记录在
  `/tmp/oh-my-vllm-py05-*`。`scripts/test.sh cpu` 通过 220 项及 67 个
  subtest（排除 196 项 GPU 用例），聚焦 B200 在 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上 5/5 通过；Rust
  workspace 58/38/26、fmt、行宽、Ruff、Clippy 与 hooks 通过。CPU 与 B200
  日志分别为 `/tmp/oh-my-vllm-py05-cpu-17413e5.log` 和
  `/tmp/oh-my-vllm-py05-gpu-focused-17413e5.log`。自有 GPU
  进程已退出。当前完整 GPU 套件及 12 行框架测试，尤其 MTP batch 4，
  仍待完成，因此 PY-05 保持 open。
- 已独立复审的 `503ea27` 是 PY-07 候选实现：sampler 在注册时一次性于 GPU
  计算 prompt 计数，生成计数仅在 commit 时更新，speculative drafts 使用临时
  计数副本。`top_k=50` 加 `top_p=0.9` 时，第 k 个分数不并列即可局部稳定
  排序；并列则回退完整稳定排序。边界判断增加一次 CUDA host 同步。在同一张
  空闲 B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上，
  262144 prompt token、248320 词表的诊断交替 A/B（5 次预热、20 组各 8
  次调用）中位数（ms）为 penalty 18.647→0.132、top-k/p 0.451→0.361、
  两者同时启用 18.870→0.418；第二轮离散度均低于 10%。一次性注册在 32768
  prompt token 时由约 0.063→2.314 ms，在 262144 时由 0.662→18.520 ms，
  其中 32768 旧版和 262144 候选样本的离散度分别为 10.65% 与 15.56%。
  这些是 dirty-tree 组件观察，不是框架门槛。日志：
  `/tmp/oh-my-vllm-py07-ab-gpu-dirty-r2.log`、
  `/tmp/oh-my-vllm-py07-register-gpu-dirty.log`、
  `/tmp/oh-my-vllm-py07-cpu-503ea27.log` 和
  `/tmp/oh-my-vllm-py07-gpu-focused-503ea27.log`。CPU 通过 226 项和
  67 个 subtest（排除 199 项 GPU 用例），B200 sampler 3/3、Rust workspace
  58/38/26、fmt、行宽、Ruff、Clippy 与 hooks 通过。自有 GPU 进程已退出。
  当前完整 GPU 套件及 12 行框架测试证明吞吐/TTFT 仍达标之前，PY-07 保持 open。
- 已独立复审的 `efdef09` 是 SRV-08 候选实现。Rust 让一个在途 prepare 与
  execute step 并行，缓存乱序 Prepared 回复，并对意外/重复协议回复停止引擎。
  Python 在一个 daemon CPU 线程中构造模板/分词/grammar 输入；DEALER 主线程
  由 pipe 唤醒且独自安装活跃请求状态。真实 bridge 与隔离 HTTP 测试覆盖无后续
  RPC 时的唤醒、活跃 SSE 推进、两种回复顺序、超时/迟到回复、取消、register
  交错、错误类型/重复回复和有界 shutdown。Qwen tokenizer/XGrammar 的 CPU
  并发及跨线程 matcher 移交回归通过。对同一当前 Rust binary 上的两种 CPU fixture
  各运行一次预热和五次实测，活跃 SSE 最大事件间隙中位数由历史同步 `f194127`
  fixture 的 0.781507 秒降到新延迟回复 fixture 的 0.081265 秒；对应
  spread/median 为 0.047% 和 0.157%。临时脚本及原始日志：
  `/tmp/oh-my-vllm-srv08-ab.py` 和 `/tmp/oh-my-vllm-srv08-ab-efdef09.log`。
  `scripts/test.sh cpu` 通过 236 项、70 个
  subtest（排除 199 项 GPU 用例），日志为
  `/tmp/oh-my-vllm-srv08-cpu-final.log`。Rust workspace 58/38/26、fmt、
  行宽、Ruff、Clippy 与 hooks 均通过；自有 CPU server/worker 已退出。
  CPU fixture 数值不代表框架吞吐。主线程 sampler 注册仍包含 PY-07 的 GPU
  prompt-count 工作；正式 B200 活跃流时延、当前完整 GPU 套件和 12 行框架门槛仍待
  验收，因此 SRV-08 保持 open。
- HEAD 为 `ed8c18c` 时的真实模型 B200 SRV-08 诊断在 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上运行：第一条请求产生
  2051 个 SSE data event，其中第二条 3016 输入/16 输出 token 请求在
  0.582 秒完成期间仍有 20 个。日志为
  `/tmp/oh-my-vllm-srv08-b200-smoke.log` 和
  `/tmp/oh-my-vllm-srv08-b200-service.log`；服务退出且无自有 GPU 进程。
  这只是单次诊断，不是 12 行门槛。
- 经独立复审的 `7e93c18` 关闭狭义 Rust MTP GDN 槽位风险。真实调度输出覆盖
  783/784/785/1568-token prompt 及分块 prefill；CPU HTTP/ZMQ fixture
  将真实 Rust execute 帧交给 Python `plan_request`/`commit`，核对
  783→784 的 source、五个候选写入槽和 checkpoint copy。
  `scripts/test.sh cpu` 通过 237 项、70 个 subtest（排除 199 项 GPU 用例），
  日志为 `/tmp/oh-my-vllm-mtp-slot-cpu.log`；Rust workspace 58/39/26、
  fmt、行宽、Ruff、Clippy 与 hooks 均通过。这不代表 GPU 数值或当前 MTP
  框架验收通过。
- 经独立复审的 `6b378d5` 在 FlashInfer 0.6.18.post1、B200 的真实
  TRT-LLM gen decode 和 context prefill 路径上验证两项 FlashInfer 风险。
  每个零值或 `0xA5` workspace 都在独立新 plan 首次调用真实后端之前写入，
  并与零值/128 陈旧 KV 尾部交叉；后端 spy 每例观察到八次真实调用。decode
  覆盖九个短长度及页边界、batch 4 的 32769、五个分组 draft 的 785、
  单条 131073；prefill 覆盖 4097/4237/4703。15 项聚焦 GPU 用例全部通过，
  污染对照输出以零容差数值一致，小型 decode 用例还与 CPU FP64 参考吻合。UUID 为
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`，原始日志为
  `/tmp/oh-my-vllm-flashinfer-poison-tests-r2.log`。`scripts/test.sh cpu`
  通过 237 项、70 个 subtest（排除 214 项 GPU 用例），日志为
  `/tmp/oh-my-vllm-flashinfer-cpu.log`；Rust workspace 58/39/26、fmt、
  行宽、Ruff、Clippy 与 hooks 均通过。自有 GPU 和 CPU worker 已退出。
  FlashInfer 文档要求的清零针对生产未调用的 XQA。本结论不覆盖 XQA、FA2
  或全部 262144-token 布局；当前完整 GPU 与框架门槛仍待完成。
- 经独立复审的 `32cdc2b` 根据 NVIDIA PTX ISA 9.4 和源码输入范围验证
  有限 BF16 倒数前提。ISA 保证 `rcp.approx.f32` 误差最多 1 ulp；有限
  BF16 scale/倒数均为 FP32 normal，因此 `.ftz` 不改变它们。非有限
  最大值和 FP16/FP32 使用精确除法。提交 `32cdc2b` 时的 B200 回归将生产
  CUDA FP8 字节及 scale 与 CPU IEEE FP32 RN 参考比较，覆盖 65,280 个有限
  BF16 编码、32,640 层有限最大值的正负中点派生样本、最大值为 448 时全部
  126 个正负精确 FP8 中点及非有限回退。聚焦 3/3 通过，UUID 为
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`，日志为
  `/tmp/oh-my-vllm-rcp-focused-32cdc2b.log`。仅改注释的源码修正之前，
  测试提交 `c0ff543` 的 CPU 套件通过 237 项和 70 个 subtest
  （排除 217 项 GPU 用例），日志为
  `/tmp/oh-my-vllm-rcp-cpu-final.log`；Rust workspace 58/39/26、fmt、
  行宽、Ruff、Clippy 和 hooks 均通过，自有进程全部退出。该有界硬件
  测试并未穷尽全部最大值/输入配对，也不证明中间 FP32 商逐位一致。
  当前完整 GPU 套件、147 项正式算子矩阵、六项长上下文和 12 行框架门槛仍待完成。
- 经独立复审的 `b464634` 是 PY-04 共享池候选。target、draft、proposal 图
  分别在本类图之间共享独立的池；draft hidden 在下一图回放前沿同一 stream
  复制到其持久输入。源码提交 `b464634` 在 B200 上的图测试8/8通过，UUID 为
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`，日志为
  `/tmp/oh-my-vllm-py04-gpu-b464634.log`。同 UUID 的 dirty-candidate
  配对诊断使用 HEAD `a02812d`、相同五文件 patch SHA-256
  `7c531868cda9e65590ecdf08e1ef467f51c4195ea8b6c371e92fa36db752b8fe`
  和 release binary SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`；
  两份日志都记录各文件 hash：`/tmp/oh-my-vllm-py04-{private,shared}-r2.log`。
  FA 1400/GDN 128 块、MTP4 batch 4、输入32768/输出512，两次预热与五次
  测量后，私有池→共享池 reserved 显存峰值由126565220352降至
  125986406400字节（减少578813952字节，约552 MiB）。吞吐中位数为
  243.936→244.011 token/s，最大 TTFT 为6.55794→6.55717 s；双方离散度
  均低于0.55%，每次运行均无抢占并接受1562个 draft。另一次**干净
  `b464634`**运行采用 FA 2333/GDN 128，跨 extent 36864→40960 捕获新的
  target/draft/proposal 形状，在 reserved 峰值176815079424字节时无 OOM
  或抢占；一次预热、一次测量的日志为
  `/tmp/oh-my-vllm-py04-late-b464634.log`。候选工作树的
  `scripts/test.sh cpu` 通过237项测试及70个 subtest（排除220项 GPU 用例），
  日志为 `/tmp/oh-my-vllm-py04-cpu.log`；Rust workspace 58/39/26、fmt、
  行宽、Ruff、Clippy 与 hooks 通过。自有 GPU/worker 进程和 socket 已清理。
  PY-04 仍为 **open**：缓存仍在捕获前分配，单个近满形状不能约束所有后期
  捕获；当前完整 GPU、147项正式算子、六项长上下文和12行框架门槛待完成。
- 经独立复审的 `5772834` 是 PY-03 有界图接纳**候选**，不等于性能验收。
  target 独占 32 槽；draft/proposal 共享 32 槽，逐出下限为 16/4。
  新形状要观察四次且衰减访问次数超过最冷可逐出图的两倍才能替换。
  元数据有界；捕获失败保留旧图，首次失败会丢弃无 owner 的本类池句柄。
  CUDA 空闲显存低于 4 GiB 时，新捕获改走 eager；CUDA OOM 仍为致命故障。
  干净 `5772834` 在 B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 的真实图聚焦测试通过 3/3，
  包括非空捕获失败后换新句柄重试；日志为
  `/tmp/oh-my-vllm-py03-clean-focused-5772834.log`。候选
  `scripts/test.sh cpu` 通过 255 项测试及 70 个 subtest（排除 223 项 GPU），
  日志为 `/tmp/oh-my-vllm-py03-cpu-full-candidate.log`；Rust workspace、fmt、
  行宽、Ruff、Clippy 和 hooks 均通过。同一干净源码下，仓库外先到先占 shim
  与 LFU 在 FA 1400/GDN 128、MTP4 batch 4、输入 36832/输出 512、两次
  预热五次测量后，吞吐中位数为 213.797→214.324 token/s（离散度
  0.324/0.361%），最大 TTFT 为 7.50258→7.50947 s（离散度
  0.390/0.693%）；不声称稳定提速。draft/proposal 的预算 eager 次数
  182/35→63/20，捕获 25/7→30/8，逐出 3/3，近期重捕获为零。日志：
  `/tmp/oh-my-vllm-py03-{oldmode,lfu}-clean-5772834.log`。早前精确 LRU
  的 dirty-candidate 诊断只有 208.301 token/s，已拒绝。全模型 FA 2333/
  GDN 128 诊断在 179080003584 字节 reserved 峰值时驻留 32 个 MTP 图，
  但捕获前最低空闲显存仍为 11524046848 字节（10.73 GiB），未实际触发
  4 GiB 门槛；日志：
  `/tmp/oh-my-vllm-py03-nearfull-final-candidate.log`。自有 GPU 进程均已退出。
  PY-03 保持 **open/candidate**，待 shape churn、262k、低余量及 12 行
  框架证据。
- 经复审的 `2837c49` 在 Serve 与 Bench 的 MTP 图缓存中加入共用捕获冷却：
  draft/proposal 形状近期重捕获后，已驻留图继续命中，非驻留形状在 32768 次
  决策内走 eager，到期恢复接纳。CPU 回归覆盖边界、两类图的逐出保底和
  热点工作集恢复。`scripts/test.sh cpu` 通过 266 项测试、70 个 subtest
  （排除 225 项 GPU），日志为 `/tmp/oh-my-vllm-py03-churn-cpu-full.log`；
  B200 聚焦 GraphCache 测试在
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 通过 4/4，日志为
  `/tmp/oh-my-vllm-py03-churn-graph-gpu4.log`。相同源码/测试提交前 Rust
  workspace、fmt、行宽、Ruff、Clippy 通过，提交时 hooks 通过；独立最终
  源码与 Serve 证据复审无正确性阻塞。
  dirty 候选 MTP batch4 单行通过本地捕获与比较审计，吞吐为基线的
  99.60%，最大 TTFT 为基线的 96.09%，原始记录为
  `/tmp/oh-my-vllm-py03-churn-bs4-candidate.json`。同一 B200 的真实
  Serve A/B 对照干净 `ed63177`，双方各完成 12 波/36 请求/36864 输出
  token，响应 hash 一致；候选与旧版的 draft/proposal 捕获为 31/10
  对 40/11。去掉首个冷启动波，逐波最大耗时之和为 33.409 对
  33.483 s；单组诊断不能证明稳定性能改善。A/B 原版 owner 为
  `/tmp/oh-my-vllm-py03-serve-ab-original.py`（SHA-256
  `7d23377d282b0323d6eaf5ecde99c5d95a8793f35e25a0e2613a0757a41ca971`），
  原始 artifact/日志为 `/tmp/oh-my-vllm-py03-serve-{candidate,old}-shift.*`。
  独立候选 Serve 长跑完成 88 波/264 请求/270336 输出 token，观察到两次
  冷却、到期后新 draft/proposal 形状捕获，退出时两类图驻留 20/12；记录为
  `/tmp/oh-my-vllm-py03-serve-candidate-recovery.*`。自有 worker、GPU
  进程与 IPC listener 均已退出。Serve 输入最高约 21k token、输出 1024，
  长跑没有旧版对照。PY-03 仍为 **open/candidate**，待干净最终 12 行
  门槛；262k 与物理低显存余量捕获是另两个证据限制。
- 经独立复审的 `3bbf926` 是 PY-06 非 greedy 读回**候选**。runner 先逐请求
  在 GPU 上抽样，再合并行做一次最终主机传输，之后逐请求验证和提交；同批重复
  request ID 在前向或 RNG 抽样前即被拒绝。纯 greedy 路径未变。干净
  `3bbf926` 在 B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` 上，以四请求×每请求
  五行×248320 词表、temperature 1、无 penalty/grammar、五次预热加二十次
  成对测量，逐请求读回中位数 0.740048 ms，合并后 0.674659 ms（离散度
  3.282/2.047%）。原始日志：`/tmp/oh-my-vllm-py06-clean-ab-3bbf926.log`。
  64–32768 个 int64 元素的 pinned H2D 暂存没有稳定显著收益，因此未修改；
  原始日志为 `/tmp/oh-my-vllm-py06-h2d.log`。MTP eager 回退和依赖输出的
  下一步调度仍串行，top-k/top-p 的并列阈值检查也可能同步。B200 聚焦测试
  通过 32/32，日志为 `/tmp/oh-my-vllm-py06-gpu-focused-r2.log`；
  `scripts/test.sh cpu` 通过 259 项测试和 70 个 subtest（排除 224 项 GPU），
  日志为 `/tmp/oh-my-vllm-py06-cpu-full.log`。Rust workspace、fmt、行宽、
  Ruff、Clippy 和 hooks 均通过。自有 GPU 进程已退出。当前完整 GPU 套件、
  12 行框架门槛及代表性整步重叠证据完成前，PY-06 保持 **open/candidate**。
- KRN-06 已调查，但建议的优化为 **open/no-go**；原流式 store 的
  `"memory"` clobber 保留。CUDA 13.1 的 `__stcs(uint2*)` 自身仍带该
  编译器 clobber，因此原审计建议的替换无效。经复审的去 clobber 试验要
  保证安全，还须加入 shape/dtype/layout/device 与输出不重叠的 FFI guard。
  试验源码在 B200 的直接 FFI 拒绝 7/7、BF16 精确/CPU FP64 4/4 均通过，
  但这些是**被拒候选的测试**，不属于保留源码。干净 `2c85dfa` 的
  norm/add_norm before 正式用例通过 25/26；受影响 2496 行通过，未受
  影响的 1248 行 add_norm 因接近零的下界失败。dirty 候选也为 25/26，
  同一用例失败；1248 行同卡隔离 3×20 重测为 PASS，显示跨运行波动。
  原始报告：`/tmp/oh-my-vllm-krn06-before-2c85dfa.json`、
  `/tmp/oh-my-vllm-krn06-after-dirty.json` 和
  `/tmp/oh-my-vllm-krn06-1248-repeat-dirty.log`。同卡、单 CPU 的 5+100
  交错旧版/无 guard/有 guard 诊断中，直接 FFI 总耗时中位数为
  21.980/21.971/22.302 微秒，含输出分配的调用为
  33.070/33.143/33.528 微秒；逐次离散度超过 10%，不声称稳定完整调用
  提速。日志：`/tmp/oh-my-vllm-krn06-host-trio-dirty.log` 与
  `/tmp/oh-my-vllm-krn06-wrapper-trio-dirty.log`。补丁已回退，没有提交
  源码/测试；自有 B200 进程均已退出。

在此 2026-09-28 检查点，PY-03/04/06、SRV-08、PY-05/07、KRN-10 及完整
算子/GPU/12 行门槛仍待完成。上方的 2026-09-29 检查点更新了该队列。
KRN-06 仍为 open/no-go，除非另一安全优化证明完整调用收益。
逐项状态与限制见[当前审计索引](audit-2026-09-23.zh.md)。

## 历史 2026-09-24 修复检查点

对 `030f60f` 的审计记录在 [audit-2026-09-23.zh.md](audit-2026-09-23.zh.md)，共 51 项。
修复工作于 2026-09-24 开始，进行中。

### 已完成批次

**批次 1 — 服务可用性**（`d0d286a`，2026-09-24）：SRV-01/02/03/04/06 已修复。
- SRV-01：`request.rs` 和 `sampling.py` 都将 temperature < 1e-5 夹到 0。
- SRV-02：子进程 pre-exec 调用 `prctl(PR_SET_PDEATHSIG, SIGKILL)`；Python 监控父进程 PID，
  父进程消失时退出；`sock.recv` 设置 5 秒超时。
- SRV-03：HTTP ready channel 携带 `(StatusCode, String)`，非 200 错误正常传播。
- SRV-04：`Parser::feed` 在 `LengthFinish` 时冲刷暂存字节；已新增测试。
- SRV-06：`register_request`/`abort_request` 降级为 fire-and-forget。

**批次 2 — 证据完整性**（`895d57b`，2026-09-24）：EVD-06/11/12/13 已修复，
EVD-01/02 仅部分处理。
- EVD-01：`compare_vllm.py` 强制 reps=5、warmup=2；高离散度只发出警告，
  因此离散度门槛仍未修复。
- EVD-02：`ttft.py` 解析了部分实际 BENCH_CONFIG 字段；FA/GDN pool 容量
  仍来自 CLI 算术，且缺少错配回归。
- EVD-02：`ttft.py` 通过 `_parse_bench_config()` 读取 worker 输出的 BENCH_CONFIG 日志行；
  `zmq-worker/src/main.rs` 在启动时输出 `info!("BENCH_CONFIG", …)`。
- EVD-06：`test_kernel_reference.py` 断言冻结参考 JSON 中的精确文件集合。
- EVD-07：`tests/test_context_boundary.py` 新增六个 REQ-CONTEXT-001 边界运行的
  跳过占位测试；这没有关闭该问题。
- EVD-11：`measurement.py` 在缓存根目录缺失时追加到 `problems`。
- EVD-12：`cuda_backend/__init__.py` 新增 `provenance()`，返回 nvcc 版本和编译后
  `.so` 的 SHA-256。
- EVD-13：稳态审计将 `TVM_FFI_CACHE_DIR` 作为第四个缓存根目录。

**批次 3 — Kernel/worker 契约（宿主端）**（`eb5ff62`，2026-09-24）：
  KRN-01/04/05、PY-01/02 已修复；MNT-02 仅部分处理。在这个检查点，KRN-02/03
  涉及设备代码，需 GPU 重跑，因此延后。
- KRN-01：`decode_attention.py` 在提供 `starts` 时断言 `group_size ≤ 5`。
- KRN-04：`elementwise.py` 和 `fp8.py` 防止 int32 溢出。
- KRN-05：`gdn.py` 将 dtype/shape/stride 校验移入 `normalize_qk`。
- PY-01：`batch_plan.py` 在 debug 模式断言可写 FA 尾页跨请求不相交。
- PY-02：`qwen.py` 对每个 FP8 缩放投影断言 `w.shape[0] % 128 == 0`。
- MNT-02：`elementwise.py` 在 `silu_mul`/`delta_gates` 对空张量提前返回；兜底
  dtype 分派和硬编码 epsilon 仍未修复。
- 新增测试：`tests/test_validation_guards.py`、`tests/test_batch_plan.py`、
  `tests/test_independent_decode_attention.py`。

**批次 4 — 调度器加固**（`42197f7`，2026-09-24）：SCH-01/02/03/04/05 已修复。
- SCH-01：`update()` 在修改状态前校验 token 数量和 `is_final_chunk` 标志。
- SCH-02：`pool.rs` 中引用计数/空闲列表不变量从 `debug_assert!` 改为 `assert!`。
- SCH-03：`aligned_prefill` 在 `count > 0` 时将结果夹到至少 1。
- SCH-04：`add_request()` 返回 `bool`，在入队时拒绝超容量请求（服务层返回 HTTP 413）。
- SCH-05：`remove_blocks_in_range` 遇到 NULL_BLOCK_ID 时改用 `continue` 而非 `break`。
- 在这个检查点，SRV-05/07 仍为 open；`96e4ecc` 后来修复了它们。

**批次 6 部分 — 清理**（`62101a3`，2026-09-24）：当时曾把 MNT-04 标为已修复，
后续复核重新打开。该提交删除未使用的 `abort_request()` 方法，并更新了部分协议注释；
`96e4ecc` 和 `ff1f944` 才完成活跃 Abort 路径与余下清理。

### 该检查点仍未修复的问题

以下问题已延后（设备代码需 GPU 重跑）或归入 P2/P3 待后续处理：

- **KRN-02/03**（P1）：设备代码修复需在 B200 上重跑正式算子用例。
- **SRV-05/07**（P1）：RPC 取消安全性与 `serving_outputs` 侧信道。
- **P2 性能问题：** PY-03/04/05/06/07、SRV-08/09/10、SCH-06/07、KRN-06/07/08/09/10。
  每项需先测量再决定是否保留。
- **P3 清理/测试问题：** EVD-03/04/05/08/09/10、MNT-01/03。

## 文档清理（本次变更）

- 新增审计文档。
- 从 `acceptance.md`、`cuda-development.md`、`plan.md` 和 `bench/baseline/README.md` 删除了
  已被取代的里程碑叙述。原始产物仍然保留，并由 `acceptance.md` 中的历史表索引。
- 删除了 `performance-gap-2026-09-22.md`。它描述的 MTP 差距已被后续工作消除，汇总数据仍保留在
  `bench/baseline/2026-09-22-performance-gap.json`。
- 修正了过时内容：
  - 命令改为 `ttft.py` 和完整的 pytest 套件。
  - 协议和抢占的描述与当前代码一致。
  - ADR 状态已更新。
- 同步更新了中文译本。

## 操作规则

- **环境：** 使用 `scripts/with-env.sh`。
- **GPU 选择：** 用 `scripts/with-gpu.sh` 按 UUID 选择空闲 GPU。
- **Git：**
  - 只在 `main` 上开发和提交。
  - 只暂存有意修改的文件。
  - 保留 hooks。
  - 维护中文 Markdown 译本。
  - 带上要求的提交署名。
- **GPU 清理：** 及时停止自己启动的所有 GPU 程序，并确认端口和显存已释放。
- **不进仓库的内容：** 临时调优和原始 GPU trace。正式的汇总证据放在 `bench/baseline`。
