# 交接记录 — 2026-09-28

[English](handoff.md)。agent 以英文原文为准。

## 当前状态

CUDA 迁移的原始验收属于历史结果。整库审计修复仍在进行。`96e4ecc` 已有受影响算子
证据；后续 fixed 状态复核又发现 P0/P1 和证据缺口，已复审的源码提交 `487f8de`
修复了这些问题。
当前 12 行门槛尚未重新测量。

- **Kernel 后端：** B200 上默认使用 CUDA。在 Python 启动前设置
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 则改用冻结的 TileLang 对照。运行时身份报告实际使用的
  后端，不会静默回退到 TileLang。
- **职责划分：** Rust 负责服务、调度和逻辑 KV 缓存；Python 负责 GPU 计算。不使用任何 vLLM
  运行时、源码或环境依赖。
- **算子（干净提交 `c36d1c9`，显式选择 CUDA）：** 全部 147 项正式用例通过。每项都通过三轮、
  每轮 20 对交替测量，且单侧 95% bootstrap 下界为正。
- **框架（同一提交）：** 全部 12 组通过。
  - 吞吐为冻结 vLLM 基线的 97.52–125.09%。
  - TTFT 为基线的 76.01–97.50%。
  - 每组都满足 10% 极差规则。
  - 余量最小的是 MTP batch 4，吞吐 97.52%。
- **默认选择变更（`030f60f`）：** 只修改后端选择和身份报告。174 项测试加 31 个子测试的
  套件以及 MTP4 服务约束和生命周期检查，是在 `ef07b2b` 之后、后来成为该提交的工作树上
  通过的（CUDA 源码未变），而不是在 `030f60f` 上单独运行的。
- **其他已通过的检查：**
  - 六组 258048+4096 边界运行完成，无 OOM、无抢占。
  - 真实文本的抢占/前缀检查。
  - 两种 oh-my-pi API。

证据见 [acceptance.zh.md](acceptance.zh.md) 以及
`bench/baseline/2026-09-22-cuda-{operators,framework,features}.json`。

## 最新修复检查点（2026-09-28）

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
  EVD-07 仍为 open，等待完整、无干扰的六行运行；六项测试已不再跳过。审计中的四项“Not verified”风险
  也尚未关闭。
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
  `/tmp/oh-my-vllm-context-boundary-f161b5e*`，自有 GPU 进程均已退出。完整六行运行
  和真实长上下文 HTTP smoke 仍待完成。
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
  宽限计时、低水位、容量、断连、完成态排空和独立排空任务退出。单次在途
  prepare/execute RPC 可按自身超时延后 30 秒检查；服务文档写明此限制。
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
  选中的 attention 用例；它不验证新源码。修改后聚焦 B200 attention 测试通过
  原有 29 项及四项新的 direct-FFI 边界、Position、分组、graph 用例。同卡 eager
  FFI A/B 的旧/新每次调用中位数为 3.4138/3.0865 微秒，spread 为 3.51%/5.33%，
  仅属主机诊断。三次完整修改后 16 项正式尝试因外部 GPU 进程（PID 849461、
  897317、913070）而拒收，均不算验收。`scripts/test.sh cpu` 通过 211 项及
  67 个 subtest（排除 194 项 GPU 用例），Rust workspace、fmt、行宽、Ruff、
  Clippy 与 hooks 也通过。自有 GPU 进程已退出。KRN-10 仍为 open，待干净的
  修改后 16 项、当前完整 147 项/GPU 套件及 12 组框架验收。

后续工作：PY-03..07、SRV-08、KRN-06 与 KRN-10、
EVD-07、四项未验证风险，以及当前完整算子/GPU 套件
和 12 组框架验收。
逐项状态与证据限制见[审计索引](audit-2026-09-23.zh.md)。

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
