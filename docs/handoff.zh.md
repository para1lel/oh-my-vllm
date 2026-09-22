# 交接记录 — 2026-09-22

## 向量化 GDN 递推

干净提交 c40cd5c 的完整矩阵通过 129/147 项，未检测到其他 GPU 进程干扰。
剩余 18 项为递推 7 项、Q/K 归一化 6 项、卷积 5 项。此完整结果与后续子集
优化分开记录，不累加计算通过数。

模型递推现在由 16 个 lane 各处理八个连续 key 元素，向量加载状态及 Q/K，
成对转换 BF16 快照，验证过程中的持续状态始终保留为 FP32。使用受限指针
前检查模型 head 数、基址/行对齐及保守的内存范围互不重叠条件。
通用 head 数、未对齐视图和状态池可能与输入重叠的情况保留原有通用 CUDA
路径。同序列写回自身源状态仍有效；未增加 TileLang 回退。

前两次正式子集均通过 7/8 项，FP32 batch3 未通过。改为四个 warp、每半个
warp 处理两个 value 行后，最终子集通过 8/8 项。完整正确性测试通过 168 项
及 28 个子测试。额外 FP64 检查覆盖通用 head 数、输入/状态偏移、奇数 token
stride、源状态别名、padding 范围重叠、混合 metadata 位宽、三序列及原位写回，
均保持原有容差。普通/MTP4 eager 模型探针与独立 TileFoundry/Nsight 观测
记录在 `bench/baseline/2026-09-22-cuda-recurrent-progress.json`。

独立审查通过。这仍是迁移中间证据：完整算子与框架验收尚未完成，默认后端
仍为 TileLang。

## 向量化 FP8 量化与倒数修正

普通量化改为每次加载、输出四个元素。主机分派时检查输入对齐，带存储偏移
且未对齐的视图保留标量加载。4–127行时，将独立量化组平铺到四个warp的线程块。
BF16的有限最大值使用近似倒数加一次FMA余差修正；scale仍用精确除法计算。
FP16/FP32及非有限最大值仍使用精确除法。SiLU的两次BF16舍入不变。
这不表示与FP32精确除法等价。

SM100/CUDA13.1上的临时穷举覆盖满足abs(input)<=maximum的有限BF16输入/最大值组合，
修正后正负FP8结果
均无差异；直接乘近似倒数则有差异。入口补测涵盖按组放入全部BF16编码、
所有支持的dtype、两种scale布局、对齐/偏移视图、通用/模型宽度、
3/4/127/128行边界以及NaN/Inf分组。这些检查的FP8输出位和scale与旧CUDA入口
完全一致，但不构成任意输入张量的穷举证明。

格式整理前后的39项量化/SiLU子集均全部通过。最终源码结果及单独采集的
TileFoundry/Nsight观测记录在
`bench/baseline/2026-09-22-cuda-quant-vector-progress.json`。
完整正确性测试和普通/MTP4 eager FP64探针均保持原容差通过。
强制输出64个token的文本含控制token和重复内容，不代表agentic验收。
完整147项算子及框架验收仍待完成；默认后端仍为TileLang。

## 稳定的 GDN beta 与完整算子矩阵

干净提交087bb51的完整矩阵完成147项：113项通过、34项未通过，未检测到
其他GPU进程干扰。未通过项为量化12项、递推7项、Q/K归一化6项、卷积5项、
门控4项。这不代表最终验收通过。

GDN beta改用有界指数和分母位于[1,2]的快速除法；decay和现有误差容限不变。
未提交源码的13项门控子集全部通过性能判定。完整CUDA正确性测试通过168项
及28个子测试。临时穷举全部65536种BF16 beta编码，与FP64 sigmoid对照，
均满足原有绝对容差1e-7、相对容差1e-6；正负零、无穷和NaN均符合预期。
有限值最大绝对误差9.1063e-8。这不表示逐位等价。

独立审查通过。单独采集的TileFoundry估算、配对Nsight观测以及正式结果摘要
保存在`bench/baseline/2026-09-22-cuda-gates-progress.json`。子集通过项不会加到
之前完整矩阵的计数上。完整框架性能、上下文及服务验收仍待完成；默认后端
仍为TileLang。

## 全注意力准备流程融合

主模型和 MTP 现在共用生产准备入口，CUDA 将 Q/K RMS/RoPE、V 布局转换及 KV
追加合并执行，返回 Q 并写入相同的 K/V 缓存行，packed gate 值保持原样。
两次 BF16 舍入和 FP64 相位约化均保留。负 slot 仅跳过 KV 写入。
TileLang 后端仍执行原有完整冻结链，保留 V.contiguous() 的真实行为：
已经连续时不复制，不人为强制增加副本。

根据变化后的真实生产调用重新推导静态配置：13 个融合配置替代 26 个独立
RMS/RoPE 和 13 个 append 调用，共 147 项。主模型 Batch 和 MTP/DraftGraph 的
positions/slots 是 int64，与注意力的 int32 tables/lengths 无关。
独立 API 保留正确性测试，并明确列为这些性能工作负载未调用的入口。
旧 173 项矩阵的 111 项通过、62 项失败记录保留，不重新算作通过。

未提交源码的融合子集 13/13 通过速度判据。CUDA 完整测试通过 168 项及 28 项
子测试；新增八组检查在冻结 TileLang 后端也通过。覆盖单/五 token、所有
positions/slots dtype 组合、接近 262144 的位置、跨页和分散 slot、V 与未写缓存
逐位一致、两次 BF16 舍入、非默认 stream，以及改变输入和写入目的地后的图重放。
真实模型 ordinary/MTP4 eager FP64 探针通过；强制长度文本不代表语义或 agentic
验收。TileFoundry 代表性逻辑链检查和单独的 Nsight 采集通过；HIR 中缓存拼接和
FP32 相位仍是估算，不代表原生流量或精确相位验证。

证据见 `bench/baseline/2026-09-22-cuda-attention-prepare-progress.json`。
独立审查和全文件检查通过，本任务 GPU 程序均已退出。完整 147 项及框架验收
尚未完成；默认后端仍为 TileLang。

## 中等规模残差 RMS 的缓存策略

仅 2048–4095 行残差 RMS 的两个输出使用 streaming store 缓存提示，保留 BF16
位模式和同 stream 可见性，并减少算子工作负载中的输入缓存淘汰。未提交源码的
26 项 RMS 子集全部通过判据；此前失败的 2496 行配置，配对平均时延减少约
1.30 μs。这不代表完整算子或框架验收，下游缓存未命中的潜在影响仍需最终
端到端性能门槛检验。

完整 CUDA 正确性测试通过 160 项及 28 项子测试，2047/2048/4095/4096 行的
FP64 边界检查通过，残差和逐位一致。首个打包 intrinsic 的编译兼容性问题已修复，
失败测试进程已停止，随后完成上述成功重跑。独立审查、单独采集的
TileFoundry/Nsight 观测和全文件检查通过。证据见
`bench/baseline/2026-09-22-cuda-rms-stream-progress.json`。
本任务 GPU 程序均已退出；默认后端仍为 TileLang，迁移继续进行。

## 门控 RMS 专用路径与完整矩阵更新

干净提交 d45c4f2 完成全部 173 项：111 项通过、62 项未通过，未检测到 GPU 干扰。
注意力 16 项全部通过。2496 行残差 RMS 虽然此前子集测试通过，但本次完整采集未通过，
优势还不可靠。其余失败项包含 KV 追加、小规模量化/旋转、Q/K 归一化、递归更新、
gates 和卷积。完整验收尚未通过。

新增 48-head、128 维门控 RMS 路径，在寄存器中保留输入，使用分母位于 [1,2] 的
稳定 sigmoid 表达式。门控计算保持 FP32，不增加 BF16 中间舍入，并支持原有 stride
和 epsilon。未提交源码的门控 RMS 子集 13/13 通过，完整 CUDA 正确性测试通过
160 项及 28 项子测试。独立审查、额外 FP64 packed/非单位 stride 检查，以及
大权重与负 gate 组合检查均通过。也检查了所有有限 BF16 gate 编码，但固定输入为 1、
权重为 0.25；这不是全部输入/权重组合的穷举或逐位一致证明。原有容差未修改。
单独采集的 TileFoundry/Nsight 观测、完整矩阵判定和子集证据摘要见
`bench/baseline/2026-09-22-cuda-gated-rms-progress.json`。

全文件检查通过，本任务 GPU 程序均已退出。默认后端仍为 TileLang，
还需继续调优算子并完成框架的完整验收。

## RMS 线程数与向量宽度调优

模型宽度 RMS 现在对中等 batch 使用 128 线程，对大型普通 RMS 使用 160 线程，
对大型残差 RMS 使用 320 线程。大型残差仅在输入和残差满足 16 字节对齐、权重
满足 32 字节对齐时使用八元素向量，否则保留四元素向量或通用路径。
编译期整除检查保证完整覆盖每行。归约顺序会变化，原有容差保持不变。

最终相同源码复测的 RMS 子集 26/26 通过。此前两次诊断分别为 24/26 和 25/26；
后者虽然三轮中位数都更快，但一个异常高延迟的 CUDA 样本导致置信下界未通过。
没有排除样本，也未检测到干扰，具体原因尚未确定。所有尝试的摘要及单独采集的
TileFoundry/Nsight 观测见 `bench/baseline/2026-09-22-cuda-rms-progress.json`。
这些使用未提交源码的子集结果不代表完整算子或框架验收。

完整 CUDA 测试通过 160 项及 28 项子测试。临时 FP64 检查覆盖
2047/2048/4095/4096 行的分派边界、非默认 epsilon，以及 4096 行的两级对齐回退；
残差和与参考仍逐位一致。独立审查和全文件检查通过，本任务 GPU 程序均已退出。
默认后端保持 TileLang；继续调优剩余算子并完成完整验收。

## 注意力片段复用与合并优化

注意力诊断的 16 个静态配置全部通过速度判据，每项完成三轮、每轮 20 对
交错测量，未检测到 GPU 干扰。本次使用未提交源码且仅覆盖子集，不代表完整验收。
Q/K 和概率/值的 MMA 片段在输出分块之间复用；静态寄存器索引消除了局部内存溢出。
合并阶段使用共享的分片权重和向量输出。非分组 KV 双缓冲仅用于四个查询，
以保留较大 eager batch 的占用率。最终采样的 eager 配置没有局部内存读写；
TileFoundry 估算仍与实测计数器分开记录。

未修改的完整 CUDA 正确性测试通过 160 项及 28 项子测试。真实模型 ordinary 和
MTP4 的 eager FP64 探针通过，包含递归状态及草稿注意力。强制长度的探针文本
不代表语义或 agentic 验收。合并归约顺序发生变化，因此通过数值容差检查并不
证明逐位一致。独立审查及全文件检查通过。源码身份、计时、硬件观测和探针覆盖
摘要见 `bench/baseline/2026-09-22-cuda-attention-progress.json`。

RMS、Q/K 归一化、KV 追加及其他失败配置仍需调优。最近一次完整、干净源码矩阵
仍为 80/173，不能累加不同子集的通过数。完整算子及框架性能验收尚未完成；
默认后端保持 TileLang。本任务启动的所有 GPU 程序均已退出。

## 成对SiLU后续优化

干净提交cb867d4完成173个算子配置测量，未检测到GPU干扰：80个通过，93个失败。
下一版诊断采用固定宽度特化的FP8 kernel和BF16成对SiLU乘法。全部13个融合SiLU配置
均满足既有三轮速度判据，包括此前失败的中等规模；本轮仍为未提交源码的子集诊断，
不能视为完整验收。完整CUDA正确性测试仍通过160项测试和28项子测试。独立审查、
布局/数据类型边界检查及全部文件钩子均通过。独立采集的TileFoundry/Nsight摘要和
源码标识见`bench/baseline/2026-09-22-cuda-silu-progress.json`。

快速指数与成对乘法均保留两处BF16舍入。针对当前SM100/CUDA13.1的临时穷举检查显示，
所有有限BF16输入的SiLU舍入结果零差异；所有有限BF16乘数组合的成对乘法，与
“FP32乘法后舍入为BF16”也零差异。检查包含正负零、次正规数、上溢和下溢，但不构成
其他架构/工具链或任意FP32 SiLU的等价保证。临时调优脚本和原始轨迹未纳入仓库。
注意力、Q/K归一化、KV写入及其他失败配置仍需调优；CUDA模型、性能及服务的最终
验收尚未完成，默认仍为TileLang。本任务的GPU程序均已退出。


## 向量算子优化进展

干净提交7437e0f完整测量了173个算子配置，未检测到GPU干扰：48个通过，125个失败。
后续向量算子诊断覆盖91个归一化/量化配置，其中55个满足速度判据；同一组配置在修改前
只有23个通过。新测量使用未提交源码且仅覆盖部分配置，因此只能作为诊断。各版本结果
和配对硬件指标摘要见`bench/baseline/2026-09-22-cuda-vector-progress.json`；CUDA算子
及框架的完整验收仍未完成。

原生量化按行和缩放组分块，利用非负FP32数值位序做无符号warp REDUX归约，并对融合
SiLU采用向量加载和FP8打包存储。小尺寸融合路径及超过CUDA grid.y容量的宽度使用
扁平网格。精确除法和两处BF16舍入保持不变。5120宽度的RMS在寄存器中保留数值，
对齐条件满足时使用向量加载，其他布局走通用路径；残差先按BF16数对相加并舍入，再归一化。
融合RoPE采用FP64只读固定频率表，相位归约仍为FP64。仅独立分配输出的kernel声明
指针不别名，原地状态和缓存kernel不使用该声明。

完整CUDA正确性测试通过160项测试和28项子测试。补充边界检查覆盖非默认RMS epsilon、
输入/权重的非对齐存储偏移、FP16/FP32量化、分派边界和极宽量化。已有长位置FP64、
stream/graph检查及全部文件钩子均通过。独立审查发现的量化grid.y宽度限制已修复。
大尺寸RMS、小尺寸量化、融合SiLU及注意力仍需调优，默认后端继续保持TileLang。
测试和剖析进程已退出，没有遗留本任务的GPU进程。


## 当前任务：CUDA/PTX 自有 kernel

用户已确认 REQ-KERNEL-002，CUDA 迁移正在实施，尚未验收。全部自有入口均已实现原生 CUDA。共享注意力合并权重优化后，未改动的完整 GPU 测试通过 153 项和 24 项子测试，无跳过。生产入口 packed Q/K、接近 262144 的位置、非默认 stream 和修改输入后的图重放通过独立 FP64 检查。早期漏掉的卷积 factory 绑定已在这些成功重跑前修复。

正式静态矩阵由 12 个工作负载产生 173 个去重配置，覆盖三条序列分派、图元数据类型和普通执行回退。完整冻结算子保留快照/合并成本。子集和有未提交源码改动的运行不能通过完整验收；采集程序校验冻结源码和依赖锁 hash。TileFoundry twin 现显式绑定冻结 TileLang。原生实现和矩阵审查发现的问题已处理。首轮性能诊断因外部 GPU 进程进入而失效，不宣称正式结果。注意力及其他小 shape 仍需调优。剩余：逐配置稳定提速、完整指标、实际模型正确性、12 组冻结 vLLM 对比及边界/服务/agentic 验收。全部完成前默认保持 TileLang，测试 GPU 程序均已退出。见 cuda-development.zh.md。以下 TileLang 结果是已验收对照。

按用户要求，删除七个生产模块中重复的 13 个 TileLang 工厂函数体。DSL 只保留在冻结 TileLang 中；公共封装 AST 不变，改为显式名称绑定工厂，参考哈希校验通过。原生注意力现使用寄存器累加、内联 PTX MMA/ldmatrix、共享内存置换和形状/位置类型特化。两个后端完整重跑均通过 154 项测试和 24 项子测试，无跳过，独立审查通过。以上仍是正确性里程碑：注意力诊断计时仍慢于 TileLang，不宣称性能验收。本任务 GPU 程序已退出；另一个无关 GPU 进程仍在运行，未停止它。


最新注意力调优将 score/softmax 状态留在寄存器中，分组路径共享 32-query/64-key 分块，较大的非分组批次重叠 KV 预取。审查发现连续 BF16 storage-offset 输入可能未对齐：原生 Q 已有安全标量加载，生产封装为两个后端对齐 KV，并为冻结 TileLang 对齐 Q；冻结源码不变。CUDA 完整测试通过 160 项和 28 项子测试，TileLang 注意力通过 16 项，包括三种新增对齐场景。硬件观测 CLI 现分别 profile 两个完整算子，单独标注 TileFoundry 估计，核验计数器/源码身份，并在取消时清理自有进程组。实际双后端 Nsight 采集和 CPU 校验通过。性能诊断仍未满足相对 TileLang 门槛，正式逐配置和完整框架 CUDA 验收尚未完成。



## 当前：TileLang 验收和修订后的稳定性门槛

项目自有的七个自定义 kernel 模块均已改用 TileLang。干净测量提交 da75c02 在用户批准的极差/中位数 10% 上限下，通过全部 12 组冻结 EngineCore 吞吐/TTFT 对比；用户于 2026-09-22 将该上限从 5% 调整为 10%。重新判定使用同一批完整原始测量，保留原判定。prefix batch 1 的 TTFT 波动为 5.97%，此前仅因旧门槛失败。最低吞吐比例 95.3098%，最高 TTFT 比例 101.4735%。判定工具输出稳定性上限，CPU 测试覆盖两个引擎、两种指标低于、等于和超过 10% 的情况。本次规则修订没有改动 GPU 推理源码。

未改动的完整 GPU 测试通过 146 项和 12 项子测试，无跳过。普通/MTP 实际模型 FP64 探针、六组总长 262144 token 边界（无 OOM/抢占）、文本/抢占/前缀、12 项 MTP4 约束、生命周期和长上下文服务均通过。真实 oh-my-pi 两种 API 各完成九次成功工具调用，均有正数 MTP 活动。回答生成于最终验收文档更新前，已保留原文和依据准确性方面的局限。83d77ff 上的最终文档读取复核也通过：Chat 成功调用 7 次工具、耗时 22.51 秒；Responses 成功调用 13 次、耗时 34.78 秒。全部 7 次模型请求均有正数 draft 提出/接受。最终请求吞吐为 252.31/234.24 tok/s，TTFT 为 279/351ms；首个冷请求包含编译。所有本任务 GPU 程序已退出，18025/18026/18027 端口已释放。独立审查发现的回答局限见 acceptance.zh.md，无待解决性能阻碍。完整证据见 acceptance.zh.md 和 bench/baseline/2026-09-22-tilelang-*.json。

TileFoundry 仅用于开发，固定个人 fork 6b1b149。离线 TileLang AutoTuner 和成本/显存分析用于选择生产参数。HIR 不能表达 FP64 相位归约，正确性以独立 FP64/模型测试为准。全部 25 份项目自有 Markdown 文档有中文配套版本。以下旧章节是历史进展，不代表当前阻碍。


## 双语文档（用户要求）

全部 25 份项目自有 Markdown 源文档现有同目录 `.zh.md` 译本。agent 继续阅读并以英文原文为准；今后每次 Markdown 修改需在同一次改动中同步中文。AGENTS.md 和 CONTRIBUTING.md 已记录规则。第三方 submodule 和生成的依赖除外。独立翻译审查、完整译本覆盖、本地链接和代码块核对均通过。文档工作不替代 kernel 验收。

最新 kernel 状态：干净 874a54a 的正式 MTP batch4 达到 94.9688%，仍低于 95%。Q/K RMS+RoPE 融合保留中间 BF16 舍入，静态审查通过。其完整 GPU suite 重跑通过 146 项测试和 12 项子测试；首次运行因另一个进程进入所选 GPU，导致进程超时测试失败，不是数值失败。实际普通/MTP 模型探针通过，但生产入口 FP64 检查发现接近位置 262144 时的 RoPE 相位误差。生产修复用 FP64 计算并约化相位，在原容差下通过 packed Q/K、int32/int64 位置和图回放。修复后完整 146+12 suite 及实际普通/MTP FP64 覆盖通过。TileFoundry 分析和代表性 twin 检查通过；文档明确说明其不能表达 FP64 相位，不能验证这个边界。最终 12 组性能和功能验收仍必需。

## 当前任务：TileLang 迁移与 TileFoundry 开发工作流

用户确认开始实现。将七个模块内全部 15 个项目自有 Triton kernel 替换为 TileLang，保留公开行为和已有正确性测试/容差。第三方内部 Triton 不在范围内。TileFoundry 仅用于开发，从上游 main fork 到 para1lel/TileFoundry 的 oh-my-vllm-integration 分支，固定为 3rdparty/TileFoundry。主项目仍在 main，不提上游 PR。取消 fork 的 Transformers 上限，在 oh-my-vllm 源码安装。允许兼容 TVM FFI/protobuf 降级，不维护 TileLang/OR-Tools fork。简单修复已授权，重大修复先讨论。

临时算子实验和记录放在仓库外，最终测试/证据限用户要求及已有文档规定的验收。保留 vLLM 比例门槛，不设相对 Triton 门槛。分析、优化并解决性能失败。用户加强目标：全部 12 组必须通过吞吐、TTFT 和稳定性；只有实测原因解释不算完成。结合 TileFoundry 分析，在所需 shape 上调优。

当前七个生产 kernel 模块均使用 TileLang，无自有 Triton 导入或回退。token 维度不决定算法/布局时采用动态维度；既有 CUDA Graph shape key 不变。TileFoundry fork 6b1b149 含依赖及 HIR sin/cos 适配，已审查并推送。开发专用完整算子 HIR/twin、安装/工作流文档位于 development/kernels 和 docs/tilelang-development.md。

动态长度修复后，完整 GPU pytest 146 项 +12 子测试通过，无跳过；Rust workspace、fmt、硬行宽、clippy 通过。依赖闭包检查和开发安装 dry-run 通过。实际 ordinary FP64 通过。MTP FP64、12 个约束、生命周期及长前缀服务在动态长度修改前通过；最终模型/agentic/边界及正式 12 组仍在进行，暂无最终性能结论。一次诊断与源码编辑重叠，延迟 JIT 检查失败，已丢弃并针对固定源码重跑。临时 MTP driver 把覆盖标志设为 4 而非 1，导致关闭覆盖检查失败；更正后重跑。没有修改容差或测试。算子实验和记录仍在仓库外。

迁移优化保留动态 token/batch，在设备端计算注意力 split 范围。KV gather 显式合并访问放置，一/两序列 GDN 使用 warp 局部归约。审查无正确性阻断。优化后完整 suite 再次 146+12 通过，无跳过；实际普通/MTP FP64 均通过。六组边界在优化前通过，优化后的边界、服务/agentic 随后在 6a6e6a5 通过。首轮正式矩阵七组通过、五组失败：MTP batch 1/2/4 吞吐不足，prefix batch 1/2 TTFT 不稳定。这不是最终验收。后续两序列 GDN 布局通过完整 146+12，继续注意力 shape 调优，最终模型/性能重跑待完成。

下一个注意力里程碑使用带谓词异步 global-to-shared copy、128 线程、64 split。只有 ceildiv 有足够余量时逻辑位置使用 int32，物理缓存偏移始终 int64。审查验证共享内存屏障和尾部补零。重跑全部 146+12 通过；较早一次进程超时测试因 GPU 争用失败，不是数值问题。全文件格式、Rust 检查和 hook 通过。现已离线使用 TileLang AutoTuner，显式合法元数据、CUDA Graph 计时和不变 FP64 检查。工作流要求完整算子和模型重验，不运行在线搜索，不在仓库保留微测试。实际普通/MTP FP64 满足全部必需覆盖，完整性能矩阵仍在进行。

f52a2d4 正式 MTP batch1/2 通过（基线的 97.386%/95.351%），batch4 仍为 92.248%，TTFT/稳定性通过。进一步离线调整 tile1 FP8 量化线程、去掉宽度 5120 的残差 padding、减少大 batch GDN 线程和短范围验证 split。完整 suite 再次 146+12，通过实际 batch4 GDN FP64 输出/状态检查。最终模型/服务/边界及 12 组仍必需。

b868c34 正式 MTP batch4 为 94.6674%，TTFT/稳定性通过，但仍低于 95%。下一里程碑流式累加注意力 merge，避免大 weighted fragment，使长非分组 decode 可采用 128 split/BK32。短行 RMS 使用一个 warp。全部 146+12 GPU 测试和全文件 hook 通过；临时长范围 FP64 覆盖不存在的位置 0、不等长度和更改长度/页表后的图回放。最终验收仍待完成。下列历史性能属于 TileLang 之前的实现，不是本次迁移验收。

## 前次调查：TTFT 与长上下文

用户批准基于最新官方 vLLM main 的新 12 组吞吐/TTFT 基线，以及普通/MTP4 batch 1/2/4 合计 262144-token 边界，见当前需求。新验收待完成，下列结果描述此前任务。冻结上游 SHA：e9f169d16b9408bb9ae44f75072b91a5521d733c。更新 main 前，将旧本地 vLLM telemetry 提交保存为 /data0/shared/dongwu.chen/vllm-before-ttft-20260922.bundle。

基线环境适配完成。最初无法获取确切 cu130 wheel 元数据，因此开始源码构建；官方 wheel 于 02:13 UTC 发布完毕，重查成功，自有源码构建在编译前停止。通过上游 editable 模式安装官方冻结提交 wheel，版本 0.29.1rc1.dev505+ge9f169d16.precompiled，Torch2.13.0+cu130。pip check 和 vllm._custom_ops 导入通过。扩展 hash 见 /tmp/oh-my-vllm-baseline-install-official.log 和 /tmp/oh-my-vllm-ttft-final/baseline-environment.json。

已实现但尚未验收：逐请求 TTFT、可选独立 GDN 池、正式测量/来源检查、修正 Qwen3.8 名称。77 个 Rust 测试通过。为新模型 ID 重建 debug 二进制后，完整 GPU pytest 100 项 +12 子测试通过，无跳过。格式、clippy、全部 hook 通过。131072 输入 batch4 短输出诊断无 OOM/抢占。首次完整 262144 边界暴露有符号 int32 decode 页偏移溢出，乘法前加宽并添加高页 FP64 测试，9 项通过。

六组 ordinary/MTP4 batch 1/2/4 边界完成，输入 258048 + 输出 4096，零抢占。峰值 reserved 显存普通 134951731200 字节、MTP4 140033130496 字节。这些冷诊断不是正式性能测量。实际普通/MTP FP64 保持原容差。12 个 MTP 约束、生命周期及两 API 131099 输入通过；重复请求复用 130928 token。全部自有边界/探针/服务 worker 退出，18015 端口释放。原始边界及日志 hash 见 bench/baseline/2026-09-22-ttft-stage1.json。

两 oh-my-pi API 使用 Qwen3.8 ID 和 MTP4 重跑：Chat/Responses 成功读取 10/8 次，含两份必需源码，工具错误零，耗时 28.02/22.16 秒。GB/GiB 单位错误和其他回答限制保留在 bench/baseline/2026-09-22-ttft-agentic.json。服务/worker 退出，18016 释放，无关 GPU 进程未动。

prefill profile 发现逐行 FP8 量化低效。16 行 tile 在模型诊断中使量化 GPU 时间从 295 降至 51ms。算术不变，小/decode shape 保留旧 kernel。最终阈值调优前完整 suite 103+12，最终边界/FP8 测试 14 项通过。限制及 hash 见 2026-09-22-prefill-quantization.json。优化后普通/MTP 实际 FP64 也通过。最初四次基线因外部 TP4 作业进入 GPU 中止，见 2026-09-22-ttft-discarded.json，结果排除。采集器改为流式写盘，干扰/超时也保留子输出。

当时十一组基线通过审计，剩余在 /tmp/oh-my-vllm-ttft-final 采集/重试。Prefix4 attempt103 吞吐稳定，但 TTFT 极差 13.94%，被拒绝。用户确认遇外部干扰继续等待/重试，当时尚无完整 12 组。候选 MTP attempt0 因复制 FlashInfer 缓存引发高代价重编译，在首次预热取消，未使用测量结果。后续使用现有独立 FlashInfer 缓存及可审计的逐运行 Triton 缓存。

长 prefill 注意力现在压紧活跃 KV，使用独立 ragged TRT-LLM，保留持久 784-token 页和小 query FA2。完整 suite 110+12、两条实际 FP64 路径通过。普通/MTP4 bs4 最大上下文重跑无抢占，峰值 reserved 不变，见 2026-09-22-ragged-prefill.json。最终数值检查含混合长 decode 隔离，3 项通过。完成基线采集、性能修复和审查后才能宣告完成。

prefill 卷积/SiLU tile 和 dual-SM 大投影 GEMM 将诊断 GPU profile 从 1.816 降至 1.721 秒。GPU suite 113+12、实际 MTP target/draft FP64 通过，见 2026-09-22-prefill-tiles.json。首批候选暴露共享 Triton 缓存审计污染，后续必须独立根目录。0ad2bb4 的 131072 输入 batch1 候选两门槛通过，其他首批仅为诊断。MTP 吞吐仍未达标。隔离基线步数诊断 898 对项目 899，指向每步成本而非接受率。临时私有 GEMM tactic 原型触发非法访问，未用于运行时。

packed GDN/卷积/RMS 输入保留 token/head stride，target/draft 元数据合并传输。审查要求每 batch 只转一次 int32 start，避免反复 FlashInfer cast，已实现。GPU suite 120+12、实际 MTP FP64、12 个服务约束、生命周期和 131099 输入前缀通过，服务 18017 退出释放。MTP bs1/2 诊断提升至 305.86/450.65 tok/s，仍低于新门槛，见 2026-09-22-strided-metadata.json。Prefix2 基线因方差及无关 GPU 任务反复进入继续重试，用户授权。

已实现残差 RMS/SiLU 量化融合、独立 CuTe-DSL 词表投影和零拷贝 16-token 子页原生 target decode。784-token 持久布局不变，draft 排除位置零仍用自定义 kernel。最终 suite 138+12。实际 batch2 target/draft 图预热 FP64 通过，与也通过的 eager GDN 探针分开。12 个约束、生命周期和长前缀通过，服务 18018/worker 退出。初始 MTP bs1/2/4 诊断 319.66/473.69/672.44 tok/s，不是正式验收，batch4 仍低于 95%。四序列更大 GDN tile 和八 warp 残差 RMS 通过最终测试/审查；短 MTP4 诊断 674.03 仍低于目标。

六组最大上下文重跑全部零抢占。launch 调优发生在边界 worker 启动之后，证据限制明确记录在 bench/baseline/2026-09-22-native-decode-fusions.json。全部自有诊断/探针/边界/服务 worker 退出，18018 释放。格式、Rust 测试/clippy 和全部 hook 通过，正式采集仍进行。静态审查无阻断，并明确要求区分图预热 FP64 与 eager 覆盖。

Prefix2 从 attempt110+ 暂用 CPU96 单核调查 TTFT 方差。attempt112 仍极差 37.94%，保留所有尝试。候选必须匹配最终接受基线的亲和性。用户再次确认无法预留独占 GPU，继续等待/重试。

全部 12 组更新基线随后采集完成，冻结在 bench/baseline/2026-09-22-refreshed-enginecore.json。Prefix2 attempt114 以 CPU96 单核通过，候选须相同。重复 attempt201 在接受后取消，worker 退出。5166885 上六 ordinary 和 MTP batch1/2 正式两门槛通过；三 prefix 的 TTFT 失败，MTP batch4 被外部 GPU 到达打断。未宣告完整验收，保留尝试在 /tmp/oh-my-vllm-ttft-final。

用户要求的实现/性能对比记录在 docs/performance-gap-2026-09-22.md，含冻结上游链接及 profile hash。draft 注意力是最明确、可隔离的不利 kernel 差异；FP8 GEMM、词表投影和 target 注意力有竞争力。CPU 采样注解含 GPU 等待，不能标成 ZMQ 开销。融合 GDN prefill Q/K 归一化、原生短前缀 target context attention、独立 TRT ragged MTP prefill 已实现并审查。

最新 MTP batch4 诊断 692.012 tok/s，对门槛 693.140，仍不足且非正式。GPU suite 146+12 无跳过。实际 FP64、12 个约束、生命周期、长前缀通过。六组最大上下文无抢占；普通边界早于最终 MTP 专用修改，MTP 边界在修改后重跑。见 2026-09-22-prefill-research-verification.json。全部自有 worker 退出，当时 nvidia-smi 为空，18020 释放。ee2fb63 已推送。

干净 ee2fb63 正式后续完成全部 12 组，十组通过。MTP batch4 中位数 692.1899 tok/s，为 94.8698%，低于 95%；稳定且 TTFT 通过。还需 0.1373% 吞吐，即每完整 batch 减少 32.45ms。Prefix batch1 两比例通过，但两次 TTFT 稳定性失败，最新 44.72–51.16ms、极差 14.20%。Prefix batch2/4、全部 ordinary 和 MTP batch1/2 通过。完整数据及被排除外部干扰尝试见 2026-09-22-performance-gap-candidates.json。自有 worker 退出，当时 nvidia-smi 为空。调查完成，但整体验收未完成。优先 mask draft 注意力，再做 GPU 常驻已接受 token 记账，另调查短前缀时延方差；保留位置零排除及全部容差。

## 前一任务：独立运行时

独立运行时已实现，原冻结 EngineCore 九组全部 >=95%。两种最终 oh-my-pi 以 MTP4 针对对齐后的文档完成。

Rust 拥有 HTTP、调度和逻辑 KV。Python 用项目代码、PyTorch/Triton/FlashInfer、XGrammar 加载运行 Qwen。不安装/导入/链接 vLLM，不依赖旧源码/环境，无旧 runner。使用 scripts/with-env.sh 和 scripts/with-gpu.sh。目标模型、单张 B200、block784、conda oh-my-vllm 不变。requirements/runtime.txt 固定依赖，ADR-006 说明扩展范围。

## 前一任务验证

- GPU pytest 92 项 +12 子测试，无跳过；Rust 75 项通过。
- fmt、硬行宽、clippy、ruff、全部 pre-commit 通过。
- 实际普通/MTP FP64 保留原容差；分组 target/draft 注意力和图 buffer 生命周期/缓存恢复通过。
- 不同城市文本经前缀复用、错峰和重计算抢占通过。12 个真实 MTP 约束和生命周期（混合 batch、续接/删除、断开/释放）通过。
- 六 ordinary/prefix 使用干净 ca5a200，三 MTP 使用干净 5c0848e；两者只改变 MTP 执行。既有图定义 AST 相同，其他非 MTP 源码不变。逐组身份、原始重复和 hash 在最终独立验收产物中。
- 源码/运行时/文件访问审计未发现旧 vLLM 依赖，kernel 缓存属于本项目。未重跑或替换原 EngineCore 基线。

数值和证据见 acceptance.md 及 bench/baseline/2026-09-21-independent-acceptance.json。中间产物保留失败性能、外部 GPU 污染、两次 CPU 重叠排除和生产前已修复的独立原型 buffer 生命周期失败。诊断绝不代替正式重复。

## 前一任务 agentic 验证与清理

Chat/Responses 成功 read 7/18 次，含两必需源码，工具错误零，并有真实后续请求。端到端 26.27/36.68 秒。两答案识别项目及完成状态；Responses 小措辞/计数错误明确保留在 acceptance.md 和 JSON，不声称完美依据文件。

前一独立运行时范围无剩余工作。全部自有 GPU 程序退出，当时 nvidia-smi 无计算进程，18013/18014 释放。没有要求保持服务运行。

仅 main 开发/提交，暂存有意改动，保持 hook 开启和规定署名。不得遗留任务自有 GPU 程序。
