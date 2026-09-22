# CUDA kernel 开发

## 模型卷积布局特化

CUDA 模型卷积针对10240个通道和16384的token stride优化，成组加载四个
BF16权重，对FP32累加值使用稳定的sigmoid；中等输入每个线程块处理四行，
大输入处理八行。权重对齐和保守的状态池/输入范围不重叠检查保护优化路径；
其他布局或可能存在别名的情况保留通用CUDA实现。候选状态写入前仍先复制
源状态快照，因此写入另一序列的原始源slot也不会污染本次计算。

未提交源码的13项卷积子集全部通过性能判定。完整正确性测试通过169项及
28个子测试；原有ragged卷积测试在10240/20480之外加入真实16384 stride，
参考计算和容差不变。额外FP64检查覆盖127/128及4095/4096行、权重偏移、
状态池/权重别名、抵消、零附近和大正负值，以及指数下溢附近的负值。
写入快照和未改动slot保持精确一致。临时检查器首次遇到参考值CPU/GPU设备
不匹配；修复该检查器问题后，补测全部通过。

普通/MTP4 eager模型FP64探针通过。强制长度文本中的控制token和重复内容
不代表agentic验收。独立审查与单独的TileFoundry/Nsight观测摘要记录在
`bench/baseline/2026-09-22-cuda-convolution-progress.json`。
完整算子/框架验收仍待完成；默认后端仍为TileLang。

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


已验收的对照源码逐字节复制到 `python/oh_my_vllm/kernels/tilelang_reference`，源码 hash、原提交、依赖锁 hash 和 TileFoundry 固定版本记录在 `development/kernels/tilelang-reference.json`。不得调优对照实现。

在 Python 启动前设置 `OH_MY_VLLM_KERNEL_BACKEND=cuda` 选择新后端，未实现的入口直接报错，不回退 TileLang。迁移期间默认仍为 TileLang，全部 CUDA 验收完成后再切换。原生代码使用独立 TVM FFI、调用方当前 CUDA stream 和 SM100a；TileFoundry 不成为原生算子的运行/构建依赖。必须在图捕获和正式测量前完成编译。

当前里程碑：所有自有 kernel 入口均有原生 CUDA 实现，包括 FP64 相位的融合 RMS/RoPE、FP32 循环 GDN、卷积和分页注意力。注意力合并优化后，现有完整 CUDA 测试通过 153 项和 24 项子测试。独立生产入口的 packed Q/K、接近 262144 的 int32/int64 位置、非默认 stream 和修改输入后的图重放通过 FP64 检查。这些是正确性结果，不代表实际模型或性能验收。默认仍为 TileLang。

正式静态矩阵位于 `development/kernels/cases.py`，完整算子 fixture 保留必需的快照及合并操作，包括三条序列的独立分派、提案/追赶元数据和图缓存用满后的普通执行路径。有效 FA 页排除空页 0，不使用动态 shape 提取。运行：

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python benchmarks/kernels.py --output /tmp/cuda-kernels.json
```

`--operations` 选择诊断子集，部分覆盖不能通过完整矩阵验收。有未提交源码改动的运行仅作诊断。采集程序记录源码 hash、GPU 身份、全部成对样本、置信判定和单独的 TileFoundry 估计；GPU 争用或源码变化使采集失效。首轮诊断因外部 GPU 进程进入而作废，不构成验收结果。注意力及其他小 shape 路径仍需调优。TileFoundry 的 `TileLang` twin 始终绑定冻结实现，不随生产后端选择改变。

`development/kernels/comparison.py` 要求至少三轮、每轮 20 对样本。采用可复现的分层 bootstrap：先重采样轮次，再采样轮内配对，要求平均节省耗时的单侧 95% 下界大于零，同时每轮 CUDA 中位耗时更低。正式程序需交替后端顺序、预热两条路径、恢复可变状态，并保留全部原始时间对；profiler 运行的耗时不能用于该判定。

静态配置选择和完整验收遵循 REQ-KERNEL-002。硬件指标、编译器资源报告和 TileFoundry HIR 估计分开记录。当前 HIR 缺少 FP64，不能证明长位置 RoPE 相位精度；该边界继续使用未改动的独立参考和完整模型测试。

本里程碑已验证：Nsight Compute 可读取 B200 硬件计数器，过滤 quantize_kernel 的采样成功；TileFoundry analyze 和 Nsight CSV 导入均实际运行。非默认 CUDA stream 和改变输入后的 CUDA Graph 重放通过精确 FP8 比较，TileLang 的 14 项回归亦通过。完整有效的最大配置性能和最终框架验收仍待完成。每次运行后停止本任务 GPU 进程。

TileLang 内核函数体仅保留在 `tilelang_reference/`。生产模块只包含校验、公共封装和显式工厂绑定，不再保存第二份 TileLang DSL。冻结的完整算子封装保留原有快照和分配行为，用于公平比较。

原生注意力使用显式 `mma.sync`/`ldmatrix` BF16 fragment、FP32 寄存器累加、共享内存 sector 置换和保留整数宽度安全边界的位置特化。删除重复函数体后，两个后端测试均通过 154 项和 24 项子测试。注意力性能仍需继续调优，正确性通过和 PTX 编译成功不代表已经提速。

采集开发观测指标时，从 `benchmarks/kernels.py --list` 中选择 ID：

```bash
scripts/with-gpu.sh scripts/with-env.sh env CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python -m development.kernels.observations --case attention-c426ebd5d5 --output /tmp/kernel-observations.json
```

该命令运行 TileFoundry HIR 分析，并分别用 Nsight Compute 重放两个后端的完整算子。预热/JIT 位于 NVTX 范围外。每次 kernel 启动必须包含全部请求的有限值指标；不可用指标明确记录，并返回失败退出码。报告区分静态估计与实测计数器，包含寄存器/共享内存资源、源码哈希和硬件身份，不用于判定任何性能门槛。临时原始 CSV 会删除；取消时终止自有 profiler 进程组，包括 GPU worker。CPU 测试覆盖不完整/非有限指标和取消清理，实际双后端注意力 profiling 已成功。

连续 BF16 输入也可能因 storage offset 而未对齐。原生注意力为此使用 Q 标量加载；生产封装为两个后端对齐 KV，并为冻结 TileLang 对齐 Q，不修改冻结源码。新增 Q/KV 对齐回归继续使用现有数值容差。最新 CUDA 完整测试通过 160 项和 28 项子测试，TileLang 注意力通过 16 项。寄存器 softmax 和按形状选择的 KV 预取仍在性能调优中。
