# 阶段时延的理论分析

[性能要求](requirements.zh.md#req-perf-001-阶段时延) 使用每次测量中各请求的计算工作.
模型读取语义执行记录. CUDA Event 时长仅用于诊断.

## 请求边界

Prefill 从提交含 token IDs 的请求开始, 到首个保留的输出 token 结束.
Decode 从该 token 开始, 到最后一个保留的输出 token 结束.
墙钟区间包括 Rust 调度, Python 执行, ZMQ 和采样.
模型加载, tokenization, HTTP 和预热位于这些区间之外.

每次重复分别使用最长的请求 prefill 区间和最长的请求 decode 区间.
两个最大值可能来自不同请求.
Batch 总时长和 cleanup 时间使用独立字段.

Worker 在已有的 token 回读位置记录已完成的工作.
阶段终点排除最后一次回读之后的工作.
分析根据 Rust 结果验证请求 IDs, 输出数量, 执行 step, 模式和状态精度.

## 语义计算量

`performance/model.py` 描述目标层, MTP, DSpark, 词表投影, greedy 选择和持久状态.
每个投影使用其有效行数及输入和输出维度.
矩阵乘法计入 `2*M*N*K` 次算术运算.
FP8 分组 scale, attention KV 长度, GDN chunk 和条件草稿路径分别计数.

DSpark backbone 执行时, 模型计入完整的七行 backbone.
模型计入被拒绝的验证行和必要的草稿计算.
Padding 和实现中的重复工作不提高下界.

参数元数据描述已加载的 tensor, 包括转换后的 scale.
模型计入每个必需的参数 buffer 和选中的 embedding 行.
GPU 流量依据 tensor, 而非 checkpoint 的磁盘大小.

### 必需的输出行

历史行继续为后续注意力提供 K/V. 选中的行提供最终 hidden 与 logits. 选择范围保留所有验证采样行, 包括被拒绝的候选. 原生 MTP 还保留供 context 输入及已注册边界特征使用的目标行.

目标最后一层 FA 对全部必需行执行输入归一化与 K/V 投影. 选中行执行 Q/gate 投影, attention, 输出投影, MLP 和最终归一化. DSpark 特征抽取位于这一层之前. 若配置第 63 层特征抽取, 则保留完整层输出.

持久化 MTP 先归一化目标 hidden 与 embedding 输入, 再对每个必需 context 行执行 FC. 每行提供 K/V, 选中末行提供草稿 hidden 输出. 后续 context 行读取目标 hidden, 因此丢弃的历史草稿 hidden 没有使用者. 临时提议行保留完整输出.

对于 `R` 个 context 行和 `S` 个输出行, QKV Tensor 工作为 `2*5120*(2048*R+12288*S)`. FP8 输入量化按 `R` 行计算一次. 实现中对选中行重复量化的部分计入开销. Q/gate 与 K/V 参数范围使用独立且不重叠的读取身份.

终止请求保留已注册的完整前缀页. 模型移除没有后续使用者的草稿 context 与 hidden 工作. 恢复的 MTP 边界特征按物理页记录具名的 10 KiB 读取. 同一 prefill 阶段内先前产生的特征允许理想的免费驻留. MTP 进入阶段的 KV 读取使用物理位置 `[1,end)`, 按页合并共享范围.

公式版本为 `qwen38-dspark-b200-semantic-v3`. 带类型的 FX 元数据识别形状查询与主机已知的提议页地址. 放松后的实现允许这些元数据操作的必需 GPU 工作为零, 实际执行耗时仍计入墙钟.

## 依赖与共享资源

语义节点保留 activation, token 和 state 依赖.
独立的参数读取可以提前开始.
CUDA stream 顺序不增加语义依赖.

要求整矩阵全部完成的屏障会阻止算子内部的理想流水.
因此, 当前路径分析将所有算术的服务时间放松为零, 包括 GEMM 和 GDN.
每个输入读取节点保留依赖, 并分别获得一份完整且宽松的存储抵扣.
完整算术工作仍计入共享资源下界.
阶段使用各 step 路径的最大值, 让参数预取可以跨 step 与计算重叠.
该路径是一种放松, 并非将整个矩阵操作视为原子步骤后的时延.

阶段下界取放松后的关键路径, 必需 HBM 时间和共享执行资源时间的最大值.
这一最大值允许独立分支重叠, 同时约束它们共享的单个 GPU.
FP8 和 BF16 Tensor Core 时间相加, 因为这些运算使用同一执行资源.
不同执行资源可以重叠.

当请求进入同一区间的时间不同, 模型扣除提交或阶段起点的偏移.
这一修正给出最长单请求区间的下界.

## 硬件上限

配置使用单个 1000 W B200 的官方非稀疏峰值.

| 资源 | 峰值 |
|---|---|
| FP8 Tensor Core | 4.5 PFLOP/s |
| BF16 Tensor Core | 2.25 PFLOP/s |
| FP32 | 75 TFLOP/s |
| FP64 | 37 TFLOP/s |
| HBM | 7.7 TB/s |

参见 [NVIDIA Blackwell 技术说明](https://resources.nvidia.com/en-us-blackwell-architecture) 的表 3.
表中将非稀疏 BF16 舍入为 2.2 PFLOP/s.
模型根据其稀疏峰值使用乐观值 `4.5/2 = 2.25`.
[CUDA 指令吞吐表](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions) 提供已公开的整数, 转换, shuffle 和 SFU 上限.

SM 数量和最高 SM 时钟来自选中的 GPU.
已审阅的完整 B200 配置包含 148 个 SM, 132644864 字节 L2 与 1965000000 Hz 最大 SM 时钟.
采集器另行读取硬件时钟上限, 必须与 worker 记录一致.
其他 SM, 缓存或时钟配置在成本分析前拒绝.
每次重复的下界也必须不超过实测阶段时延, 时间戳允许 1 microsecond 误差.

实测 kernel 吞吐和效率系数不改变这些峰值.

模型允许理想使用 L2, 寄存器, 统一 L1/shared 存储, TMEM 和 constant cache.
每个 SM 为这一宽松的存储抵扣贡献 `(3*256+8)*1024` 字节.
各项分别是寄存器, 统一存储, TMEM 和 constant cache: 256 KiB, 256 KiB, 256 KiB 和 8 KiB.
参见 [SM100 存储](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html#compute-capability-10-0) 和 [TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory).

寄存器与 compute capability 检查验证所选配置.
模型允许该存储跨算子边界保留值.
模型将未公开的片上带宽上限设为无穷大.

执行资源映射尚未确定的 packed BF16/FP8 算术和舍入使用零服务时间.
报告列出这些已识别且被放松的操作.
未知操作会阻止验收.

## 必需 HBM 流量

读取统计使用互不重复的输入 tensor 范围, 并在每个反馈区间使用一次存储抵扣.
Prefill 区间允许跨 chunk 复用参数.
每个 decode 反馈区间获得一次新的理想存储抵扣.
超过这一抵扣的必需参数必须来自 HBM.

`performance/memory.py` 跟踪物理 KV page, recurrent state 和持久草稿数据.
它在接受进度, checkpoint 注册和终止释放之后统计最终存活的写入.
被拒绝的 snapshot 和被覆盖的版本不成为必需的最终写入.
Page 复用及其他请求的写入会使旧物理版本失效.

读取和最终写入的计算使用独立抵扣.
这允许理想的初始及最终 cache 内容.
临时 activation 可以通过理想融合留在片上.
因此模型统计必需流量, 而非观测到的 kernel 流量.

## 证据与判定

`benchmarks/framework.py` 采集十五项固定负载, 每项完整预热两次并测量五次.
每次测量都有根据自身 trace 推导的 prefill 和 decode 下界.
每个阶段须满足 `median(wall) <= 3*median(bound)` 和 `(max(wall)-min(wall))/median(wall) <= 0.10`.
TPS, 接受率, GPU 时间和 cleanup 用于诊断.

采集器检查未变化的已提交源码, 二进制 Rust 构建标记, checkpoint, 已加载 CUDA 模块和稳态 cache 目录树.
经审查的操作契约绑定项目源码, provider 源码, 模板头文件, 构建参数和包版本.
判定记录模型版本和公式源码哈希.
失败尝试的原始记录保存在外部存储.
可移植摘要标明原始哈希和源码适用范围.

算子诊断见 [profiling](profiling.zh.md), 命令见 [testing](testing.zh.md).
