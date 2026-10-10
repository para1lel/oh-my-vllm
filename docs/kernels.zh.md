# 内核开发

CUDA 是 B200 上项目内核的默认后端.
冻结 TileLang 后端作为显式比较对象.
第三方库内部可以使用 Triton.
实测源码身份见 [验收](acceptance.zh.md), 剩余限制见 [审计](audit.zh.md).

## 后端与来源

Python 启动前设置 `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 选择比较后端.
无效后端或缺失 CUDA 实现会报错.
不自动回退到 TileLang.
公共 Python wrapper 保留校验和 factory binding.
冻结实现保留在 `python/oh_my_vllm/kernels/tilelang_reference/`.
不调优参考实现.

`development/kernels/tilelang-reference.json` 绑定以下身份:

| 身份 | 值 |
|---|---|
| 已验收参考记录 | `02bf35f98ceb6b40a2a813aa3e0cccbfa9067600` |
| 内核实现 | `da75c02bd9e01556436877acba3243636dec80d6` |
| 运行依赖 lock SHA-256 | `5f02f2051db2ba92fcdc1864283d4101eba331ed9c9677f3dac627f04c37e15a` |
| TileFoundry fork | `6b1b1493efcb55f3f2f327b5cc90a01181adf690` |

Manifest 还记录每个参考文件的 SHA-256.
无关修改中保持这些文件和运行依赖 lock 不变.
修改 lock / reference 需要显式的新比较决策.

`python/oh_my_vllm/kernels/cuda_backend/kernels.cu` 包含 `operators/` 中的 10 个功能头, 统一构建 native 模块.
TVM FFI 的 `build_inline` 在首次使用时编译这些输入; `load_module` 加载返回的模块.
目标为 `sm_100a`, 使用调用方当前 stream.
包装脚本设置 `TVM_FFI_CUDA_ARCH_LIST=10.0a`; 直接启动时必须显式设置.
Sidecar 绑定源码 / 配置和精确模块 SHA, nvcc 不可用时支持验证后的复用.
Capture 和正式计时前完成编译.
TileFoundry 不属于 native 构建或服务依赖.

### Native 模块划分

| `operators/` 中的模块 | 职责 |
|---|---|
| `common.cuh` | PDL, 启动, 错误检查, 存储检查, 计数器, 归约和向量访问. |
| `quantization.cuh` | FP8 分组量化和 SiLU 乘法. |
| `normalization.cuh` | 目标模型和 DSpark 的 RMS, 残差 RMS 及 gated RMS. |
| `attention_prepare.cuh` | Q/K 准备, RoPE, 连续行复制及目标 / 草稿 KV 写入. |
| `gdn.cuh` | GDN gates, Q/K 归一化和递归状态更新. |
| `convolution.cuh` | 因果卷积和候选状态写入. |
| `attention_math.cuh` | Softmax 辅助函数, shared-memory 地址计算和 BF16 MMA 辅助函数. |
| `decode_attention.cuh` | 目标模型的 split-KV 注意力及稳定合并. |
| `dspark_decode.cuh` | 草稿注意力读取已提交上下文及具有双向注意力的 7 行候选块. |
| `normalized_quantization.cuh` | 融合残差 / gated RMS 与 FP8 量化, 保留必要的 BF16 舍入点. |

统一构建使导出函数与设备定义保持在同一模块中.
Native 模块将全部功能头摘要绑定到缓存键和加载来源记录.
GEMM 模块独立绑定 `groupwise_fp8.cu` 及根目录的模板头.
根目录的模板头包括 `groupwise_accum.cuh`.
构建和加载检查拒绝变化的源码或头文件.
正式阶段和 profiler 检查将加载的头文件身份与已审查源码比较.

自有 CUDA 后端的每个 `.cu`, `.cuh` 和 `.py` 文件至多 800 行.
`clang-format` 23.1.3 使用 `.clang-format`, 保留 include 顺序.
Pre-commit 钩子格式化 native 文件并检查模块大小:

```bash
scripts/with-env.sh python scripts/format_cuda.py
scripts/with-env.sh python scripts/format_cuda.py --check
```

## 合同与实现

| 算子 | 当前特性 |
|---|---|
| FP8 | Four-element vector, warp REDUX, 精确 scale division 和 packed store. |
| BF16 finite FP8 scale reciprocal | `rcp.approx`, 一次 residual correction 和两次 `__fmaf_rn`. |
| 其他 FP8 reciprocal | Nonfinite maxima, FP16 和 FP32 使用精确 `__fdiv_rn`. |
| SiLU/multiply | Fast exponential 和 paired BF16 multiplication 保留两个 BF16 舍入点. |
| RMS | Dense width5120 使用对齐, 寄存器内 specialization, epsilon `1e-6`. |
| Gated RMS | 48 head, width128, stable sigmoid, FP32 gating, 无额外 BF16 舍入点. |
| Q/K | Packed 16-head row, token stride10240, 对齐 eight-element load. |
| Convolution | 10240 channel, token stride16384, weight alignment 保护和 source-state snapshot. |
| GDN | 受保护的 vector state update, FP32 arithmetic, BF16 MTP 或 FP32 ordinary storage. |
| Attention preparation | Q/K RMS/RoPE, V conversion 和 KV write, 使用 FP64 phase reduction. |
| Decode attention | BF16 `mma.sync`/`ldmatrix`, FP32 accumulation, sector swizzling, split-KV 和独立 merge. |

其他支持的 layout 使用 generic CUDA 路径.
Fast-path guard 包括 alignment, stride, shape 和 disjoint-storage 检查.
所需 snapshot 完成后, state update 可以合法覆盖自身 source.
直接 recurrence FFI 要求连续 `int32`/`int64` index tensor, 单调 starts, 合法 reads, 以及合法 writes 或 `-1`.
不在每次 launch 将 GPU index 内容复制回主机.
Batch plan 按报告容量校验逻辑 slot.

区分 fresh-output non-aliasing 与 in-place state/cache 合同.
无效 wrapper 输入在 wrapper 或 C++ 边界拒绝.
负 attention slot 跳过 KV write, 保留 Q.
原生 MTP4 verification 最多 5 个 query, DSpark verification 最多 8 个.
Production wrapper 通过复制对齐未对齐 KV view, 并报告该成本.
Clone counter 和故障归因见 [profiling](profiling.zh.md).

Reduction order 可随 row 数量和 alignment 变化.
应用 FP64-reference 容差; RMS bitwise batch invariance 不属于合同.
Gated-RMS sweep 覆盖 input1, weight0.25 时所有 finite BF16 gate.
没有穷尽全部 input/weight pair.

Finite BF16 maxima 的精确 scale 位于 `[2.232142829e-13,7.565917940e35]`.
Reciprocal 近似范围为 `[1.321716729e-36,4.480000066e12]`.
Scale 和 reciprocal 保持 FP32 normal value.
[PTX reciprocal 规则](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp) 指定近似指令误差界限.
B200 production 回归覆盖 65280 个 finite BF16 encoding, 32640 个 finite maxima, 有符号 midpoint 派生样本和 nonfinite fallback.
还覆盖 maximum448 时 126 个正 / 负 FP8 midpoint.
这些测试的最终 FP8 字节和 scale 与 CPU RN 参考一致.
此有限测试不穷尽全部 pair, 也不证明中间 quotient 相同.

## 正式用例与计时

`development/kernels/cases.py` 在 12 个工作负载内静态推导并去重每个路径的最大合法调用.
FA fixture 排除 null page0.
完整操作包括 snapshot, copy 和 split merge.
`development/kernels/semantics.py` 描述逻辑行为; `twins.py` 绑定冻结的完整操作.

DSpark 为 target verification 和草稿算子增加静态补充 shape.
原用例 ID 和 8 个冻结 TileLang 文件保持不变.
`development/kernels/dspark-reference.json` 绑定补充参考及其原始 manifest 哈希.
`dspark_tilelang_reference.py` 提供不同的 DSpark 性能比较对象.
独立 FP64 PyTorch 参考提供数值比较.
新用例包括 BF16 hidden RMS 舍入, head-128 YaRN Q / K preparation, context append 和 7-row 草稿 attention.
DSpark hidden RMS 使用 vector load 和成对 BF16 乘法.
每次 normalization 和 multiplication 保留规定的 BF16 舍入.
Q / K preparation 在 register 中保留 rotary 的两个半部, 每对 sine / cosine 只计算一次.
小规模调用每个 head 使用 64 个线程, 大规模调用每个 head 使用一个 warp.

Context append 对地址对齐且 storage 互不重叠的 tensor 使用 128-bit load / store.
容量检查选择 32-bit 或 64-bit 地址除法.
Kernel 在除法前检查 slot 范围.
Kernel 在计算缓存地址前读取源数据.
其他 layout 使用 CUDA `append` 操作.

相同 K / V 和 slot 的重复 context append 写入相同字节.
计时阶段的两个操作使用相同目标缓存, 消除缓存地址差异.
数值校验在计时前使用独立缓存, 最初 147 项用例保留其数值 fixture 契约.

每个新用例沿用完整输出, cache write, 3 轮, 20 个配对和置信下界门槛.
源码身份及测量范围见 [验收索引](acceptance.zh.md).

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --output "$EVIDENCE_DIR/operators.json"
```

Graph 内部的 external CUDA Event 包围全部 100 次完整操作.
计时区间排除 graph replay 前后的主机提交停顿.
采集器等待 graph 结束 event 后才读取耗时.

无状态 Q/K 计时先做独立数值校验, 再使用共享 graph pool 和 capture stream.
两个完整函数保留各自的两次输出分配. 串行 replay 之间不使用输出值.
采集器检查当前输出与上一次调用输出存储的 alias.
它检查全部 100 次调用的对应输出地址一致, 输出与输入存储范围不重叠.

计时结束后还检查全部输入存储字节. Graph 和 stream 保留到最终同步完成.

DSpark append 以外的可变用例沿用不同目标 pool.

使用仓库外可写的 `EVIDENCE_DIR`.
`--operations` 选择诊断子集; 不完整覆盖或 dirty source 不能通过完整矩阵.
GPU 干扰和源码变化使采集失效.
采集器记录源码, 模块, hardware, 完整输出校验, 全部样本和单独标识的估算.

计时前在现有容差下比较完整返回值和实际写入缓存.
包括 FP8 scale stride 和逐 slot recurrent 界限.
恢复 mutable state, 预热两条路径.
每个用例需要 3 轮, 每轮 20 个交错配对, CUDA 每轮中位数更快.
平均节省时间的单侧 95% hierarchical-bootstrap 下界必须为正.
不设最低百分比增益, 不新增相对 TileLang 的框架比率.

以下用例的正式输出门槛要求 1 次 fast, 0 次 generic host dispatch:

| 算子 | Fast-path 输入 |
|---|---|
| `norm` | Dense nongated5120 row, 对齐 input/weight. |
| `add_norm` | Dense5120 row, 对齐 input/residual/weight. |
| `gated_norm` | 48 个 head, 每行宽度 128, 使用 gated RMS. |
| `qk` | 打包的 16-head 行, token stride 为 10240, Q/K 对齐. |
| `recurrent` | 16 个 Q head, 48 个 V head, Q/K/state 对齐, state pool 不重叠. |
| `convolution` | 10240 个 channel, token stride 为 16384, weight 对齐, state pool 不重叠. |

`variant_launch_counts()` 测量 host dispatch, 不统计 graph replay.
`append` 不属于正式矩阵, 有独立 fast/generic GPU 测试.
其 fast path 要求 width 可被 8 整除, K/V/cache 对齐.

## 分析与 profiling

按 [开发指南](development.zh.md) 安装固定 TileFoundry 工具.
Fork 支持 HIR sin/cos evaluation, printing 和 cost classification, 不生成 device code.
使用兼容的 released TileLang 和 OR-Tools 版本; 不维护这些库的本地 fork.
Editable TileFoundry 源码必须持续可用.

```bash
scripts/with-env.sh tilefoundry analyze development/kernels/semantics.py:Operators.silu "$EVIDENCE_DIR/silu-analysis.txt" --compute-cost --memory --roofline
scripts/with-gpu.sh scripts/with-env.sh python -m development.kernels.observations --case attention-c426ebd5d5 --output "$EVIDENCE_DIR/observations.json"
```

HIR arithmetic, memory 和 roofline 输出是估算.
分析 topology 不等于实际 device placement.
Observation 工具通过 Nsight Compute 分别 profile 完整后端操作.
Warmup 和 JIT 位于 NVTX range 外.
缺失或非有限 requested counter 会使 observation 失败.
所需 counter 包括 local-memory 读取和写入的 sector 数, 用于识别寄存器 spill 流量.
报告分别标识估算, counter, resource 和源码身份.
不提供验收计时样本.

固定 HIR 无 FP64 dtype.
其 FP32 RoPE phase 不能证明最大上下文精度.
应使用独立 FP64 和模型测试.
原始 profiler CSV 和临时调优放在 Git 外.

## 修改流程

1. 阅读 wrapper, 合同, 测试和相关审计项.
2. 保留支持的 shape, stride, dtype, alias rule, state output 和舍入边界.
3. 用分析选择假设, 再使用合法 metadata 测量完整操作.
4. Production dispatch 保持确定; 不做运行时 autotuning 搜索.
5. 完成受影响的正确性和正式算子用例.
6. 声称性能验收仍成立前, 完成受影响框架工作负载.
7. 提交带原始哈希的可移植证据; 临时记录放在仓库外.
8. 完成独立审查, 停止所属 GPU 程序.
