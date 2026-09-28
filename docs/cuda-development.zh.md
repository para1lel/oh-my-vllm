# CUDA kernel 开发

[English](cuda-development.md)。agent 以英文版为准。

## 状态

CUDA 是 B200 上默认的自定义 kernel 后端。[验收文档](acceptance.zh.md)中的
147/147 个算子用例和 12/12 个框架行仅适用于历史干净提交 `c36d1c9`；审计整改后的
当前完整矩阵与框架验收仍待重跑。已知的 kernel 契约与可观测性问题列在
[audit-2026-09-23.md](audit-2026-09-23.zh.md) 的 `KRN-*`、`MNT-*` 和 `EVD-09` 条目下。
修改 kernel 前请先阅读这些内容。

## 后端选择与冻结参考实现

- **选择后端。** 后端在每个进程导入时选定一次。在 Python 启动前设置
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 会选择冻结的对比后端。无效名称会抛出错误。运行时身份报告的是实际选中的后端。
- **无静默回退。** 缺失的 CUDA 入口会抛出 `NotImplementedError`，绝不回退到 TileLang。
- **冻结源码。** 已接受的 TileLang 源码逐字节复制到 `python/oh_my_vllm/kernels/tilelang_reference`。
  以下信息记录在 `development/kernels/tilelang-reference.json` 中：
  - 它们的哈希
  - 原始提交
  - 依赖锁文件哈希
  - TileFoundry pin

  绝不调优参考实现。
- **kernel 所在位置。** 生产模块包含校验、公开 wrapper 和显式的工厂绑定。TileLang kernel 主体只存在于
  `tilelang_reference/` 中。
- **原生构建。**
  - 原生代码为 `python/oh_my_vllm/kernels/cuda_backend/kernels.cu`。它通过独立的 TVM FFI
    `build_inline` 延迟编译，再用 `load_module` 从返回路径加载。目标为 `sm_100a`，
    在调用者当前的 CUDA stream 上运行。
  - `scripts/with-env.sh` 隔离 `TVM_FFI_CACHE_DIR` 并设置
    `TVM_FFI_CUDA_ARCH_LIST=10.0a`；直接启动时须显式设置该架构。构建 sidecar 绑定
    源码/配置及准确的 `.so` 哈希，供后来无法查询 nvcc 时经验证复用缓存。
  - 由主机 CUDA 13.1 编译。需要时使用 `CUDA_HOME=/usr/local/cuda-13.1`。
  - 所有编译必须在 CUDA Graph 捕获和正式测量之前完成。
  - TileFoundry 不是运行时或构建依赖。

## 实现说明

以下描述当前代码及其所依赖的数值性质。

- **FP8 量化。**
  - 行和缩放组按 tile 划分。kernel 使用四元素向量，并对非负 FP32 幅值使用无符号 warp REDUX。
  - 融合 SiLU 的加载是打包的，FP8 存储也是打包的。
  - 扁平 grid 处理较小的融合行，以及宽度超过 `grid.y` 的情况。
  - 对于有限的 BF16 最大值，缩放倒数使用 `rcp.approx` 加一次 FMA 残差校正。缩放本身仍使用精确除法。
  - 在 SM100/CUDA 13.1 上对所有有限 BF16 输入/最大值对进行的临时穷举检查未发现任何 FP8 差异。
    2026-09-23 审计中的精确算术复核结论一致，前提是 `rcp.approx` 误差不超过 1 ulp。
- **SiLU 与乘法。**
  - 使用快速指数和成对 BF16 乘法。两个 BF16 舍入点均保留。
  - 在 SM100/CUDA 13.1 上，对每个有限 BF16 输入以及所有有限操作数对的穷举检查均未发现差异。
  - 扁平偏移为 32 位（审计 KRN-04）。
- **RMS 归一化。**
  - 模型宽度为 5120。值保存在寄存器中，加载为宽度 4 或 8 的对齐向量。
  - 线程数取决于行数以及是否加残差：≥ 4096 行时使用 320（残差）或 160 个线程；
    2048–4095 行使用 128；更少的行使用 256。
  - 行数在 2048–4095 之间的残差 RMS 使用流式存储提示。
  - 未对齐或其他布局使用通用 kernel。
  - 因此归约顺序随行数和对齐情况而变化（审计 MNT-03）。
- **Gated RMS。**
  - 针对 48 个 head × 128 宽度特化。值保存在寄存器中。
  - 使用分母在 [1,2] 内的稳定 sigmoid，门控用 FP32 计算，无中间 BF16 舍入。
- **Q/K 归一化。**
  - 打包的 16 head、10240 stride 路径使用对齐的八元素加载和成对 FP32 运算。
  - 其他布局使用通用 kernel。
- **卷积。**
  - 针对 10240 通道、token stride 16384 特化。使用四元素权重加载和稳定 sigmoid。
  - 中等输入每个 block 处理 4 行，大输入处理 8 行。
  - 权重对齐检查和保守的池/输入区间不相交检查守护特化路径。
  - 在写入候选之前对源状态做快照。
- **GDN 递推。**
  - 模型布局让 16 个 lane 各持有八个连续的 key 值，使用四个 warp、每个半 warp 处理两行 value，并保持 FP32 持久状态。
  - head、基址/行对齐以及存储不相交检查守护 restricted 指针，否则运行通用 kernel。
    对请求自身源状态的原地更新是有效的。
- **Full-attention 预处理。**
  - 一个入口融合 Q/K RMS 与 RoPE、V 布局转换以及 KV 写入，由目标模型和 MTP 共用。
  - 保留两个 BF16 舍入点，并在 FP32 sin/cos 之前以 FP64 对相位做归约。
  - 负槽位只跳过 KV 写入。越界槽位也会被静默跳过（审计 KRN-02）。
- **Paged decode attention。**
  - 使用显式的 `mma.sync`/`ldmatrix` BF16 fragment 并以 FP32 累加，采用共享内存 sector swizzling，
    以及带独立合并步骤的 split-KV。
  - Q/K 与 P/V fragment 在各输出 tile 之间复用。没有 local memory 溢出。
  - 非分组 KV 双缓冲限于四个 query。
  - 分组验证假设每个请求至多五个 query 行（审计 KRN-01）。
- **其他性质。**
  - 产生新输出的 kernel 声明不别名（`__restrict__`）指针；原地更新 state/cache 的 kernel 不声明。
  - 连续的 BF16 视图可能从未对齐的存储偏移开始。原生 attention 处理标量 Q 加载；生产包装层
    对两个后端都对齐 KV，只对冻结的 TileLang 对齐 Q（KRN-07 涵盖 KV 克隆）。
  - attention merge 的归约顺序在调优中发生了变化；结果满足容差，但与早期版本并非逐位一致。
  - gated-RMS 门控扫描以固定输入 1 和权重 0.25 覆盖了每个有限 BF16 门控值；并未穷举
    输入/权重组合。
  - TileFoundry 的 `TileLang` 孪生实现总是绑定冻结实现，与所选的生产后端无关。

## 正式算子对比

正式静态矩阵为 `development/kernels/cases.py`。按照 REQ-KERNEL-002 的要求，它以静态、去重的方式
推导出每个 workload 下每条不同实现路径的最大合法调用。完整操作 fixture 保留所需的快照与合并。
有效 FA 页不包括空页 0。

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python benchmarks/kernels.py --output /tmp/cuda-kernels.json
```

- **子集。** `--operations` 选择诊断子集。部分覆盖和脏源码运行不能通过完整矩阵门槛。
- **收集器记录的内容：** 源码哈希、GPU 身份、全部成对样本、判定结果，以及单独标注的 TileFoundry 估计。
- **失效。** GPU 争用或源码变更会使一次收集失效。
- **判定规则**（`development/kernels/comparison.py`）：
  - 至少三轮、每轮 20 对，后端顺序交错。
  - 两条路径都已 warmup，且每次运行之间恢复可变状态。
  - 每一轮中 CUDA 的中位数都更快。
  - 节省时间均值的单侧 95% 分层 bootstrap 下界大于零。
- **harness 比较的内容。** 它只比较时间（审计 EVD-09）。数值一致性来自 FP64 测试套件。

## 开发观测

```bash
scripts/with-gpu.sh scripts/with-env.sh env CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python -m development.kernels.observations --case attention-c426ebd5d5 --output /tmp/kernel-observations.json
```

- **运行内容：** TileFoundry HIR 分析，以及对两个后端完整操作分别进行的 Nsight Compute 重放。
  warmup 和 JIT 位于 NVTX 范围之外。
- **必需计数器：** 请求的计数器缺失或非有限值会使运行失败。
- **报告内容：**
  - 分别标注的 HIR 估计与实测计数器
  - 寄存器和共享内存资源
  - 源码哈希与硬件身份
- **不是性能门槛：** 该报告不能通过任何性能门槛。
- **清理：** 原始 CSV 文件是临时的。取消操作会终止所属的 profiler 进程组。
- **HIR 局限：** 固定版本的 HIR 不支持 FP64，无法证明长位置 RoPE 相位的精度。这一点请使用 FP64 参考和完整模型测试。

## 修改检查清单

1. 保持 wrapper 契约和文档记载的舍入点。新的前置条件应放在 wrapper 或 C++ `ICHECK` 中，而不能只放在上游调用者中。
2. 在容差不变的情况下运行完整 GPU 测试套件，并运行实际模型的普通/MTP4 FP64 探针。
3. 重新运行每个实现发生变化的正式算子用例。脏源码子集只能作为诊断报告。
4. 在声称 12 行结果仍然成立之前，重新检查受热路径影响的框架行。
5. 在 `bench/baseline` 中记录汇总证据。临时脚本和原始 trace 保存在仓库之外。停止你启动的所有 GPU 程序。
