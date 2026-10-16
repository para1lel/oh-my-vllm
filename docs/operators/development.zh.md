# 算子开发

使用[算子规格](model.zh.md), [fusion 计划](fusion.zh.md)和[评测流程](evaluation.zh.md).
原生 provider 提供全部七个 serving 边界. 完整负载验收是必要的.
推理 worker 加载原生模块. 占位实现仅用于 `--model stub`.

## 工具

使用 CUDA 13.1, Nsight Systems 和 Nsight Compute.
工具路径保存在被忽略的 `LOCAL.md` 中.
将 `CUDA_HOME` 设为 CUDA 安装目录.

安装算子开发包:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

推理绑定包位于 `requirements/kernels.txt`.
开发包不是 worker 依赖.

将 `CUTLASS_ROOT` 设为仓库外的依赖目录.
获取指定版本的 CUTLASS 头文件:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

构建只接受来自干净依赖 checkout 的 Git 修订版本 `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
将 `OH_MY_VLLM_KERNEL_CACHE` 设为仓库外的构建目录.

构建原生模块:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels build
```

推理加载器使用这些模块, 不使用编译器或开发工具.
加载器拒绝与其 manifest 不一致的源码或二进制标识.
每个二进制文件在外部 cache 中都有不可变副本及构建 manifest.

## 评测命令

获取负载计划:

```bash
scripts/with-env.sh python -m development.kernels plan
```

执行 CPU 契约检查:

```bash
scripts/with-env.sh python -m unittest development.kernels.test_contracts development.kernels.test_reference
```

将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `ARTIFACT_ROOT` 设为仓库外的产物目录.
测量 FFN provider:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode graph --output "$ARTIFACT_ROOT/ffn-graph.json"
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode eager --output "$ARTIFACT_ROOT/ffn-eager.json"
```

获取单个 shape 的 Systems trace 和 Compute counters:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels profile --model "$MODEL_ROOT" --operator ffn --rows 1 --mode graph --output "$ARTIFACT_ROOT/ffn-profile"
```

执行命令前, 选择尚不存在的 profile 目录路径.
Profiling 不提供验收计时.

使用 `--operator` 选择 `ffn`, `fa_attention`, `gdn_attention`, `vision_block`, `sample`, `mtp_context` 或 `mtp_draft`.
使用 `--rows` 选择部分建模计时行数.
此选项不改变[小型数值用例清单](evaluation.zh.md#数值范围).
每次 benchmark 都在计时前使用两种执行模式对此清单进行检查.
子集不足以满足算子覆盖率门槛.
负载不完整或 provider 失败会阻止验收.

为每个算子和执行模式提供一份来自同一组冻结源码的完整报告.
使用报告路径执行完整验收审计:

```bash
scripts/with-env.sh python -m development.kernels audit --reports "$ARTIFACT_ROOT"/suite-*.json --output "$ARTIFACT_ROOT/acceptance.json"
```

审计对当前源码, 不可变二进制, checkpoint 身份, 冷权重, 小型数值用例及全部建模计时 shape 进行检查.
版本 2 报告将数值证据与计时覆盖分开.
大型计时用例将数值状态记为 `not_run`. 此状态不会从性能覆盖中移除该 shape.
源码变更后, 旧报告不再符合验收条件.

对于每个选定的数值用例, FFN 检查会确认所有输出元素都是有限值.
所有残差元素都必须满足 BF16 精确相等.

FFN 的 FP64 投影参考比较首行, 中间行和末行的全部列.
每个 provider 在报告中注明其数值抽样范围.

GDN 检查全部输入投影及私有状态写入. Sampling 检查所有行和词表列.
FA 和 MTP 检查 KV 写入及未改动的保护存储. Vision 检查原始 patches, 输出抽样和帧隔离.
这些检查不为所述范围以外的元素提供容差证据.

## 方法参考

将外部实现作为设计参考. 遵守[运行时库规则](fusion.zh.md).
递推, value tile 和序列分区采用 [GDN 执行计划](gdn.zh.md).

[CUTLASS 分块示例](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu)使用 E4M3 操作数和 FP32 缩放因子.
当前的 [DeepGEMM SM100 FP8 接口](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md)仅使用打包的 UE8M0 缩放因子.
不得为了使用另一种矩阵指令而改变 checkpoint 缩放值.

[FlashAttention SM100 实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py)分离加载, 矩阵乘法, softmax 和输出修正.
其分页实现提供不使用 TMA 的加载路径.
对 head 宽度为 256 的输入, 比较此调度与当前 FA 实现.
保留 784-token cache page 和 BF16 概率存储.

[CUTLASS TMEM 复制指令](https://github.com/NVIDIA/cutlass/blob/v4.8.0/include/cute/arch/copy_sm100.hpp)为已知行数提供不同加载宽度.
保留所有参与的 lane 及必要的目标寄存器.
已知行数可以移除未使用的尺度及累加器计算.

对于 [FlashInfer GDN prefill 接口](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py), checkpoint 间隔必须可被 64 整除.
其 [Blackwell 实现](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py)将 live state 转换为输入类型后执行矩阵乘法.
使用其算法前, 对照模型契约检查这些条件.

[NVIDIA FP32 仿真](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation)将每个操作数分解为三个 BF16 分量.
它组合九个乘积. 对于 GDN chunk 变换, 比较此方法与原生 FP32 乘积.
保留 FP32 live state 和 784-token BF16 checkpoint.
额外的实现工作不能增加必要工作量的下界.

[固定版本的 vLLM sampler](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/v1/sample/sampler.py) 在 top-k 和 top-p 处理前使用 `argmax` 选择贪心 token.
比较贪心 `argmax` 归约与完整词表排序.
在 token 选择前应用 mask 和 penalty.
为随机行保留所需的随机分布.

## 优化流程

在 GPU 实验前, 使用 [TileSight](https://github.com/tile-ai/TileSight) 比较 tile 调度方案.
其[模型局限](https://arxiv.org/html/2607.22432v1#S7)包括小批量延迟和 B200 内存拓扑.
将模型与 kernel 的数据类型, 布局, 依赖及物理内存事务进行比较.
包含 TMEM 分配, 带 mask 的加载和存储, 部分结果及合并操作.
检查候选排序是否随不确定的延迟参数改变.
使用数值检查和完整边界测量选择实现.

使用以下经验选择另一个候选:

- 比较 K 分区时包含合并工作. 只有投影变快并不足够.
- 将融合投影扩展到更多行前, 检查寄存器和共享内存用量.
- 将归一化移入投影 block 时, 包含重复的输入读取.
- 当本地归一化提供 shared-memory 矩阵输入时, 保留原有归约顺序.
- 在 cluster 融合实验之前, 用编译后的 kernel 及其完整 launch 配置查询 active cluster 数.
- 比较输出融合与独立 launch 时, 计入 cluster 同步及调度限制.
- 行复制已有足够并行工作时, 限制列 block 数量.
- 移除复制前, 检查缓冲区生命周期及 graph 输入.
- 按数据类型合并固定 metadata 数组. 保持视图对齐和 graph 地址固定.
- 从 TMA 字节计数, 矩阵乘积及累加器读取中排除无效 K 组. 零缩放因子不能阻止 NaN 传播.
- 使用不带 profiler 的独立进程进行对比计时.
- HTTP 对照期间不并行运行参考计算. 保留实测的推测接受轨迹.

每次实验使用仓库外的临时目录.
保留当前报告, 直到该实验的审查及适用的验收审计完成.
随后删除原始报告, profiler 文件及过时的实验二进制.
保留当前运行时二进制及其 manifest.
在本指南中简短记录可复用经验. 不在项目文档中记录测量结果.
