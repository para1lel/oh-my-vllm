# GPU 峰值校准

对每个 GPU UUID 使用实测资源吞吐.
算子和 HTTP roofline 计算要求使用匹配的校准文件.
不回退到标称峰值.

## 命令

将 `GPU_UUID` 设为设备清单中的一个 UUID.
将 `PEAK_PROFILE` 设为仓库外一个新的 JSON 文件.
按[算子工具设置](operators/development.zh.md#工具)配置 `CUDA_HOME` 和 `OH_MY_VLLM_KERNEL_CACHE`.

```bash
nvidia-smi --query-gpu=uuid,name --format=csv
export OH_MY_VLLM_GPU_UUID="$GPU_UUID"
scripts/with-env.sh scripts/with-gpu.sh python -m development.peaks --output "$PEAK_PROFILE"
```

对每个可用的 GPU UUID 重复此命令, 使用不同的输出文件.
GPU wrapper 为每条命令预留一张空闲 B200.
CUDA Driver UUID 必须与 wrapper 的选择一致.
后续算子或 HTTP 测量使用此 UUID 及其文件.
一次算子审计不能混用不同 GPU 的校准记录.

## 资源测量

| 资源 | 方法 | 配置 | 每次调用的工作量 |
|---|---|---|---|
| HBM | 使用伪随机输入的 CUDA 向量复制 | 每个 buffer 为 1 或 2 GiB; 每个 SM 使用 4, 8 或 16 个 block | 读取字节数加写入字节数 |
| FP8 | Dense cuBLASLt TN GEMM; E4M3 输入; FP32 累加及输出; 标量 scale 为 1 | 方阵维度 4096, 8192, 16384; 最多八种启发式算法 | `2 * M * N * K` FLOPs |
| BF16 | Dense cuBLASLt TN GEMM; BF16 输入; FP32 累加及输出 | 相同的 GEMM 配置 | `2 * M * N * K` FLOPs |
| TF32 | Dense cuBLASLt TN GEMM; FP32 输入; `CUBLAS_COMPUTE_32F_FAST_TF32` | 相同的 GEMM 配置 | `2 * M * N * K` FLOPs |
| FP32 | 原生标量 FMA 指令 | 16 个独立累加器; 4096 次迭代; 每个 SM 使用 2, 4 或 8 个 block | 每次 FMA 计两个 FLOPs |
| FP64 | 原生标量 FMA 指令 | 相同的标量配置 | 每次 FMA 计两个 FLOPs |

所有自定义 kernel 的每个 block 使用 256 个线程.
每个复制 buffer 必须大于 L2 容量的四倍.
存储的结果包含每个累加器的值.
构建检查要求二进制中存在原生 FP32 和 FP64 FMA 指令.
输出样本检查必须无错误.

每种配置在测量前至少预热 200 ms.
七个样本各用 CUDA events 在同一 stream 上包围十次调用.
分配, 输入准备和算法选择均在计时前执行.
按 `elapsed_milliseconds / 10000` 计算每次调用的秒数.
对每种资源, 在全部配置中选择最高的 `work / median(seconds)`.
不选择单个最快样本.

## 身份与计算

仓库外的文件保留每种配置, 工作量和计时样本.
它记录 UUID, GPU 属性, 驱动, 显存时钟, 功率上限和 SM 时钟快照.
它绑定校准源码, 编译选项, 原生二进制以及实际加载的 cuBLASLt image.
使用方检查记录的工作量, 并重新计算所选吞吐.
峰值缺失, 源码或二进制变化以及 UUID 不同均阻止测量.
驱动, 显存时钟或功率上限变化后必须重新校准.

这些吞吐描述实测负载及运行条件.
它们不证明所有可能负载的最大吞吐.
Latency 下界是基于这些实测吞吐的 roofline 估计.
算子可能超过校准吞吐. 保留超过 100% 的效率供分析.
不根据算子或 HTTP 的实测延迟拟合资源吞吐.

使用[算子公式](operators/evaluation.zh.md#性能计算)及 [HTTP trace 计算](performance.zh.md#下界计算).
算子效率没有最低要求. HTTP 效率要求仍为每种配置达到 50%.
每次交付前重新测量算子.
然后更新两种语言的 README 表格.
校准文件和原始计时报告保存在仓库外.
