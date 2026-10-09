# Profiling 与诊断

用测量定位 prefill 与 decode 时延差距.
Profiling 与正式计时分开.
TileFoundry 的 arithmetic, memory 和 roofline 结果是估算.
使用 [需求](requirements.zh.md) 中的阶段时延协议.

## 日志与关联

Rust 使用 `RUST_LOG`; Python 使用 `OH_MY_VLLM_LOG_LEVEL`.
默认日志包含 UTC 时间和 run 身份, 不做逐步 I/O.
Debug 日志增加 request / step 关联和主机时长.
主机执行时间不等于 CUDA 内核时间.
Device 归因使用 CUDA Event 或 profiler trace.
原始日志和 trace 放在跟踪内容外.

## 稳态审计

正式采集检查 FlashInfer, 第三方 Triton, TileLang 和 native CUDA cache tree, 包括文本文件.
所需根目录未设置或不存在时审计失败.
测量重复前完成编译和 capture.
源码和 cache hash 绑定观察到的运行.
审计无法排除静默内存内重新编译, 或进程轮询之间极短干扰.
参见 [测试](testing.zh.md) 和 [验收](acceptance.zh.md).

## CUDA 故障归因

默认项目内 launch 使用 `cudaPeekAtLastError`, 不清除先前 runtime error.
异步失败可能在后续 launch 或 host synchronization 才出现.
消息标明观察点, 不证明故障指令.

诊断进程在 Python 启动前设置两个 flag:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_CUDA_DEBUG_SYNC=1 OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_KERNEL_BACKEND=cuda python -m tests.gpu_cuda_error_case eager
```

Debug 模式在每次项目内 launch 前后检查已有错误并同步所选 stream.
拒绝 graph capture.
Prior/after-launch 标记观察顺序; 更早的异步故障仍可能在这里出现.
用兼容 CUDA 安装中的 `compute-sanitizer --tool memcheck` 获取 device 细节.
Debug synchronization 时间不能作为验收测量.
参见 [CUDA error API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html).

## 复制与内核观察

未对齐的连续 BF16 KV view 使用可观察的对齐复制.
`oh_my_vllm.kernels.decode_attention.unaligned_cache_clone_count()` 报告调用次数.
Warning 在次数 1, 2, 4, 8 和后续 2 的幂时报告复制字节.
对齐 view 不进入 clone 路径.

`python -m development.kernels.observations` profile 完整 CUDA 和冻结 TileLang 操作.
报告区分 HIR 估算, Nsight counter 和 resource.
校验冻结参考身份及所需 counter 完整性.
Warmup/JIT 保留在 NVTX range 外.
原始 profiler CSV 放在 Git 外.
取消时停止所属 profiler 进程组.
命令和限制见 [内核开发](kernels.zh.md#分析与-profiling).


## 大型算子调优

先使用模型的有效尺寸, 再搜索 tile.
记录时延, DRAM 与 L2 吞吐, L2 命中率, 寄存器数量, shared storage, occupancy 和 warp 等待.
用 HIR 估计选择假设, 再用 CUDA Events 与 Nsight 计数器验证.
Occupancy 增加本身不能说明时延改善.

Swizzle 选择使用五轮随机顺序计时.
SM 选择与后续 swizzle 确认使用十轮随机顺序计时.
候选必须保持相同的舍入, scales 与持久写入.
原有 230 项, 十六项融合投影和二十八项其余 FP8 投影用例用于算子验收.
完整模型阶段采集用于框架验收.

| 优化 | 诊断观察 | 选择 |
|---|---|---|
| FP8 gate/up GEMM 遍历 | 在 32144 by 34816 by 5120 时, DRAM 为 75.04%, L2 命中为 26.10%. Swizzle 8 将时延中位数从 9.235 改为 6.989 ms. | M 至少 2048 且 N 至少 32768 时使用 swizzle 8. |
| 小型 FP8 gate/up GEMM | M 为 624 时, one-SM 约 0.141 ms, two-SM 约 0.170 ms. | 宽投影在 M 至少 4096 时使用 two-SM tile. |
| 四通道卷积 | 32144 行时, scalar 为 1.603 ms, vector 为 0.624 ms. Warp 指令从 1.484 billion 减为 0.589 billion. | 每线程四个相邻通道; 大型输入八行, 较小输入四行. |
| SiLU 与 FP8 量化 | 32144 行时, 单行版本为 1.130 ms, 四行版本为 0.711 ms. 624 行时, 两行版本为 0.0165 ms. | 大型输入每 warp 四行; 较小模型输入两行. |
| Gated RMS | 32144 token 行时, 单行版本为 0.399 ms, 四行版本为 0.310 ms. | 大型输入每 warp 四行; 小型输入单行. |
| 残差 RMS 与 FP8 量化 | 在 32144 by 5120 时, 分开的操作为 0.325 ms, 融合操作为 0.195 ms. | 保留残差与归一化 BF16 舍入, 删除归一化中间值的读写. |
| 单行归一化投影 | 单行 gate/up 的诊断时延在融合时约为 0.03431 ms, 在之前的 CUDA 链中为 0.03319 ms. | 单行使用之前的 CUDA 链. |

表格使用诊断算子微基准与独立 profiler replay.
短输出完整模型诊断属于不同范围.
它们不提供十五项阶段验收的判定.
卷积 vector 路径的 shared-memory bank conflict 为零.
其 DRAM 吞吐为 15.80%, SM 吞吐为 83.38%, short-scoreboard 等待为 2.43%.
该 profile 支持指令与 shared-memory 开销减少的解释.

残差 RMS/量化候选测量使用十轮随机顺序计时, 每个计时样本重放图四十次.
FP8 字节, scales 和残差和与分开的操作相同.
融合 profile 使用每线程 40 个寄存器与 64 字节的静态 shared storage.
其 DRAM 吞吐为 46.34%, SM 吞吐为 74.66%, occupancy 为 57.87%.
这些计数器来自独立 profiler replay.
融合后的短输出模型诊断中, prefill 中位数仍约为 1.39 s.

K=256 GEMM tile, 较小 M tile 和 two-SM down 投影没有表现出稳定的成对增益.
调度不使用这些候选. 失败尝试保存在外部证据目录.
N=256 tile 将大型 gate/up 的诊断时延从约 6.72 增加到 19.39 ms.
调度也不使用它们.
