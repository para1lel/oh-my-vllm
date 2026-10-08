# Profiling 与诊断

用测量定位吞吐或 TTFT 差距来源.
Profiling 与正式计时分开.
TileFoundry 的 arithmetic, memory 和 roofline 结果是估算.
[需求](requirements.zh.md) 中后续 roofline 门槛尚未替代当前验收协议.

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
