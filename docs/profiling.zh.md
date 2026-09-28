# 性能分析指南

先看带时间戳的日志，再分析导致差距的部分。性能分析与吞吐验收分开进行，禁止提交原始 trace。

## 主机诊断

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=debug OH_MY_VLLM_LOG_LEVEL=DEBUG target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-diagnostic.ipc bench --batch-size 1 --input-len 32768 --output-len 256 --warmup 1 --repetitions 1
```

关联 run_id 和 step_id，对比 Rust 调度耗时/空闲块数、RPC 往返、Python 解码、主机执行及编码/发送。RPC 与 worker 执行差距大，提示传输/唤醒开销，但不能仅凭此确定哪个线程或 kernel 导致延迟。DEBUG I/O 会改变时延，优化后需恢复普通日志重新验证。

纯 CPU echo worker 可以隔离传输开销，但不能证明模型吞吐。CPU 亲和性实验必须让基线与框架设置一致；记录 CPU 掩码并调查方差。

## Rust 火焰图

如果已有 cargo-flamegraph 和 perf，通过相同包装脚本运行。需要符号时，使用开启 release 调试信息的隔离构建，不要替换其他基准正在使用的二进制。

```bash
scripts/with-gpu.sh scripts/with-env.sh env CARGO_TARGET_DIR=/tmp/oh-my-vllm-profile-target CARGO_PROFILE_RELEASE_DEBUG=1 cargo flamegraph --output /tmp/oh-my-vllm-flamegraph.svg --bin oh-my-vllm-zmq-worker -- --socket /tmp/oh-my-vllm-flamegraph.ipc bench --batch-size 1 --input-len 32768 --output-len 256 --warmup 1 --repetitions 1
```

检查调度/分配、不必要的 token 历史复制、块表更新、序列化和运行时唤醒。在测量前，任何预期的开销预算都只是待验证假设。

## Python 与 CUDA 性能分析

使用 conda oh-my-vllm Python 以及 GPU/环境包装脚本，在诊断程序中包装选定的 worker 调用。torch.profiler 的 CPU/CUDA activity 可以区分 kernel 工作与主机 launch/等待时间；cProfile 只能解释 Python 和主机调用耗时。通过 OH_MY_VLLM_WORKER_PYTHON 指定的诊断包装器必须接受解释器形式的参数：`-m oh_my_vllm.worker.zmq_bridge --socket URI`。

```python
from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    worker_out = worker.execute_model(sched_out)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/oh-my-vllm-trace.json")
```

结合初始化日志与真实后端配置识别 kernel。当前探针包装独立分页注意力及 GDN prefill/recurrent 调用；不要求历史 FA kernel 名称。检查 FP8 GEMM 分派、CUDAGraph 覆盖、prefill 分块边界、draft 数量和运行时 JIT 警告。不同 kernel 路径需要调查，不能自动认定某个后端名称对所有版本都正确。

## 配对验收

按 testing.md 使用 `benchmarks/ttft.py`。它输出原始重复测量、中位数、比例、稳定性、配置及身份；任一门槛失败则以失败状态退出。profiling trace 不属于验收运行。

## 自定义 kernel

TileFoundry 的静态成本/显存/roofline 分析是开发假设，不是测得的 kernel 时间。分析实际模型以定位吞吐/TTFT 差距。临时算子调优脚本和报告放在仓库外。正式测量审计 FlashInfer、Triton、TileLang 和 `TVM_FFI_CACHE_DIR` 的完整缓存树，包括文本产物；根目录未设置或不存在会令审计失败。修复后的审计尚未经新的完整 12 行采集验证。所有编译在正式重复测量前完成。默认 CUDA 路径用 `cudaPeekAtLastError` 检查各自有 kernel 的启动，不清除当前主机线程的 CUDA runtime 错误。此前的异步故障仍可能在此处或之后的主机同步中暴露；默认错误只标识观察点，不证明哪一个 kernel 出错。

排查 KRN-09 故障时，在 Python 启动前同时设置以下两个变量，并使用新的 eager 进程：

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_CUDA_DEBUG_SYNC=1 OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_KERNEL_BACKEND=cuda python -m tests.gpu_cuda_error_case eager
```

该诊断模式在每次自有 kernel 启动前检查现存错误并同步选中 stream，启动后再次检查和同步；CUDA Graph capture 会被拒绝。“先前”表示启动前观察到，“启动后”表示在后置检查中观察到。CUDA 同步也可能报告别处更早的异步故障，因此消息本身不能证明故障指令。要定位设备侧细节，对缩小的失败用例运行 `/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck`。诊断输出放在 Git 仓库外，不得把 debug-sync 计时用于性能验收。参见 [CUDA 错误 API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html) 和 [CUDA Graph capture 规则](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

分页 decode 对 storage offset 未按 16 字节对齐的连续 BF16 cache view 通过克隆维持支持。
每个进程的 `oh_my_vllm.kernels.decode_attention.unaligned_cache_clone_count()`
报告触发克隆回退的调用次数；在第 1、2、4、8 次及之后的 2 的幂次发出附带复制字节数的警告。
对齐 cache 不会进入克隆路径。

## CUDA/TileLang 算子观测

通过 GPU/环境包装脚本运行 `python -m development.kernels.observations`，采集正式配置下两个后端的完整算子，命令见 cuda-development.zh.md。报告分别标注 TileFoundry HIR 估计和 Nsight 计数器/资源指标，核验冻结对照/源码身份和指标完整性。该 profiling 流程独立于 `benchmarks/kernels.py` 的 CUDA Event/Graph 计时，不提供性能验收样本。原始 profiler CSV 仅临时保存在 Git 仓库外。
