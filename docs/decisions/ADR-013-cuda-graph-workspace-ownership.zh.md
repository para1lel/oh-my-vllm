# ADR-013: CUDA 图工作区所有权

状态: 生效. 日期: 2026-10-12.

## 问题

完整模型的 DSpark 采集因 CUDA 非法内存访问失败.
同步诊断将首次失败操作定位到上下文注入.
该上下文图投影 29 行, 每行 25600 个 BF16 特征.

固定的 Torch 2.14.0 按 handle 和 stream 缓存 cuBLAS 工作区.
图 reset 移除该 capture stream 缓存的工作区 owner.
其他驻留图仍可能使用对应地址.

捕获外预热可能在默认 pool 分配这个工作区.
后续缓存释放与 allocator 清理可能移除重放仍需使用的存储.
[PyTorch issue 193402](https://github.com/pytorch/pytorch/issues/193402) 描述了此故障机制.
固定版本的[图 reset](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CUDAGraph.cpp) 和[工作区缓存](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CublasHandlePool.cpp) 包含相关生命周期规则.

## 决策

全部七个 worker 捕获入口使用统一的项目捕获边界.
先同步已有 GPU 工作, 再在进入捕获前清除缓存的 cuBLAS 工作区 owner.
第一个被捕获的 GEMM 在所属图的私有 pool 中分配新工作区.
工作区大小, 权重, GEMM 配置和数值容差保持不变.
不修改已安装的 Torch 或 CUDA provider 代码.

图执行要求启用 native caching allocator, 且不能显式设置 `max_split_size_mb`.
有限的 split 上限可能在图 owner 仍驻留时释放私有 pool 中的缓存块.
`cudaMallocAsync` backend 的 pool 规则不同, 此修复尚未对其验证.
关闭 caching 同样会绕过图私有存储, 因此也被拒绝.
在 worker 启动时校验实际运行时设置, 位于图缓存的捕获异常回退之前.
在捕获边界再次校验. 拒绝不支持的配置, 不改写原值.
显式 `OH_MY_VLLM_ENFORCE_EAGER=1` 对其他 allocator 配置保留原有诊断行为.

Worker 主机线程串行执行捕获和重放.
持久输入, 权重和 KV 存储位于共享图临时 pool 之外.
同一 family 的下一次捕获或重放前, 必须读取或复制上一轮输出.
不同图 family 保留独立 pool. 图驱逐保留现有所有权与同步规则.

## 验证与限制

外部同 stream BF16 GEMM 复现在无关图 reset 后首次重放驻留图时失败.
捕获边界在工作区预算不变的条件下完成二十轮 reset 和 allocator 压力循环.

外部真实 checkpoint DSpark 上下文测试完成 42 个案例, 210 次层比较.
它保留 32 个捕获 shape, 测试原始 29 行 metadata, 并使用不经手动捕获的 33 行编译路径.
全部比较通过 `atol=0.03`, `rtol=0.03`, 最大绝对误差为零.
未修改行的 canary 通过精确比较. 工作区预算保持 32 MiB 和 1 MiB.
Native allocator 快照显示, 无关图 reset 和清理后, 对应工作区大小的存储仍在驻留私有 pool 中.
这些快照支持所有权机制, 但没有直接识别内部 cuBLAS 指针.

零工作区实验未通过现有数值容差, 已排除.
保留的成功诊断使用合成 target 特征和真实 draft checkpoint.
它未执行完整 target 模型, Rust 调度, Markov 采样或 HTTP 请求.
仓库回归测试以随机权重执行生产上下文方法和维度.
CPU 测试验证同步, 校验, 错误传播以及未调用预算修改 API.

所有受影响的正式负载都需要在已提交源码上重新测量.
修复增加了捕获边界的工作. 诊断不能支持性能结论.
原始失败与取消记录保留各自状态和源码身份.
