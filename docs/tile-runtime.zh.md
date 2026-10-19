# Task 引擎

Task 引擎遵守 [tile 设计](tile-execution.zh.md)中的资源契约.
Task family 在相邻缓冲区元素上提供 FP32 copy, scale 和 add 操作.
模型执行仍使用 native 算子.
[迁移计划](tile-migration.zh.md)包含语言, 采样, vision 和准备 task.

## Plan 与提交

接口位于 [mega.py](../python/oh_my_vllm/kernels/mega.py).
`prepare("elementwise", capacity, bindings)` 提供拓扑及缓冲区地址稳定的 plan.
Bindings 提供 device, FP32 缓冲区, task 及输出缓冲区索引.
不同缓冲区 binding 不能使用同一内存. 一个 task 可以读写同一个 binding.

当多个 task 使用同一内存且其中一个 task 写入该内存时, 必须具有依赖.
这些依赖也控制缓冲区复用.

`execute(plan, metadata)` 在调用者的 stream 上提交工作, 并提供 completion 对象.
`completion.wait()` 先等待 CUDA event, 再读取设备状态.
调用者可以在不同 stream 上等待.
Completion 保留其输出缓冲区的引用.

下次提交前, 调用者必须调用 `completion.wait()`, 并增加 epoch.
访问 `completion.outputs` 也会调用 `completion.wait()`.
每个 plan 由一个调用者控制. 对同一个 plan 的主机提交不能同时发生.

`StepMetadata` 检查 query 长度, 页索引, 私有状态 slot 及有效行上限.
设备 task family 只接收 epoch 和有效行数.
语言 task binding 必须在模型集成前添加设备页表和状态 slot.
Task 不改动无效行.

执行 CUDA Graph 时, 使用 `Device.graph` 捕获 `plan.launch`.
通过 `replay` 参数将重放 callable 传给 `execute`.
元数据传输先于 graph 重放发生, 两者均使用调用者的 stream.
Graph lease 防止在 graph 仍能使用缓冲区时关闭 plan.
`close(plan)` 等待进行中的工作, 并释放 plan 资源.

## 设备进展与错误

[Native 引擎](../python/oh_my_vllm/kernels/cuda_backend/mega.cu)使用 cooperative launch.
Grid 大小不能大于完整 kernel 的 occupancy 上限.
Grid barrier 位于初始化, task 执行及完成发布之间.
Task 执行使用依赖. 它不在 task 之间使用 grid barrier.

[队列协议](../python/oh_my_vllm/kernels/cuda_backend/tile_runtime.cuh)使用固定容量.
设备原子操作在子 task 启动前使父 task 的写入可用.
队列已满时记录错误并终止. 生产者不等待可用队列空间.

错误的描述符, 未递增的 epoch 和错误的行数会导致终止.
图中无法完成的 task 也会导致终止.
后续设备提交仍保持故障状态.

执行失败后不能复用 plan.
Worker 对 `TileExecutionError` 使用 `device` 错误类型.
服务控制移除 worker, 并清除其调度器 epoch.
设备无法完成工作时, 由 step deadline 控制终止.
执行失败后, tensor 状态不能变成已接受状态.

## 检查

使用[开发命令](operators/development.zh.md)构建 native 模块.
运行 CPU 协议检查:

```bash
scripts/with-env.sh python -m unittest development.tile.test_engine
```

使用 600 秒 step deadline 运行 GPU 协议 fixture:

```bash
scripts/with-env.sh scripts/with-gpu.sh timeout --kill-after=5s 600s python -m unittest development.tile.gpu_fixture
```

Fixture 检查依赖, 变化的输入数据, 无效行, epoch 复用及 graph lease.
它还检查队列溢出, 持续保持的错误, 跨 stream 完成及大地址偏移.
大偏移检查使用一个 8 GiB 缓冲区, 并访问其最后一个元素.
该 fixture 不提供模型数值或性能证据.

分别使用 `memcheck`, `racecheck` 和 `synccheck` 运行 Compute Sanitizer:

```bash
scripts/with-env.sh scripts/with-gpu.sh timeout --kill-after=5s 600s "$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 2 python -m unittest development.tile.gpu_fixture
```

报告保存在仓库外.
