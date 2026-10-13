# 术语参考 C

| 术语 | 词性 | 项目含义 |
|---|---|---|
| `confidence head` | 名词 | 使用 checkpoint 权重的草稿 projection, 产生条件草稿 confidence, 用于选择草稿数量. |
| `cumulative confidence` | 名词 | 截至当前草稿位置的条件草稿 confidence 乘积. |
| `probability distribution` | 名词 | 配置采样变换及 mask 后的非负 vocabulary 概率, 总概率质量为 1. |
| `conditional distribution` | 名词 | 给定已接受历史及此前推测草稿后, 一个位置的 vocabulary 概率. |
| `stochastic sampling` | 名词 | 从配置的条件 vocabulary 概率中随机选择 token. |
| `hierarchical paired-bootstrap` | 名词 | 先重采样轮次, 再在选中轮次内重采样配对样本, 计算实测值之间差值的单侧置信界限. |
| `probability` | 名词 | 单个 vocabulary 结果或接受事件的非负概率质量, 取值为 0 到 1. |
| `confidence bound` | 名词 | 指定置信水平下, 从重采样测量数据计算的统计界限. |
| `greedy sampling` | 名词 | 选择配置模型分数的最大值, 并按确定的 token ID 顺序处理相同分数. |
| `native MTP4` | 名词 | 通过 target checkpoint 自带 MTP 层进行的 4-token 推测. |
| `backbone` | 名词 | 共享 vocabulary head 及 Markov 修正前的 5 层 DSpark attention / MLP. |
| `context injection` | 名词 | 投影已接受 target feature, 并将各层 KV 追加到 DSpark context 存储. |
| `base logits` | 名词 | Markov head 修正前, 共享 target head 产生的草稿 vocabulary 分数. |
| `Markov step` | 名词 | 以此前草稿 token 为输入, 计算一个 proposal row 的草稿 head. |
| `median` | 名词 | 测量样本排序后的中间值, 偶数样本时取两个中间值的平均数. |
| `load` | 动词 | 将 checkpoint tensor 或可执行代码读入推理运行时内存. |
| `initialize` | 动词 | 将运行时对象, device 存储和请求状态设置为规定的初始值. |
| `return` | 动词 | 将软件函数, RPC 或 HTTP 操作的输出数据传给调用方. |
| `commit` | 动词 | 使接受的 token 历史, grammar 或 cache 状态成为后续引擎步骤使用的状态. |
| `reset` | 动词 | 将指定软件计数器, 请求状态或 cache metadata 恢复为初始值. |
| `forward` | 动词 | 通过验收代理传输客户端 HTTP 请求和上游回复, 保持 body 不变. |
| `overwrite` | 动词 | 在保存此前计算值的 device 地址写入新的 tensor 值. |
| `generate` | 动词 | 通过选定推理和采样路径计算模型输出 token. |
| `concatenate` | 动词 | 沿指定维度连接 tensor 值, row 顺序不变. |
| `conditional draft confidence` | 名词 | 给定此前草稿 token 和当前 hidden row, confidence head 在一个 proposal 位置产生的分数. |
| `normalize` | 动词 | 按规定 norm 缩放 tensor 值, 或将 vocabulary 概率缩放为总概率质量 1. |
| `share` | 动词 | 不同软件执行路径使用相同物理模型权重, tensor 存储或 graph 内存 pool. |
| `bidirectional attention` | 名词 | 读取已接受 context KV 及全部 7 个临时 proposal KV row 的草稿 attention. |
| `synthetic token IDs` | 名词 | 按固定索引公式生成的基准输入 ID, 不包含自然语言文本或 tokenization. |
| `DAG` | 名词 | 语义操作及其数据依赖构成的有向无环图. |
| `HBM` | 名词 | 保存参数与持久数据的 GPU 高带宽内存. |
| `SFU` | 名词 | 执行指定原生特殊函数指令的 GPU 资源. |
| `TMEM` | 名词 | 保存 Tensor Core 操作数与累积结果的 SM100 tensor memory. |
| `PDL` | 名词 | 通过 producer trigger 和 consumer 数据等待实现的 CUDA programmatic dependent launch. |
| `CUDA stream` | 名词 | 通过显式跨 stream 依赖事件连接的 GPU 命令队列. |
| `swizzle` | 名词 | 改变 cache 局部性或 memory-bank 访问的 tile 遍历或地址排列. |
| `occupancy` | 名词 | 活跃 GPU warp 数与硬件 warp 容量之比. |
| `REDUX` | 名词 | 执行资源契约单独规定的 PTX warp reduction 指令. |
| `feedback interval` | 名词 | 两个 token 反馈边界之间的语义工作, 使用一次理想 cache 抵扣. |
| `pipeline stage` | 名词 | 用于使 GEMM 输入传输与计算并行执行的一块 shared-memory buffer. |
| `column-major scale` | 名词 | 相邻行索引在存储中连续的 FP32 scale 布局. |
| `K-major scale` | 名词 | 相邻归约分块索引在存储中连续的 FP32 scale 布局. |
| `CTA cluster` | 名词 | 一组共同执行并可使用硬件 multicast 的 CUDA thread block. |
| `TMA multicast` | 名词 | Tensor Memory Accelerator 将一个输入 tile 传输到 CTA cluster 中每个 block 的共享内存. |
| `register spill` | 名词 | 寄存器存储不足时, 将线程局部值存入 local memory 的行为. |
| `sector` | 名词 | NVIDIA profiler 统计的 32 字节访存单位. |
| `epilogue` | 名词 | 转换累加值并写入结果存储的 GEMM 输出阶段. |
| `soak test` | 名词 | 通过多个 agent 请求和资源测量验证长时服务的测试. |
| `processing start` | 名词 | 准入排队结束后, 框架开始准备请求的时间. |
| `admission lease` | 名词 | 单个请求的数量与字节预留的共享 owner, 在最后一个 engine, body 或 drain owner 释放后归还容量. |
| `worker epoch` | 名词 | 完成加载的一代 worker 的身份, 用于拒绝针对旧 worker 的准入. |
| `compiler variant` | 名词 | 针对某个 code identity 和输入 metadata 编译的程序, 带有复用条件. |
| `encoded state reservation` | 名词 | 为编码输入, 历史和输出的保守估计保留的字节容量. JSON, object 和 allocator 额外开销另计. |
| `reference cycle` | 名词 | 回到起始对象的 Python 对象引用, 可使不再使用的对象持续占用内存. |
| `normal request` | 名词 | 持续测试测量负载中的任务或压缩模型请求, 与有意安排的探测分开. |
| `persistent session` | 名词 | 在任务和压缩期间保持同一会话身份的客户端进程及会话历史. |
| `compaction` | 名词 | 在同一会话中用摘要替换部分历史. |
| `context snapshot` | 名词 | 随已完成 assistant 消息保存的 provider 输入计数和上下文 metadata. |
| `weak reference` | 名词 | 不使其目标对象继续驻留内存的 Python 对象引用. |
| `sandbox` | 名词 | 具有独立进程和文件系统 namespace 的可删除 agent 容器. |
| `native goal` | 名词 | 在同一会话中保存状态并自动续行的 OMP 目标. |
