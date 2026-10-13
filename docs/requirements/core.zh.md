# 核心契约

## REQ-GOAL-001: 推理目标

为单张 B200 GPU 上的 Qwen3.8-27B-FP8 提供 Rust 推理框架.
保留全部现有离线和 HTTP 功能.
每个性能工作负载必须满足 `REQ-PERF-001` 和 `REQ-PERF-002`.

## REQ-ARCH-001: 职责

Rust 负责服务, 请求队列, token 预算, 逻辑 KV 分配, 前缀复用和抢占.
Python 负责模型加载, GPU forward 计算, 内核和手动 CUDA Graph.
使用一条 ZMQ DEALER 通道传输 msgpack.
Tensor 保留在 Python 侧.

## REQ-MODEL-001: 目标

使用包含 48 层 GDN 和 16 层 FA 的 Qwen3.8-27B-FP8.
所有生产路径使用块大小 784.
FA group 为 0, GDN group 为 1, 使用 `mamba_cache_mode="align"`.
Checkpoint 位置通过 `OH_MY_VLLM_MODEL` 或 `--model` 设置.

## REQ-EXEC-001: GPU 执行

模型计算使用 CUDA Graph, 多个 CUDA stream 和 Programmatic Dependent Launch (PDL).

图执行必须启用 native caching allocator, 且不能显式设置 `max_split_size_mb`.
启动时拒绝不支持的实际设置, 不改写原值. 显式 eager 诊断仍可使用.
被捕获的 cuBLAS 工作区保留在所属图的 pool 中, 工作区预算不改.

通过语义操作和 tensor / state 读写定义图依赖.
独立分支可以并行执行.
PDL consumer 必须先等待, 再读取 producer 输出.

保留 graph capture 状态恢复, 缓存所有权和有界的 graph 存储.
通过完整算子与框架测量调优执行.
记录每项选定优化的配置, 数值检查, 显存和时延.
通过单独测量区分 graph, stream 和 PDL 的效果.
阶段时延与算子门槛继续有效.

## REQ-CONTEXT-001: 上下文与显存

支持输入与输出合计 262144 token.
普通和 MTP4 模式, batch 1, 2, 4, 完成输入 258048, 输出 4096 的用例.
这些运行不得发生 OOM 或重计算抢占.
保持 32768-token 步预算和 784-token 块.
记录性能和 GPU 显存峰值; 不增加额外比率门槛.
覆盖长上下文前缀恢复和约束解码.
保留 6 个普通 / MTP4 用例作为独立门槛.

## 功能需求

| ID | 合同 |
|---|---|
| REQ-FUNC-001 | 在 32768-token 预算内分块 prefill; 保存对齐的 GDN checkpoint. |
| REQ-FUNC-002 | 每步先调度活跃解码, 再调度新 prefill. |
| REQ-FUNC-003 | 使用链式哈希前缀复用, LRU 淘汰和协调的 FA / GDN group. |
| REQ-FUNC-004 | 优先抢占较晚接纳的运行请求; 重计算保留已接受历史, 清除未验证草稿. |
| REQ-FUNC-005 | 提供 MTP4, 显式返回保留 token 和新草稿 ID; 保持 BF16 GDN 状态和 block784. |
| REQ-FUNC-006 | 提供可选 DSpark, 使用本地 BF16 checkpoint, 至多 7 个草稿, target 验证和 block784. |

DSpark 使用从 0 开始编号的 target 层输出 `(5,19,33,47,61)`, 以及 5 层草稿 GQA.
保留原生 MTP4, 每个 worker 选择一种草稿模式, 不按请求切换.

DSpark 累计 confidence 阈值的初始设置为 `0.2`, proposal 数量为 0 至 7.
首项 confidence 乘积低于阈值时, 该步由目标执行单 token 解码.
设置 `--dspark-confidence-threshold 0.0` 可保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.

随机采样使用条件 proposal 概率及 target 接受概率 `min(1,p(x)/q(x))`.
拒绝时从归一化的 `max(p-q,0)` 采样, bonus token 从 target 分布采样.
每个条件分布都应用用户采样参数和 grammar mask.
推测模式保持 BF16 GDN 状态及既有半精度舍入约束.
只有接受的输入行才能提交 target 状态或 DSpark context.

## REQ-ACC-001: 数值精度

实际 GQA 和 GDN 推理路径与独立 CPU FP64 参考比较.
覆盖 prefill, decode, 分块边界和 recurrent state.
使用实际舍入后的输入.
BF16 输出使用 `atol=rtol=0.03`.
Recurrent state 的 NRMSE 至多 1%, 最大绝对误差至多为参考峰值的 2%.
保留路径相关容差和现有测试.
展示连贯文本, MTP 和前缀命中.
完整模型 token 相等不作为门槛.

## REQ-RUNNER-002: 独立运行时

使用项目实现和独立库.
项目构建, 测试和推理不得安装, 导入或链接 vLLM.
不得依赖其 checkout, 旧环境或编译缓存.
旧 adapter 已移除; [ADR-002](../decisions/ADR-002-gpuworker-adapter.zh.md) 和 [ADR-005](../decisions/ADR-005-v2-model-runner.zh.md) 记录替代关系.
`REQ-RUNNER-001` 已退役, 由本需求替代.
按依赖顺序选择兼容的稳定版本, 固定已验证版本.
保留移植代码的来源和许可证.
未来架构, 多 GPU 和其他 NVIDIA GPU 只写简要扩展说明.

## REQ-OBS-001: 诊断

提供 UTC 时间戳, 单调时长, run / request / step 关联和可配置日志级别.
默认级别避免逐步 I/O.
Debug 日志暴露调度, 缓存, 传输和 worker 时长.
区分主机时间与 CUDA 内核时间.
Profiling 需显式选择.

分别记录每种模式的已调度, 接受和返回草稿数.
接受率为接受草稿总数除以已调度草稿总数.
返回的下一步 proposal 使用不同的诊断计数.

将加载的 DSpark 配置和权重哈希与指定 checkpoint 比较. 两者必须一致.
