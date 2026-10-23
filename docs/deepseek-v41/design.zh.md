# DeepSeek V4.1 架构

## 范围与状态

本已批准设计在 `nightly` 扩展服务.
它不表示当前可执行程序已经支持 DeepSeek.
执行[阶段计划](plan.zh.md), 不再逐阶段等待批准.
[验证计划](validation.zh.md)给出已批准的测量矩阵.

保留单张 B200 上的 Qwen3.8-27B-FP8 及其现有功能和数值门槛.
增加四张 B200 GPU 上的 DeepSeek-V4.1-Flash.
支持 64 个活跃请求, 每个请求的输入与输出总上限为 1048576 tokens.
支持普通解码和五个草稿 token 的 DSpark.
Engram 权重保存在 pinned host memory.

Rust 负责服务, 准入, 调度, 逻辑状态和经过验证的状态提交.
Python 负责模型计算和物理张量.
进程控制继续使用 ZMQ DEALER 和 msgpack.
模型算子使用 CUDA C++ 或 CuTe DSL.
原生 CUDA 和 NCCL 可以提供设备及通信机制.
不添加外部算子库或 vLLM 依赖.

## 来源事实

本地 checkpoint 配置选择 `DeepseekV41ForCausalLM` 和 `deepseek_v41`.
它有 40 个目标层, 宽度 5120, 词表大小为 129280 tokens.
每个目标层有 384 个路由专家, 选择六个专家, 并有一个共享专家.
配置指定三个预测层, `dspark_block_size=5` 和 `max_position_embeddings=1048576`.
视觉配置有 32 层, patch 大小为 14.

架构使用因果编码器-解码器计算, 压缩稀疏注意力, 共享缓存来源, hyper-connections 和基于 token 的 Engram 查表.
FP8 矩阵采用 32-by-32 分块及 UE8M0 scales.
专家采用打包的 FP4 权重.
主压缩 KV 使用 FP4 及独立的 scale 规则.
这些格式与 Qwen 算子不同.

[夹具](fixtures.zh.md)绑定完整检查点身份.
[内存计划](memory.zh.md)给出权重放置与分配门槛.

以 checkpoint 提供的参考模型, 编码器和图像处理器为数学来源.
其生成循环实现普通采样, 而非完整的 DSpark 接受算法.
提示集成使用权威 Python 编码器和[固定编码用例](fixtures.zh.md#提示词编码).

[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) 仅作为参考, 不作为运行时依赖.
它的默认 checkpoint 带有 `-1017` 后缀.
复用前检查源码与权重的兼容性.
在实现契约中记录固定源码 revision, 许可证和算法差异.
不复制其外部算子导入, 调度器或环境.
它的[并行配置](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/config.py)使用上下文并行, 不是本文的 TP4 定义.
它的[重放指南](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)允许近似的有界重建.
它的[采样代码](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py)使用数值 penalty 调整拒绝分布.
不得将这些近似作为精确状态恢复或精确采样采用.

## 架构选择

采用模型适配器.
保留[算子设计](../operators/design.zh.md)中的现有算子包布局.
不构建通用模型图编译器.
新模型族提供适配器和缺失算子, 不在服务各处修改代码.
此结构允许后续支持 dense, MoE, recurrent, sparse-attention 和 multimodal 模型.
它不保证立即支持尚未实现的架构.

## 模块边界

已实现的[模型适配器](adapters.zh.md)和 [rank 控制](distributed.zh.md)规定共享接口.
将 DeepSeek 权重, 执行, 缓存, 媒体和 DSpark 保留在 `python/oh_my_vllm/models/deepseek_v41/`.
DeepSeek 模型计算尚待完成.
GPU 操作使用 `python/oh_my_vllm/operators/` 中现有的 API/plan/workspace 包.

模型选择使用配置中的架构与模型类型, 不使用目录名.
未知架构, 格式, 能力或协议版本导致启动错误.
Qwen 专属块大小, 词表大小, 层数和草稿数量移出共享代码.
迁移期间保持 Qwen 状态语义不变.

## 并行策略

以下是已批准的项目定义, 不代表 DeepInfer 命令名称.
所有策略均使用一个四 rank 组, 而非十六张 GPU.

| 策略 | 注意力与 dense 投影 | 路由专家 | 请求 |
|---|---|---|---|
| TP4 | 在四个 rank 间进行张量分片 | 每个专家有四个张量分片 | 所有 rank 调度相同请求 |
| TEP4 | 在四个 rank 间进行张量分片 | 专家所有权分布在四个 rank | 相同请求, 配合 token 分发与合并 |
| DEP4 | 每个 rank 一份注意力副本 | 专家所有权分布在四个 rank | 请求分配到四个 rank |

TEP4 在专家边界转换张量布局.
DEP4 在同一组四个 rank 之间交换专家 token.
它不是四个独立的完整模型副本.
在每种权重计划中指定共享专家, embedding, 词表 head, 视觉和 DSpark 的放置方式.
复制的 latent KV 与分片的注意力头分别计费.
DEP4 请求留在其注意力所有者上, 直到显式状态传输完成.
空 rank 也参与约定的 collective 序列.

Rust 控制一个组步骤及其 rank 调度.
每个 rank 使用独立 DEALER 身份, 报告容量和完成状态.
张量数据通过原生 collectives 传输, 不经过 msgpack.
Rust 在所有必要 rank 结果通过验证后发布一个请求结果.
消息携带协议版本, group epoch, step ID, rank ID 和 state transaction ID.
拒绝过期, 重复, 不完整或不一致的结果.
一个 rank 故障导致整个组及其活跃请求失败.
全部旧 rank 退出后才重启. 不承诺活跃请求的透明恢复.

## 状态与执行

缓存组描述包含身份, 类型, token 区间, 位置, 格式, 所有者 rank 和容量.
初始类型覆盖 full attention, recurrent checkpoints, sliding windows, compressed KV, index keys 和模型专属 checkpoints.
相关组组成一个状态事务.
状态提交或前缀发布前验证分配结果和 rank 完成状态.

DeepSeek 状态还包含未完成的 compressor 组, 共享来源位置, candidate indices, Engram 历史和 DSpark hidden states.
可复用的已接受状态与推测写入分离.
Copy-on-write 保护部分前缀和源块.
前缀键绑定模型权重, tokenizer, 媒体, 位置规则, 缓存格式和并行布局.
Host offload 保存精确续算所需的已接受状态.
有界滑窗重放必须覆盖推测回退距离和未完成压缩组.
从模型契约推导重放长度. 不未经证明直接复制参考常数.

含任意 prefill 的 Execute 在整个组使用 Eager 执行.
仅含 decode 的执行在全部参与 rank 使用 CUDA Graphs.
Graph key 包含模型, 拓扑, batch 容量, 草稿容量和通信计划.
Capture 和 replay 期间保持张量地址, collective 顺序及 padding 规则固定.
Engram 传输与 GPU 消费者使用显式完成依赖.
保持现有 metadata 上传和资源释放规则.

## 内存与容量

Engram 张量与 scales 的权威副本保存在 pinned host storage.
本地 rank 共享或分片物理主机存储. 默认不创建四份完整主机副本.
查表行和 Engram 投影计算使用有界设备暂存.
不得将主机存储悄悄替换为完整 GPU 表.
分别管理 Engram, host KV, 上传缓冲区和传输预算.
保留其所有者, 直到全部读操作和传输完成.
服务启动前分配大型主机存储. 执行 NUMA 放置和启动内存检查.

模型报告给出的全局 KV 为每 token 890 bytes.
64 个请求各 1048576 tokens 对应 55.625 GiB, 未计复制和其他状态.
该数值不是服务内存预算.
按 rank 计算权重, 全部缓存组, 通信, graph pools, 媒体及临时存储预算.
限制 prefill chunk 和 candidate workspace. 不分配上下文长度平方规模的 dense attention scores.
使用 64-bit 字节大小和带检查的算术.
支持 64 个活跃请求, 不以串行准入冒充并行.
调度器并行度与较小的物理 kernel 分片区分开.
容量验收必须同时覆盖并行度和上下文目标.

## 功能对齐与风险

保留验证清单中的共享 API, CLI, tools, 结构化输出, 采样, 流式输出, 存储, 取消和故障行为.
使用模型专属的提示, 推理, 工具, 图像和视频编解码器.
DeepSeek 不复用 Qwen token IDs 或三轴媒体位置.
DeepSeek 视频仅在编码和数值验证后采用有序帧图像与时间戳.
该路径通过验收前不得声称完全对齐.

DSpark 是独立算法, 不是改变循环次数后的 MTP4.
五个草稿可能需要六个目标验证行. Confidence 控制验证长度, 不决定接受概率.
其契约须指定 proposal 概率, 置信度调度, 验证位置, 拒绝修正, RNG 和已接受状态恢复.
目标和草稿路径使用相同的采样约束.
被拒绝草稿不得改变 grammar, penalties, 历史或前缀状态.

首批门槛是 checkpoint 兼容性, 数值参考, 四 rank 故障安全及完整内存预算.
不得修改数值门槛来使实现通过.
完成指定测量前不宣称吞吐量.
