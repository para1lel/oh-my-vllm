# 工作状态

从基线 `b99701a` 在 `nightly` 开展规划的 DeepSeek 扩展.
阅读[架构提案](deepseek-v41/design.zh.md), [阶段计划](deepseek-v41/plan.zh.md)和[验证计划](deepseek-v41/validation.zh.md).
实现仍然只支持 Qwen. 尚无 DeepSeek 运行时改动.

架构和测量矩阵已获批准.
执行全部阶段, 不再逐阶段等待批准.
阶段 0 正在进行: 冻结 checkpoint 身份, 张量布局, 模型数学行为和独立数值参考.

阶段 1 [适配器抽取](deepseek-v41/adapters.zh.md) 已完成.
Python 注册表在导入 GPU 后端前选择 Qwen.
Init 协商描述符版本 1, Ready 提供描述符, 在构建调度器前进行验证.
Python Qwen 描述符构建器绑定已加载身份和已分配缓存容量, 不导入 GPU 模块.

Qwen Python 控制位于模型系列目录.
共享工具约束必须使用模型提供的语法.
Rust 输出解析使用 Ready 阶段接受的模型系列.
Rust 模型策略拥有默认值, 选项验证和模型系列特定的就绪规则.
CPU 模型选择发生在绑定 socket 或启动 worker 之前.

[适配器](deepseek-v41/qwen-regression.zh.md), [生命周期](deepseek-v41/qwen-foundations.zh.md), [运行时](deepseek-v41/qwen-collectives.zh.md) 和[缓存策略](deepseek-v41/qwen-cache-policy.zh.md) 报告记录 Qwen HTTP 矩阵.
[类型化执行报告](deepseek-v41/qwen-typed-execution.zh.md)记录后续 Qwen HTTP 矩阵.
阶段 2 基础设施已完成.
阶段 0 数值限制和完整分配边界是阶段 3 算子的前置条件.

[状态 arena](deepseek-v41/state-arena.zh.md) 提供类型化逻辑存储预留和读 lease.
存储选择仅保留其源所有者已经持有的块.
它不替换 Qwen 协调器, 也不发布已接受前缀.

[GPU lease 包装器](deepseek-v41/gpu-groups.zh.md) 通过协作锁提供单设备和四设备选择.
[rank 检查器](deepseek-v41/rank-control.zh.md) 验证就绪和全部 rank 的完成元数据.
[进程监督器](deepseek-v41/rank-supervisor.zh.md) 保留事务, 直到完成或所拥有的进程终止.
它为每个 rank 提供一个可见设备, 并控制 NCCL bootstrap ID 交换.
[集合通信计划](deepseek-v41/collective-plan.zh.md) 在 Rust 与 Python 之间绑定操作顺序, 数据类型, 计数和拓扑.
可选的 [native 集合通信所有者](deepseek-v41/native-collectives.zh.md) 提供 NCCL 提交, 私有对齐存储和图资源所有权.

这些机制不发布已接受前缀.
[缓存策略](deepseek-v41/cache-policy.zh.md) 为共享调度中的状态分配和主机迁移所有权提供模型类型.
Qwen 保留现有 coordinator 和 wire 格式.
[类型化执行接口](deepseek-v41/typed-execution.zh.md) 将调度器事务转交给组 client.
它在发布已接受状态之前验证模型输出和全部 rank 完成.
DeepSeek 状态选择和物理模型执行仍是后续工作.

Qwen 计算和已接受状态语义保持不变.
DeepSeek 执行未启用.

[checkpoint 工具](deepseek-v41/checkpoint.zh.md)提供完整字节身份和惰性 CPU FP64 权重读取器.
[模型契约](deepseek-v41/model.zh.md)和[状态契约](deepseek-v41/state.zh.md)给出本地数学来源和后续参考门槛.
[内存规划器](deepseek-v41/memory.zh.md)为全部三种四 rank 策略指定张量放置和存储预算.
checkpoint 工具还提供独立 CPU Engram 哈希和续算历史.

[参考构建指南](deepseek-v41/reference.zh.md)规定流式权重和独立的 PyTorch FP64 数学检查.
[固定夹具](deepseek-v41/fixtures.zh.md) 绑定短文本端点, DSpark 恢复端点和提示词编码用例.
其中的上游比较在没有相同权重身份凭据时拒绝检查点替换.
其中的 DSpark 恢复生成器和视觉参考提供 CPU FP64 数学实现, 而不是服务执行.

图像参考提供与源实现兼容的像素和 patch 顺序, 不导入运行时代码.
组合图像和视觉检查时, 使用独立的媒体参考环境.
[视频参考](deepseek-v41/video.zh.md) 提供帧选择和逐帧时间戳块.
[媒体生成器](deepseek-v41/media-reference.zh.md) 提供包含静态图片和合成已解码视频帧的完整目标模型端点.
其固定产物绑定完整的 prefill 和 decode 向量.
视频解码尚未提供.

[采样约定](deepseek-v41/sampling.zh.md) 定义待处理 token 对齐, 精确修正和请求内 RNG 地址.

[长上下文生成器](deepseek-v41/long-reference.zh.md) 提供候选池和原始旋转位置编码上下文边界的构建.
其固定输出, 数值门槛和完整的运行时分配上界仍是前置条件.
DeepSeek 和新的 nightly 回归测量放入独立文档, 不放进 README.

以下状态描述 Qwen 实现.
[算子包](../python/oh_my_vllm/operators) 将各算子族的接口, 计划, 工作区和 CUDA 代码放在一起.
[结构设计](operators/design.zh.md) 定义其依赖关系和所有权.
将共享资源机制放入 `runtime`, 将共享计算放入 `primitives`.

native 构建器为 `development.native_build`. 其默认选择构建七个 Qwen native 模块.
使用 `--module collectives` 构建可选模块.

worker 使用显式层权重, 命名输出和算子计划键.
按照[执行规则](execution.zh.md) 使用 Eager 执行, CUDA Graph 重放和资源释放.
Rust 和 worker 控制持久状态选择.
算子只写入调用方提供的状态位置.

仅含 decode 的[草稿链](../python/oh_my_vllm/models/qwen38/draft_chain.py)可以将随机采样和私有 presence 历史保留在 GPU 上.
仅第一个草稿步骤从主机元数据获取输入 token.
后续草稿步骤使用上一步的 GPU token 输出.
依赖前缀的 mask 和不支持的停止条件使用逐步路径.
保持各草稿的 proposal 概率和 RNG offset 与逐步路径相同.

GDN prefill 按序列顺序存放 chunk workspace, 并为每个序列计算容量.
保持打包后的 offset 与当前 schedule 和 plan 容量一致.
[服务 cache 预算](serving.zh.md)使用固定 FA 默认值, 并检查运行时显存和文本 workspace 预留量.
Rust 使用 worker 返回的容量.

小规模 FFN 和 GDN 运算在执行规则规定的边界使用依赖启动.
FFN prefill 在 65 到 1024 行时使用这些依赖.
[前缀 cache](serving.zh.md#前缀-cache) 可以使用来自未满块的 checkpoint.
保留源块, 直到续算结果没有校验错误.

修改 kernel 前使用[算子验证框架](../development/operators).
该框架提供固定数值 fixture, 直接 PyTorch 参考实现和理论延迟倍率门槛.
常规数值测试读取仓库 fixture, 不执行 PyTorch.
保持[精度策略](operators/precision.zh.md)和[全模型门槛](operators/evaluation.zh.md).
实现改变后不得自动修改门槛.

算子性能采样在不确定结果的扩展期间保留计时事件及其测量 Graph.
计时迁移保持已批准的延迟门槛不变.

FA 使用 BF16 attention 操作数和缓存状态. GDN 使用 FP32 保存循环状态.
FP8 投影使用 checkpoint 中的 E4M3 权重和 FP32 block scales.
[模型契约](operators/model.zh.md#中间精度)定义中间计算精度.

HTTP 计费使用串行 kernel 分组和当前精度规则.
融合或执行边界变化时, 使用[性能计算](performance.zh.md#下界计算).
计费变化时, 算子倍率门槛保持已批准的最大延迟.

共享 FP8 prefill 投影 kernel 根据输入宽度选择 tile 分组大小.
FFN 和投影 kernel 在累加期间将 TMEM 地址保留在寄存器中.
HTTP efficiency 不设最低交付门槛.
保持数值门槛和算子倍率门槛.
报告实测性能.

交付前执行适用的[测试范围](testing.zh.md).
实现交付时, 按 nightly 验证计划选择 24 组 Qwen [HTTP 配置](performance.zh.md).

[Host KV cache](kv-offload.zh.md) 支持被中断请求及由容量压力触发的前缀存储.
保持传输引用直到完成, 包括已取消请求.
部分 GDN 读取 checkpoint 与续算写入 slot 必须分开, 直到经过验证的提交.
Pinned 存储在空闲窗口扩容. Metadata 上传使用 GPU kernel, 不进入 bulk copy 队列.

[AgentX 入口](agentx.zh.md)使用四个会话树测量 MTP4.
它通过流式 usage 和完整响应 ID 关联服务指标.
将其获准保存的 CSV 和图片保存在 `docs/assets`.
将原始测量产物保存在仓库外.
将主机设置放入被忽略的 `LOCAL.md`.
