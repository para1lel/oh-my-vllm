# 工作状态

从基线 `b99701a` 在 `nightly` 开展规划的 DeepSeek 扩展.
阅读[架构提案](deepseek-v41/design.zh.md), [阶段计划](deepseek-v41/plan.zh.md)和[验证计划](deepseek-v41/validation.zh.md).
实现仍然只支持 Qwen. 尚无 DeepSeek 运行时改动.

架构和测量矩阵已获批准.
执行全部阶段, 不再逐阶段等待批准.
阶段 0 正在进行: 冻结 checkpoint 身份, 张量布局, 模型数学行为和独立数值参考.

阶段 1 [适配器抽取](deepseek-v41/adapters.zh.md) 也在参考生成期间进行.
Python 注册表在导入 GPU 后端前选择 Qwen.
Rust model-contract crate 提供描述符类型, 但当前 worker 消息尚不包含这些类型.
Qwen 后端及其状态语义保持不变.
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
视频解码和完整的媒体模型端点尚未实现.
[采样约定](deepseek-v41/sampling.zh.md) 定义待处理 token 对齐, 精确修正和请求内 RNG 地址.

媒体和长上下文参考输出, 数值门槛和完整的运行时分配上界仍是前置条件.
DeepSeek 和新的 nightly 回归测量放入独立文档, 不放进 README.

以下状态描述 Qwen 实现.
[算子包](../python/oh_my_vllm/operators) 将各算子族的接口, 计划, 工作区和 CUDA 代码放在一起.
[结构设计](operators/design.zh.md) 定义其依赖关系和所有权.
将共享资源机制放入 `runtime`, 将共享计算放入 `primitives`.
native 构建器为 `development.native_build`. 它构建七个 native 模块.

worker 使用显式层权重, 命名输出和算子计划键.
按照[执行规则](execution.zh.md) 使用 Eager 执行, CUDA Graph 重放和资源释放.
Rust 和 worker 控制持久状态选择.
算子只写入调用方提供的状态位置.

仅含 decode 的[草稿链](../python/oh_my_vllm/worker/draft_chain.py)可以将随机采样和私有 presence 历史保留在 GPU 上.
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
