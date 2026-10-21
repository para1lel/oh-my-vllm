# 工作状态

使用 `main` 开展工作.
[算子包](../python/oh_my_vllm/operators) 将各算子族的接口, 计划, 工作区和 CUDA 代码放在一起.
[结构设计](operators/design.zh.md) 定义其依赖关系和所有权.
将共享资源机制放入 `runtime`, 将共享计算放入 `primitives`.
native 构建器为 `development.native_build`. 它构建七个 native 模块.

worker 使用显式层权重, 命名输出和算子计划键.
按照[执行规则](execution.zh.md) 使用 Eager 执行, CUDA Graph 重放和资源释放.
Rust 和 worker 控制持久状态选择.
算子只写入调用方提供的状态位置.

小规模 FFN 和 GDN 运算在执行规则规定的边界使用依赖启动.
FFN prefill 在 65 到 1024 行时使用这些依赖.
[前缀 cache](serving.zh.md#前缀-cache) 可以使用来自未满块的 checkpoint.
保留源块, 直到续算结果没有校验错误.

修改 kernel 前使用[算子验证框架](../development/operators).
该框架提供固定数值 fixture, 直接 PyTorch 参考实现和理论延迟倍率门槛.
常规数值测试读取仓库 fixture, 不执行 PyTorch.
保持[精度策略](operators/precision.zh.md)和[全模型门槛](operators/evaluation.zh.md).
实现改变后不得自动修改门槛.

FA 使用 BF16 attention 操作数和缓存状态. GDN 使用 FP32 保存循环状态.
FP8 投影使用 checkpoint 中的 E4M3 权重和 FP32 block scales.
[模型契约](operators/model.zh.md#中间精度)定义中间计算精度.

HTTP 计费使用串行 kernel 分组和当前精度规则.
融合或执行边界变化时, 使用[性能计算](performance.zh.md#下界计算).
计费变化时, 算子倍率门槛保持已批准的最大延迟.

保持长输出 efficiency 高于 40%. 每种配置至少测量两次.
每种实测耗时为 5 秒或更长的短输出配置, 其 efficiency 必须为 40% 或更高.
继续优化大规模 prefill 操作.

交付前执行适用的[测试范围](testing.zh.md).
同时执行全部 24 组 [HTTP 配置](performance.zh.md).
将测量产物保存在仓库外.
将主机设置放入被忽略的 `LOCAL.md`.
