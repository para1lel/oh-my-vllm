# 工作状态

在 `main` 上工作.
Worker 使用[算子调度](execution.zh.md): 包含 prefill 的 Execute 消息使用 Eager, 仅含 decode 的消息使用 CUDA Graph 重放.
Native 构建入口是 `development.native_build`.
FA 的 BF16 cache 状态和派生 FP8 操作数使用相同的逻辑页所有权.
Attention 精度见[模型契约](operators/model.zh.md#中间精度).

[数值工具](operators/evaluation.zh.md)将全模型 logits 与冻结的 [PyTorch FP64 reference](../development/model/reference.npz) 比较.
保留其身份检查和数值门槛.
使用 24 种 [HTTP 配置](performance.zh.md)测量时间和标称资源效率.
这些测量使用 serving 默认采样参数和 auto 工具约束.
测量产物保存在仓库外.

后续优化只限于有明确性能收益的变更.
保持数值精度, 状态所有权和服务行为.
变更影响 MTP, vision 或长上下文时, 扩展全模型输入.
主机配置保存在被忽略的 `LOCAL.md` 中.
