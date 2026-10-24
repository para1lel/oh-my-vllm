# 共享 worker 机制

将 ZMQ 和 msgpack 传输放在 bridge 中.
仅当模型路径为 `stub` 时使用占位实现.
模型加载或计算失败必须返回错误.

使用运行时包和原生 CUDA 模块. 不导入开发工具.
Rust 管理逻辑 cache 标识. Python 保存 tensors 和私有状态.
推测拒绝后, 约束状态必须与已接受 tokens 一致.

将模型计算和 token 语法放在所选模型适配器中.
所有权和共享契约参见[适配器指南](../../../docs/adapters.zh.md).

遵守[执行规则](../../../docs/execution.zh.md)中的 Eager 和 CUDA Graph 选择规则.
拆分 batch 前, 根据完整 Execute 消息选择执行方式.

添加测试前, 向维护者说明其必要性.
