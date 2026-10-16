# 推理 worker

将 ZMQ 和 msgpack 传输放在 bridge 中.
仅当模型路径为 `stub` 时使用占位实现.
模型加载或计算失败必须返回错误.

使用运行时包和原生 CUDA 模块. 不导入开发工具.
Rust 管理逻辑 cache 标识. Python 保存 tensors 和私有状态.
MTP 拒绝后, 约束状态必须与已接受 tokens 一致.

仅含 decode 请求的 batch 使用 CUDA Graph 执行.
不使用 PDL.
保持 graph 缓冲区地址固定. Replay 前更新元数据.

添加测试前, 向维护者说明其必要性.
