# Python worker

worker 进程与推理实现保持相同的协议边界.
后端是占位实现. 不加载权重, 不使用 GPU.

bridge 负责传输. 占位后端负责 token 模拟.
worker 包使用运行时依赖文件.

Rust 负责请求准入, 调度和逻辑块标识.
worker 必须返回对应的 RPC ID, 并为每个已调度请求返回一项结果.

最终分块之前的 prefill 分块不返回输出 token.
已接受 token, 新 drafts 和请求完成状态使用不同的状态字段.

收到 Abort 或已完成请求通知后释放请求状态.
使用清晰名称和小函数.

新增测试前, 向用户说明必要性.
