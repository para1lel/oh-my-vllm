# 进程边界

## Rust

[KV crate](../crates/kv-cache/src/lib.rs) 负责块标识, 引用计数, LRU 和前缀复用.
[调度器](../crates/scheduler/src/lib.rs) 负责请求准入, decode 优先级, prefill 分块, 抢占和推测 token 的状态提交.

[服务 crate](../crates/zmq-worker/src/lib.rs) 负责 CLI, HTTP API, worker 生命周期及传输.

调度器先选择 decode 请求, 再选择 prefill 请求.
它在 prefill 请求之间分配剩余 token 预算, 并按 cache 边界调整.

每个调度步骤预留 FA 和 GDN 状态.
调度器先验证全部输出行, 再更新 token 历史或发布前缀.

事务出错后, 调度器不能发布未验证的前缀.
被抢占请求保留已接受的历史. 抢占会移除新 drafts.

## Python

[bridge](../python/oh_my_vllm/worker/zmq_bridge.py) 负责 DEALER socket 和 msgpack 帧.
[占位实现](../python/oh_my_vllm/worker/placeholder.py) 负责 token 准备和执行模拟.

后续推理后端沿用此进程边界.
逻辑 cache 归 Rust 管理. Python 保存 tensors. GPU 计算由 Python 执行.

[协议类型](../crates/zmq-worker/src/protocol.rs) 给出 Init, Register, Prepare, Execute, Abort, Stats 和 Shutdown 消息的字段.
回复使用对应 RPC ID. 每个已调度请求有一行结果.

最终分块之前的 prefill 分块不返回输出 token.
最终 prefill 或 decode 步骤返回已接受 drafts 加一个 target token.

已接受的历史不包含新 drafts.

## 失败与资源

服务在读取 body 或复制已存储历史前, 先预留请求数量和字节容量.
流式缓冲和已存储响应都有数量及字节限制.

取消操作在操作边界释放请求状态.
worker 故障终止受影响的请求. HTTP listener 保持运行.

恢复时先移除一个 epoch 的 worker 和逻辑 cache 标识, 再接纳下一个 epoch 的请求.
服务不重放已经输出的内容或工具调用.

worker 清理会停止其进程组, 并移除自己创建的 IPC 路径.
请求, 步骤和关闭期限的范围不同.

命令和占位限制见[服务指南](serving.zh.md).
