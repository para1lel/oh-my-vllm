# ADR-001: ZMQ DEALER socket

日期: 2026-09-19. 状态: accepted.

## 背景

单个 Rust driver 需要与一个 Python 子进程交换双向消息.
固定的 `zeromq` crate 没有 `PairSocket` 或 `pair-socket` feature.
REQ/REP 会强制严格交替回复, 限制流水线.
其他 transport 也会改变 Python 接口.

## 决策

两侧都使用 DEALER, 启用 `tokio-runtime` 和 `ipc-transport`.
Rust 绑定 IPC endpoint; Python 连接.
使用带显式 type 和 RPC 关联的 msgpack body.

## 影响

每个 driver 保持一个 worker, 每个活跃进程使用独立 socket 路径.
DEALER 本身不限制为单个 peer.
不将另一个 worker 连接到相同 endpoint.
用所属进程清理流程处理遗留 IPC 文件.
只有确认没有活跃 owner 使用后才删除遗留路径.
当前 framing 和异步 prepare/execute 处理见 [架构](../architecture.zh.md#传输合同).
