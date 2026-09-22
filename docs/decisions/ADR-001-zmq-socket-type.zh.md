# ADR-001 — ZMQ socket 类型：用 DEALER 代替 PAIR

**日期：** 2026-09-19
**状态：** 已接受

## 背景

Rust 二进制需要与单个 Python 子进程建立双向、一对一 IPC 通道。自然的 ZMQ 类型是 `PAIR`；初始设计使用 `PairSocket` 和 Cargo 的 `pair-socket` feature。

基于 `zeromq = "0.6"` 实现 `zmq-worker` 时发现，该版本既没有 `PairSocket` 类型，也没有 `pair-socket` feature，使用会导致编译错误：

```
error[E0432]: unresolved import `zeromq::PairSocket`
```

zeromq 0.6.0 提供 `DealerSocket`、`RouterSocket`、`PubSocket`、`SubSocket`、`PushSocket`、`PullSocket`、`ReqSocket` 和 `RepSocket`，没有 `PairSocket`。Cargo feature 包括 `tokio-runtime`、`ipc-transport`，没有 `pair-socket`。

## 决策

Rust 和 Python 两端都使用 `DealerSocket`。Rust bind，Python connect。

`Cargo.toml`：

```toml
zeromq = { version = "0.6", features = ["tokio-runtime", "ipc-transport"] }
```

`client.rs`：

```rust
let mut sock = zeromq::DealerSocket::new();
sock.bind("ipc:///tmp/oh-my-vllm.ipc").await?;
```

`zmq_bridge.py`：

```python
sock = zmq.Context().socket(zmq.DEALER)
sock.connect("ipc:///tmp/oh-my-vllm.ipc")
```

## 影响

**优点：**

- 使用 zeromq 0.6.0 实际提供的 feature，可以正常编译。
- 两端各只有一个 peer 时，DEALER-DEALER 是合法的一对一双向消息拓扑。

**缺点及注意事项：**

- DEALER 不强制一对一。如果第二个 Python 进程连接相同 IPC 路径，它可能收到发给第一个进程的消息。当前每次二进制调用只启动一个 Python worker，因此暂非实际风险，但 socket 类型本身不保证这一点。
- 某些配置下 DEALER 会在消息体前附加零长度身份帧。pyzmq 的 `zmq.DEALER` 默认 `recv`/`send` 不加该帧（与 `zmq.ROUTER` 不同），因此 msgpack 分帧无需调整。
- 崩溃不会清理 IPC 文件 `/tmp/oh-my-vllm.ipc`。已有文件时二进制不能启动；启动路径应在 bind 前清理 socket，或由调用方清理。当前已有文件时返回 EADDRINUSE。调用方需在运行之间执行 `rm -f /tmp/oh-my-vllm.ipc`。

## 考虑过的替代方案

**PAIR socket：** zeromq 0.6.0 不支持。若后续 crate 版本提供该类型，可重新评估。

**REQ/REP：** 严格交替发送/接收简化分帧，但阻碍流水线，且消息丢失后需要重置。

**nanomsg / nng：** crate 和 API 不同，Python 端也需由 `pyzmq` 改为 `pynng`，不值得为此迁移。
