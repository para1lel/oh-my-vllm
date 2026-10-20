# 项目术语

| 术语 | 词性 | 项目含义 |
|---|---|---|
| `CLI` | 名词 | 命令行接口. |
| `DEALER` | 名词 | Rust 和 Python 进程使用的 ZMQ socket 类型. |
| `FA` | 名词 | 目标模型中的全注意力. |
| `GDN` | 名词 | 目标模型中的门控增量网络. |
| `KV cache` | 名词 | 模型 token 的 key 和 value 状态. |
| `RPC` | 名词 | 跨进程边界的请求及对应回复. |
| `SSE` | 名词 | 服务发送给客户端的 HTTP 事件流. |
| `checkpoint` | 名词 | 模型权重或已接受 token 对应的循环状态快照. |
| `commit` | 动词 | 将选定文件保存为新的 Git 修订版本. |
| `decode` | 名词 | worker 为新输出 token 执行的操作. |
| `draft` | 名词 | target 验证前的候选 token. |
| `execution plan` | 名词 | 特定算子的实现选择, 存储, 准备工作和执行条件描述. |
| `hook` | 名词 | Git 保存修订版本前启动的程序. |
| `inference` | 名词 | 根据输入 token 产生输出 token 的模型计算. |
| `msgpack` | 名词 | 进程边界使用的二进制消息格式. |
| `operator workspace` | 名词 | 持有算子计算所用临时 GPU 张量和可变 native 执行资源的对象. |
| `prefill` | 名词 | worker 对 prompt 或已接受历史执行的操作. |
| `prefix reuse` | 名词 | 对相同的已接受 token 前缀复用缓存状态. |
| `push` | 动词 | 将 Git 修订版本发送至远端仓库. |
| `rollback` | 名词 | 移除尚未接受的 token 或 cache 状态. |
| `state commit` | 名词 | 接受已验证 token 历史和 cache 状态的操作. |
| `worker` | 名词 | 准备 token 并返回计算结果的 Python 进程. |
| `workspace` | 名词 | 由一个根 Cargo manifest 管理的一组 Rust 包. |
