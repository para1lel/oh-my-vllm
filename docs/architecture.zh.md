# 进程边界

## 控制和状态

[服务](../crates/zmq-worker/src/lib.rs)负责 CLI, HTTP API, worker 生命周期和传输.
[调度器](../crates/scheduler/src/lib.rs)负责准入, 请求队列, token 预算, 抢占和经过验证的状态提交.
[缓存策略](cache-policy.zh.md)提供模型专属分配, 前缀查询, 状态分配和恢复.
[KV crate](../crates/kv-cache/src/lib.rs) 提供逻辑存储和所有权机制.

调度遵循[运行时契约](requirements.zh.md)中的优先级.
调度器验证全部输出行后才修改已接受历史或发布前缀.
事务失败不能发布未经验证的前缀.
抢占保留已接受历史, 丢弃新草稿.
物理状态释放还取决于设备和传输完成状态.

## 模型执行

[Bridge](../python/oh_my_vllm/worker/zmq_bridge.py) 负责 ZMQ DEALER 和 msgpack 帧.
[注册表](../python/oh_my_vllm/models/registry.py)在 GPU 导入前选择模型适配器.
各适配器通过[公共契约](adapters.zh.md)提供计算, 编解码器, 物理缓存和已接受状态选择.
显式 `stub` 后端仅提供服务测试行为.

模型权重遵循适配器的设备和主机放置计划.
按顺序执行的层可以复用临时缓冲区.
录制的 graph 保持全部引用的缓冲区和资源所有者.
较大的调度步骤可以使用较小的原生 batch.
[执行规则](execution.zh.md)规定分片选择, rank 协调和 Graph 资源所有权.

在验证完成前, target 和 proposal 分布使用独立存储.
只有已接受 token 修改约束和请求历史.
最后一块之前的 prefill chunk 不返回输出 token.
完成的生成步骤返回已接受 proposal 和一个 target token, 并遵守请求停止条件.
新 proposal 不属于已接受历史.

[单 worker 协议](../crates/zmq-worker/src/protocol.rs)通过 RPC ID 和已调度请求结果关联回复.
[Rank 协议](distributed.zh.md)增加组 epoch, 步骤, 事务和全 rank 完成检查.
模型专属 client 将逻辑分配转换为选定的 wire 布局.

## GPU 计算

| 部分 | 职责 |
|---|---|
| Python 适配器 | 按依赖顺序调用算子并拥有模型状态 |
| 原生算子 | 按选定精度和状态规则计算数值 |
| 设备资源 | 分配存储并管理 stream, Graph, lease 和释放 |

[算子设计](operators/design.zh.md)将语义接口与 kernel 和库边界分离.
Eager 执行和 Graph 录制使用相同的原生算子.
完整操作有收益且所有权规则成立时, 融合可以跨内部阶段进行.

## 失败和资源

服务在读取请求体或复制存储历史前预留请求数量和字节容量.
Stream 和已存响应都有数量及字节限制.
取消在安全操作边界释放状态.
Worker 失败会使相关请求失败, HTTP listener 继续运行.

恢复先移除旧 worker 和逻辑身份, 再在新 epoch 中准入.
恢复不重放输出或工具调用.
清理停止自有进程组并移除其 IPC 路径.
请求, 步骤和关闭截止时间具有不同范围.
命令和限制见[服务指南](serving.zh.md).
