# 运行时所有权

## 请求流程

请求经过准入, 准备, 调度, 执行和已验证提交.
每个阶段有一个所有者. GPU 工作不能发布未验证的前缀.
取消在已完成算子的边界生效; worker 失败时先使其状态失效, 然后替换.

读取[服务准入](../../crates/zmq-worker/src/serving/), [请求准备](../../python/oh_my_vllm/worker/serving.py),
[worker 执行](../../python/oh_my_vllm/worker/model_runner.py)和[调度器提交](../../crates/scheduler/src/lib.rs).
[服务契约](../serving.zh.md)定义可观察的 API 行为.

## Rust 模块

Rust 负责请求寿命, 已接受历史, 调度预算, 逻辑槽位和响应交付.
[调度器](../../crates/scheduler/src/lib.rs), [缓存协调](../../crates/kv-cache/src/coordinator.rs),
[子进程所有权](../../crates/zmq-worker/src/client.rs)和[服务资源](../../crates/zmq-worker/src/serving/resources.rs)
是实现入口.

## Python 模块

Python 负责分词, grammar 和采样状态, 模型 tensor, graph 存储及 GPU 执行.
[Bridge](../../python/oh_my_vllm/worker/zmq_bridge.py), [batch plan](../../python/oh_my_vllm/worker/batch_plan.py),
[目标模型](../../python/oh_my_vllm/models/qwen.py)和[采样器](../../python/oh_my_vllm/worker/sampler.py)
呈现这些边界. Python 使用 Rust 提供的逻辑 ID.

## 传输合同

进程通过 ZMQ DEALER socket 交换 msgpack 消息.
关联回复必须属于有效请求和 worker epoch. 已取消准备的迟到结果不能创建活跃状态.

Wire 字段和默认值来自 [Rust 协议](../../crates/zmq-worker/src/protocol.rs)
和 [Python 协议](../../python/oh_my_vllm/worker/protocol.py).
集成测试验证共同边界. 此处不维护另一份字段列表.
