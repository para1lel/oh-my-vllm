# 模型缓存策略

[Rust 调度器](../crates/scheduler/src/lib.rs)使用显式的模型专属[缓存策略](../crates/scheduler/src/policy.rs).
共享调度控制队列, token 预算, 抢占, 取消和输出验证.
它不解释模型分组名称, block table 或物理 slot 布局.
KV 缓存状态保存在 GPU 上. 没有主机缓存协议或主机存储预算.

## 模型操作

策略提供准入需求, 前缀查询, 有效 prefill 长度, 预留, 分配信息和资源计数.
有效 prefill 长度遵循模型状态约束.
调度偏好不能排除可用的零散 token.
共享调度器应用[运行时契约](requirements.zh.md)中的优先级.

`PrefixHit` 包含恢复长度和模型选择的元数据.
分配失败后, 策略可以提供回退前缀.
模型的缓存契约允许时, 完整 prompt 命中可以使用保存的 hidden 值.

`ScheduledRequest<State>` 包含共享请求字段和模型专属分配信息.
分配信息标明状态位置, 不拥有存储.
策略 pin 被引用存储, 直到执行和必要 copy 允许释放.
分配失败不能发布前缀, 也不能保留不可用的请求分配.

## 提交和收尾

调度器验证全部 worker 输出后, 才能对任何请求调用 `commit_cache`.
策略根据经过验证的进度选择已接受状态.
复制的分配信息和 block 值不能授权存储所有权.
使用 arena 的策略通过[状态 arena](state-arena.zh.md) 中的活跃所有者选择存储.

调用方在向调度器提供结果前证明设备完成.
分布式调用方还证明[全部 rank 完成](distributed.zh.md#事务和发布).
[Wire 契约](adapters.zh.md#执行协议)将结果绑定到完整执行身份.

`completion_plan` 选择前向计算, 端点导出或已完成端点.
`validate_cache_step` 在提交前检查模型专属状态证据.
`finish_request` 区分成功, 取消和失败.
只有验证后的成功才能发布成功请求尾.
[会话注册表](../crates/scheduler/src/session.rs)为此发布提供准入顺序.

## 退休

只有调用方允许时, `free` 才释放请求所有权.
仅调用 `Scheduler::abort` 不能证明设备完成.
已提交请求取消时, 如果执行仍在途, 请求会进入 `Retiring`.
[生命周期实现](../crates/scheduler/src/lifecycle.rs)只在经过验证的完成清除执行身份后释放 pin.

物理释放前, kernel 工作, 辅助 stream, rollback, 导出和 copy 必须完成.
已完成请求和被抢占请求的通知仍负责释放 worker 状态.
Worker 替换先证明旧 worker 已退出, 再在新 epoch 中重置状态.
必要的 token 和 metadata 传输遵循[执行规则](execution.zh.md).

## 实现

[Qwen 策略](../crates/scheduler/src/policy/qwen.rs)使用独立的 checkpoint 池和 KV page 池, 配合一棵 token radix tree.
其[缓存指南](models/qwen38/cache.zh.md)规定独立的前缀位置, 精确会话尾和物理存储.

[DeepSeek 策略](../crates/zmq-worker/src/models/deepseek_v41/cache.rs)使用自己的驻留成对 block 分配和已接受状态规则.
它不使用 Qwen 的容量比例, checkpoint 间隔或精确尾计算规则.
其[缓存指南](models/deepseek_v41/cache.zh.md)规定模型存储.
通用策略接口不能证明模型数学, 容量或设备安全.
