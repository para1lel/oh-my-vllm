# 模型缓存策略

[Rust 调度器](../crates/scheduler/src/lib.rs)使用显式的模型专属[缓存策略](../crates/scheduler/src/policy.rs).
共享调度控制队列, token 预算, 抢占, 取消和输出验证.
它不解释模型分组名称, block table 或物理 slot 布局.

## 模型操作

策略提供准入需求, 前缀查询, 有效 prefill 长度, 分配单位, 预留, 分配信息和资源计数.
有效 prefill 长度反映状态正确性约束.
调度偏好不能排除可用的零散 token.
共享调度器应用[运行时契约](requirements.zh.md)中的优先级.

`PrefixHit` 包含 token 长度和模型选择的元数据.
默认命中长度为零, 不含新前缀状态.
前缀命中至少留下一个已接受历史 token 供计算.
分配失败后, 策略可以提供回退前缀.

`ScheduledRequest<State>` 包含共享请求字段和模型专属分配信息.
分配信息标明状态位置, 不拥有存储.
策略保持被引用存储可用, 直到执行和传输允许释放.
分配失败不能发布前缀, 也不能保留不可用的请求分配.

## 提交和所有权

调度器验证全部 worker 输出后, 才能对任何请求调用 `commit_cache`.
策略根据经过验证的进度选择已接受状态.
复制的分配信息和 block 值不能授权存储所有权.
使用 arena 的策略通过[状态 arena](state-arena.zh.md) 中的活跃所有者选择存储.

调用方在向调度器提供结果前证明设备完成.
分布式调用方还证明[全部 rank 完成](distributed.zh.md#事务和发布).
策略回调是可信实现代码, 不是 worker 输入.
它们必须在转移所有权前验证完整身份集合.

只有调用方允许时, `free` 才释放请求所有权.
仅调用 `Scheduler::abort` 不能证明设备完成.
请求取消后, 待完成的主机传输继续保持所有者, 直到传输完成.

## 主机传输

[Host cache](../crates/scheduler/src/cache.rs) 使用三个策略类型:

| 类型 | 含义 |
|---|---|
| `Snapshot` | 可克隆的主机元数据, 不是活跃 GPU lease |
| `Transfer` | 传输所有权, 不要求实现 `Clone` |
| `TransferPlan` | Wire 元数据, 不拥有存储 |

策略准备请求存储, 前缀存储和加载预留.
加载预留失败时释放临时 pin 和请求分配.
传输 pin 保护源和目标存储, 直到异步访问完成.

验证完成后, `finish_transfer` 释放传输 pin 并返回 snapshot 元数据.
延迟和容量响应也使用此释放步骤.
它们必须证明没有传输继续使用缓冲区.
请求所有权与传输所有权具有独立生命周期.

`CacheOperation<TransferPlan>` 保持公共传输身份和展开的模型计划.
Drop 操作没有传输所有者, 使用策略的空计划.
可克隆计划和 snapshot 不能替代待完成传输的所有者.

## 实现

[Qwen 策略](../crates/scheduler/src/policy/qwen.rs)使用 `HybridCoordinator` 和成对 FA/GDN 分配.
其[适配器缓存指南](models/qwen38/cache.zh.md)规定物理恢复和 wire 字段.
[事务策略](../crates/scheduler/src/policy.rs)将类型化存储与 rank 执行连接.
通用策略接口不能证明模型数学, 容量或设备安全.
