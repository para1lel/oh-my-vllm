# 模型缓存策略

## 范围

[Rust 调度器](../../crates/scheduler/src/lib.rs) 使用模型选择的缓存策略.
[策略接口](../../crates/scheduler/src/policy.rs) 通过 Rust 关联类型提供缓存操作.
共享调度不解释缓存组名, 块表或物理槽布局.
它控制请求队列, token 预算, 抢占, 取消和输出验证.

默认策略是现有 Qwen `HybridCoordinator`.
[Qwen 实现](../../crates/scheduler/src/policy/qwen.rs) 保留其块池和分配算法.
它支持共享池, 或 FA 和 GDN 各自使用一个池.
本次抽取不使用[状态 arena](state-arena.zh.md) 替换这些池.

服务仍只选择 Qwen.
Qwen worker 消息保持现有字段名和值.
可选事务策略为[类型化 rank 执行](typed-execution.zh.md) 提供所有权转移.
DeepSeek 缓存布局和已接受状态选择仍是后续工作.
此接口不会完成阶段 2, 也不会启用 DeepSeek 执行.

## 模型操作

策略在调度器添加请求前检查准入需求.
它提供前缀 hash, 前缀查找, prefill 对齐和 prefill 分配单位.
它还提供分配, 前缀驱逐检查, 步骤分配和缓存资源计数.
调度器不计算 FA 页需求或 GDN 槽需求.
Qwen 准入和部分检查点规则保留在其策略中.

`PrefixHit` 包含 token 长度和模型选择的命中元数据.
默认命中长度为零, 且没有新的前缀状态.
分配失败后, 策略可以提供回退前缀.
对 Qwen, 部分前缀分配失败时, 此回退使用对齐前缀.

`ScheduledRequest<State>` 包含共享请求字段和模型选择的状态分配.
分配向 worker 提供状态位置.
它不拥有这些位置的存储.
策略必须保持所有被引用存储可用, 直到执行生命周期允许释放.
Qwen 分配包含其现有 FA, GDN, copy 和 checkpoint 元数据.
Qwen client 将这些分配转换为现有 wire 字段.

## 策略义务

策略回调是受信任的实现代码, 不是 worker 输入.
Prefill 对齐后的 token 数不得大于传入数量.
前缀命中必须至少保留一个已接受历史 token 用于计算.
分配失败不得发布新的已接受前缀, 也不得保留不可用的请求分配.
失败的主机加载预留必须释放其临时 pin 和请求分配.

调度器在对任何请求调用 `commit_cache` 前验证完整 worker 输出.
策略使用已验证的请求进度选择已接受状态.
状态分配和复制的块值不能授权新的存储所有权.
基于 arena 的策略必须通过活动所有者选择存储, 而不是通过 wire 值.
调用方必须在向调度器提供结果前证明设备完成.
分布式调用方还必须通过 [rank gate](rank-control.zh.md) 证明全部 rank 完成.

`free` 仅在调用方允许释放时释放请求所有权.
现有 worker 生命周期控制执行取消和 worker 替换.
请求取消后, 主机缓存保留待处理迁移直到迁移完成.
本次抽取不向 `Scheduler::abort` 添加设备完成检查.

## 主机迁移

[主机缓存](../../crates/scheduler/src/cache.rs) 保存三种不同的策略类型.
`Snapshot` 是可复制的主机快照元数据, 不是活动 GPU lease.
`Transfer` 持有迁移专用所有权, 没有 `Clone` 要求.
`TransferPlan` 提供 wire 元数据, 不含存储所有权.
调度器不解释这些模型选择的字段.

策略准备请求存储, 前缀存储和加载预留.
Qwen 请求存储保持现有请求所有权.
Qwen 前缀存储 pin 源块表.
Qwen 加载在安装和下一步分配前 pin 预留的目标块表.
这些 pin 在 DMA 可以使用目标存储期间保持该存储可用.
当分配淘汰之前的 GDN 检查点时, 这些 pin 生效.

迁移完成通过验证后, `finish_transfer` 释放迁移专用 pin 并返回快照元数据.
Deferred 和 capacity 响应也经过此释放步骤.
这些响应必须证明没有迁移仍在使用缓冲区.
请求所有权的生命周期与迁移所有权不同.
调度器现有的完成和取消分支控制请求释放.
已取消的迁移不能仅因请求被取消而释放其所有者.

`CacheOperation<TransferPlan>` 保留公共迁移标识和展平的模型计划.
Qwen 计划保留 `fa_blocks` 和 `mamba_blocks` wire 字段.
Drop 操作没有迁移所有者, 使用策略的空计划.
可复制的计划和快照不能替代待处理迁移所有者.

## 边界

Qwen 服务仍使用现有缓存诊断键.
共享调度 trace 记录策略的资源计数.
Python 迁移 helper 仍验证成对的 Qwen 缓存格式.
其他模型族需要自己的 wire 转换和迁移验证.

[阶段计划](plan.zh.md) 控制后续 typed-state 和分布式模型工作.
[验证计划](validation.zh.md) 控制数值和性能验收.
泛型 Rust 类型不能证明模型的数学, 容量或设备安全性.
