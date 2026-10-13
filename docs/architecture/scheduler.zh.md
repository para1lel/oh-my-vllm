# 调度与缓存概念

## 调度与分配

Continuous batching 合并活跃请求中已就绪的 token 工作.
Decode 保持推进, 对齐 chunk 共享剩余 prefill 预算.
Checkpoint 边界支持未来复用; 任意部分状态不能替代该 checkpoint.

[调度器](../../crates/scheduler/src/lib.rs)负责队列策略和 token 预算.
[请求](../../crates/scheduler/src/request.rs)区分已接受历史与未验证草稿.
[分配](../../crates/kv-cache/src/coordinator.rs)在执行前协调各 group.

容量失败可以使较晚请求被抢占后重算. 保留已接受历史, 不保留未验证草稿.
Worker 回复必须先通过验证, 然后改变历史, 已计算进度或可复用身份.
实现见[提交边界](../../crates/scheduler/src/lib.rs#L619) <!-- source: pub fn update(&mut self, mut worker: -->.

## 缓存 group 与前缀复用

FA page 保存 attention 历史. GDN checkpoint 保存 recurrent 历史.
复用前缀必须一致. 单独更长的 FA 前缀不能恢复缺失的 recurrent state.
引用保护活跃槽位; LRU 淘汰可以回收未使用的缓存槽位.

[前缀哈希](../../crates/kv-cache/src/hash.rs)将 token 内容链接到之前的前缀.
[Group 状态](../../crates/kv-cache/src/group.rs)区分暂存身份与已完成缓存数据.
[Block 所有权](../../crates/kv-cache/src/pool.rs)控制引用和淘汰.

只有已完成且被接受的计算可以发布缓存身份.
取消删除未提交身份, 有效的已提交前缀可以继续复用.
逻辑槽位复用和 GPU graph 存储具有不同的所有者与预算.

固定 block 大小, 层数, 容量和状态精度由[模型/上下文需求](../requirements/core.zh.md)定义.
