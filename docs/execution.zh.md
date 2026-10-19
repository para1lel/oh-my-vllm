# GPU 执行

## 算子调度

Rust 负责请求, 调度, 逻辑 KV 身份和 state commit.
Python 通过 CUDA 算子及现有 ZMQ DEALER 和 msgpack 边界执行模型.
[算子边界](operators/fusion.zh.md)规定计算和私有状态所有权.

Worker 为每条 Execute 消息选择一次执行方式.
只要任一已调度行包含 prefill, 该 Execute 的所有分片都使用 Eager 执行.
仅含 decode 的 Execute 使用 CUDA Graph 录制和重放.
此规则也控制采样, MTP context, proposal 和 acceptance.
Vision 和权重准备使用 Eager 执行.

Eager 和 CUDA Graph 执行使用相同的 native 算子.
Python 按模型依赖顺序调用这些算子.
CUDA kernel 可以融合操作并在内部使用矩阵 tile.
没有 tile task 调度器, task DAG 或 persistent task engine.

## Graph 资源

Workspace 拥有已录制操作的权重, cache 引用, metadata 和临时 tensor.
Warmup 在录制前准备存储.
采样和 MTP acceptance 在 warmup 后恢复 RNG 状态.
Graph 存在期间, 缓冲区地址保持固定.
Metadata 在重放前于调用者的 stream 上更新.
重放使用同一个 stream.

录制使用独立的 nonblocking stream, 并在退出时恢复调用者的 stream.
录制失败会释放失败前创建的资源.
录制成功后, 资源 lease 转交给 workspace.
关闭时先等待 GPU 工作, 再销毁 graph executable, 最后释放其 owner 和 lease.
执行失败沿用现有 worker 移除和状态清理规则.

## 数值与状态契约

保持[模型契约](operators/model.zh.md)规定的 checkpoint 值, FP8 scale, 操作数类型和舍入位置.
保持完整 attention context, 262144-token 总长度上限和 784-token checkpoint.
验证允许 state commit 前, candidate 写入保持私有.
Acceptance 选择状态槽, 不复制完整循环状态.
CPU 约束状态跟随已接受的 token.

使用 PyTorch FP64 [全模型 logits 对照](operators/evaluation.zh.md).
不使用单算子容差.
保留服务, graph 资源, 内存安全和状态所有权检查.
24 种 HTTP 用例不能替代这些检查.

## HTTP 性能

保留 [HTTP 流程](performance.zh.md)中的 24 种配置和完整请求计时范围.
此比较继续使用 32768 的 token 预算, 最多同时运行 4 个请求.
最终验收时, 每种配置的效率必须达到 50% 或更高.
中间阶段变更可以报告尚未测量或未达标的配置.

使用 [NVIDIA 数据表第 8 页](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf)中的 HGX B200 dense 数值.
吞吐为 HBM 7.7 TB/s, FP8 4.5 PFLOP/s, BF16 2.25 PFLOP/s 和 TF32 1.1 PFLOP/s.
标量吞吐为 FP32 75 TFLOP/s 和 FP64 37 TFLOP/s.
不使用 sparse 数值, GB200 数值或实测校准值替代.

```text
reference_seconds = max(
    necessary_bytes / HBM_rate,
    FP8_work / FP8_rate + BF16_work / BF16_rate + TF32_work / TF32_rate,
    FP32_work / FP32_rate,
    FP64_work / FP64_rate,
)
efficiency = reference_seconds / HTTP_seconds
```

这是基于明确数学和内存假设的资源下界估计.
它不是最小物理延迟的证明.
计算先汇总完整 trace 的工作量, 再取最大值.
不累加独立算子的延迟估计.

带版本的参考调度确定逻辑模型 pass.
请求顺序, decode 优先级, prefill 分配, 短尾部和 checkpoint 规则必须确定.
绑定身份的 benchmark manifest 将请求序号映射到响应和 trace 的请求 ID.
计费前验证 ID 唯一性, 成员完整性及配置的输入/输出长度.
HTTP 到达顺序不得改变参考分组.

Trace 提供 token 位置, 完整 context, proposal, verification 行和实测 acceptance 数.
参考调度必须遵守这些数据的因果依赖.
不能虚构 proposal, 也不能合并不相容的 step.

每个参考 pass 对每个必要权重和 scale 区域计数一次, 该 pass 的 batch 共享此计数.
每次必要的 KV 或状态访问在对应参考位置计数一次.
后续 decode step 使用同一历史时, 历史读取仍要重复计数.
固定的参考算法和精度决定算术工作量.
满足数值要求的实现算法可以与计费参考算法不同.

增加 tile, split-K 分区, launch, padding, 重复计算或临时复制不能增加计分工作量.
排除 fusion 可以消除的中间存储.
明确列出普通标量算术及全部省略项.
不为缺少资源模型的操作虚构吞吐系数.

产物将每行结果关联到源码, 二进制, checkpoint, device, 计费版本和计时身份.
旧结果不能作为新计算方式的验收证据.
两种语言的 README 只包含规定的 24 行性能表.
