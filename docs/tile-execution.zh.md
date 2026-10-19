# Tile 执行设计

## 范围与效力

本设计规定 `nightly` 的迁移要求.
它取代旧的算子性能门槛, 实测峰值规则及强制算子测量要求.
[运行时契约](requirements.zh.md)继续规定服务功能, 进程所有权和数值行为.

最终实现必须通过 tile 依赖和 MegaKernel 执行全部 GPU tensor 计算.
这适用于准备, 初始化, 普通 decode, prefill, MTP4, vision, 采样和状态计算.
在新的 Python 接口后包装普通算子 launch, 不代表完成该算子的迁移.

迁移可以分阶段进行, 过程中不设性能门槛.
每个交付阶段必须通过适用的正确性检查和独立审查.
最终验收分别检查覆盖范围, 正确性和 [HTTP 性能](#http-性能).

## 计算边界

| 范围 | 最终覆盖 |
|---|---|
| 语言模型 | Embedding, 64 层, 归一化, 量化, 投影, 注意力和词表投影 |
| Prefill | FA, GDN 分块计算, metadata, 归约和 checkpoint |
| MTP4 | Context, 四次提案, verification, acceptance, 状态选择和端点 gather |
| 采样 | Penalties, GPU 上的 mask 应用, temperature, 排序, 分布, 随机选择和 RNG counter |
| Vision | GPU 像素缩放, 归一化, patch, 位置, 27 层, merge 和 embedding gather |
| 准备 | GPU 权重转换, scale 准备, 重排和激活表 |
| 辅助计算 | 索引 gather/scatter, tensor 初始化, 状态导出和 GPU metadata 计算 |

CPU tokenization, 媒体解码, shape 选择, grammar, 停止规则和请求控制可以继续在 CPU 上执行.
以后若把这些计算放到 GPU, 必须使用 tile task.

CUDA 分配, host 准备的 view, stream, event, graph 和同步属于基础设施.
不改变字节的 H2D, D2H 和 D2D 传输可以使用 CUDA 传输操作.
TMA 和 DMA 可以为 task 提供数据.
这些操作必须遵守数据依赖和缓冲区生命周期.
模型 tensor 初始化属于计算, 即使其输出全部为零.

## 所有权与执行

Rust 负责 admission, 请求调度, 逻辑 KV 身份和 state commit.
Python worker 拥有 device tensor, 执行计划和私有状态槽.
ZMQ DEALER 和 msgpack 保持现有进程边界.

准备阶段构建静态或参数化 task 有向无环图 (DAG).
每个计划在其生命周期内使用固定缓冲区地址.
运行时 metadata 提供长度, page table, 状态槽, 有效行和执行 epoch.
各层共享 task 实现.
层编号和权重地址作为 task 参数.

初始引擎使用 CUDA C++ 和 CUTLASS device 组件.
后续 task 实现可以使用 CUDA C++ 或 CuTe DSL.
单个 persistent kernel 是起点, 不是限制.
多个专用 persistent kernel 可以共享同一套 task 和 event 契约.
Kernel 类别, 算法和调度可以随研究及性能证据调整.

全部推理提交在对应 Execute 边界结束.
准备计算可以在初始化或准备阶段提交.
本设计不包含永久运行的 GPU 服务循环.
纯 decode 执行继续通过 CUDA Graph 提交, 计划缓冲区地址保持稳定.

首个完整里程碑覆盖从 embedding 到词表投影的普通 decode.
其 CPU 约束处理和独立 GPU 采样起初可以使用现有路径.
最终验收前, 采样必须接入 tile 执行.
Prefill, MTP4 和 vision 遵循相同迁移规则.

## Task, 内存与进展

每个 task 标识操作, 输入输出区域, 精度规则, 状态权限和资源类别.
Graph 包含数据依赖, 状态版本依赖和缓冲区生命周期依赖.
仅靠物理地址重叠不能确定这些依赖.

初始 task 单位包括融合投影 tile, 注意力分区, 状态 head tile 和归约.
小型局部操作可以放在其生产者或消费者 task 中.
Tile 维度, task 范围, 队列布局和 CTA 分工由实现决定.
这些选择不得改变模型契约.

Ready queue 的容量固定.
执行 CTA 只领取就绪 task.
Planner 在 launch 前拒绝依赖环.
它检查缓冲区容量和资源驻留条件.
当消费者不能执行时, 调度器和生产者必须仍有可推进的执行路径.
多个 persistent kernel 必须有明确的驻留和进展计划.

Task 完成必须晚于相关 TMA, MMA 和普通写入的完成.
Device scope 的 release/acquire 顺序使这些写入对消费者可见.
Epoch 标识复用的 event, descriptor 和队列项.
完成计数区分已入队, 执行中和已完成的 task.
队列溢出, metadata 无效和执行不完整都返回错误.

失败时不允许 state commit.
执行失败或超时后, 现有 worker 移除和 epoch 清理流程废弃可能被修改的状态.
重置队列不能使这些状态安全地重新使用.
取消继续使用现有操作边界.

## 数值与状态契约

保持 checkpoint 值, FP8 scale, 允许的操作数类型, 舍入位置和具名数值容差.
不通过修改容差来取得性能验收通过.
保持完整 attention context, 262144-token 总长度上限和 784-token checkpoint.
在模型契约规定的位置保持 FP32 live carry 和 BF16 持久状态.

Candidate 写入在验证允许 state commit 前保持私有.
Acceptance 选择状态槽, 不复制完整循环状态.
CPU 约束状态跟随已接受 token.
正常推理行为必须在不同 task 调度, 计划复用和合法 batch shape 下保持一致.

24 个 HTTP 性能配置不能替代数值, 状态, 媒体或随机采样检查.
移除旧计时工具时, 必须保留现有独立参考和状态检查.
新增检查应覆盖这些参考不能检测的故障, 例如 event 复用或调度器死锁.
新增每个测试前说明其必要性.

## HTTP 性能

保留 [HTTP 流程](performance.zh.md)中的 24 种配置和完整请求计时范围.
此比较继续使用 32768 的 token 预算, 最多同时运行 4 个请求.
最终门槛是每种配置至少达到 50%.
迁移提交可以报告尚未测量或未达标的配置.

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

## 研究参考

以下资料用于指导实现选择, 不增加验收门槛.

- [MPK, 2026-06-10 修订](https://arxiv.org/html/2512.22219v2): task/event graph 以及静态和动态结合的调度. 其寄存器限制说明专用 kernel 类别具有价值.
- [Event Tensor, 2026-04-21 修订](https://arxiv.org/html/2604.13327v2): 参数化 event 和运行时 task 范围, 用于可变 shape.
- [Cohere, 2026-09-08](https://cohere.com/blog/megakernels): task descriptor, CTA 分工和有限 decode 提交. 其 H100 BF16 实现保留普通 prefill.
- [TileRT, 2026-05-21](https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html): persistent 执行和 tile 流水线. 完整内部调度器源码未公开.

这些工作不能证明本服务已完整覆盖模型, 也不能证明本服务可以加速.
在测量和数值检查支持时采用其中的机制.
[迁移计划](tile-migration.zh.md)规定实施顺序.
