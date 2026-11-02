# DSV4.1 TP4 物理内存门

日期: 2026-10-26

## 决策

DSV4.1 TP4 物理内存门仍未解决.

保持当前安全门. 不要提高 1 GiB 上限. 不要扣除加载前的 opaque 占用. 在独立解决方案完成前, 不要运行普通计算, CUDA Graph 或 memcheck.

当保守 kernel 字节数加 opaque 占用超过 1 GiB 时, 该门拒绝 rank. 此规则保护模型准备期间的物理内存所有权. 它不是性能上限.

## 已记录事实

当前本地证据记录了以下 TP4 准备值:

| 证据 | 记录值 | 含义 |
|---|---:|---|
| 每个 rank 的 typed arena | `79,177,028,608` bytes | 模型状态分配下界 |
| 加载峰值 | `79,178,595,328` bytes | 记录的最高加载分配 |
| Unit D callback charge 下界 | `175,558,569` bytes | 仅包含已知 callback charge |
| v6 记录中的设备总内存 | `191,505,498,112` bytes | 门检查时报告的设备总量 |
| v6 记录中的设备空闲内存 | `111,210,004,480` bytes | 门检查时报告的空闲内存 |
| v6 记录中的 allocated | `79,177,028,608` bytes | 显式分配计数器 |
| v6 记录中的 used | `80,295,493,632` bytes | driver 使用计数器 |
| v6 记录中的 opaque 占用 | `1,118,465,024` bytes | 保守 opaque 计数器 |
| v6 记录中的 kernel 字节数 | `0` bytes | 保守 kernel 计数器 |
| v6 超过 1 GiB 的数值 | `44,723,200` bytes | 门拒绝余量 |

v7 诊断增加了边界序列:

- 加载前 opaque 占用为 `1,115,422,720` bytes.
- 加载峰值 opaque 占用为 `1,118,995,456` bytes.
- rank 0 的加载后 opaque 占用为 `1,118,465,024` bytes.
- 加载增加了 `3,572,736` bytes.
- rank 0 的加载后增加了 `3,042,304` bytes.
- rank 2 和 rank 3 的加载峰值相同, 加载后增加了 `2,976,768` bytes.
- rank 1 在 group abort 前没有产生边界记录.

因此, 即使 `kernel_bytes=0`, v6 记录仍超过 1 GiB. v7 的加载前值也超过 1 GiB. 与加载前超额相比, 加载增量很小. 这些事实排除了 "只有权重加载导致首次超额" 这一充分解释.

checkpoint contract 记录了 checkpoint 中 12 个 Engram tensor 共 `203,073,076,240` bytes. 它也记录了 MP4 的行分片, 以及参考 loader 将参数加载到 CUDA. 该 contract 没有记录运行时 opaque 驻留, allocator 分类或 pinned host memory 传输 contract. 仅凭 checkpoint 大小不能解决此门.

## 预算状态

下表分开记录已测量值, 不完整的估计和未知预算.

| 预算类别 | 状态 | 数值或原因 |
|---|---|---|
| typed model-state allocation | 已测量下界 | 每个 rank 的 typed arena 为 `79,177,028,608` bytes. 它包含超出权重的 model state. |
| 仅权重的驻留 checkpoint | 未知 | 没有独立的仅权重测量. |
| 完整加载后驻留 model state | 未知 | 准备过程在 weights-complete 记录前停止. |
| checkpoint 逻辑总量 | 已测量逻辑总量, 不是驻留估计 | checkpoint 总计 `510,286,023,000` bytes. |
| 临时 workspace | 未知 | 没有 prepared 记录. |
| 通信器和 collective | 未知 | opaque 总量没有核对到这些所有者. |
| CUDA Graph reservation | 未测量 | 没有运行 graph capture. |
| pinned-host Engram 驻留 | 未知 | 参考 source 没有定义 pinned-host contract. |
| Engram 逻辑 tensor 总量 | 已测量逻辑总量, 不是驻留估计 | 12 个 tensor 总计 `203,073,076,240` bytes. |
| transfer buffer | 未知 | 没有 transfer lifetime 记录. |
| 剩余 KV/cache 预算 | 未知 | 没有创建 runtime cache. |

不作剩余预算声明. checkpoint 和 Engram 的逻辑总量不能表示每个 rank 的驻留量或可用 KV/cache capacity.

## 门为何仍未解决

当前记录显示安全违规, 但没有识别 opaque 所有者. 记录没有把 opaque 总量分解并核对到 CUDA library, 通信, allocator, 注册或模型准备分类.

跨 rank 的证据也不完整. rank 1 没有 v7 边界记录. 没有记录证明四个 rank 在加载前状态相同, 或加载增量相同.

准备过程在未改变的门处停止. 结果没有产生 weights-complete, prepared, compute 或 close 记录. 现有结果不能证明加载后的安全驻留, 执行行为或释放行为.

仓库规则要求物理所有权保持明确且有界. 执行规则也要求准备和 graph setup 期间保持资源所有权. 没有所有权账本时放宽该门, 会移除当前阻止 TP4 的安全证据.

## 最小安全下一次测量

使用冻结输入和未改变的 1 GiB 门, 运行一次受控 TP4 prepare-only 诊断.

在三个时间点为每个 rank 记录相同的有界字段:

1. 加载权重前.
2. 加载权重后立即记录.
3. 门检查时.

记录 rank, 阶段, allocated bytes, used bytes, free bytes, total bytes, kernel bytes, opaque bytes 和记账来源. 在未改变的门处停止. 不要运行模型计算, CUDA Graph capture, memcheck 或 shutdown 实验.

这一次运行足以回答第一个决策问题: 每个 rank 是否都在加载前超过上限, 以及加载权重是否增加了实质性的 opaque 占用. 如果任何 rank 缺少记录, 或 opaque 总量不能核对到有界所有者, 该门仍未解决.

## 来源

- [GPU 执行规则](../execution.zh.md)
- [运行时 contract](../requirements.zh.md)
- [DSV4.1 模型与 checkpoint contract](dsv41-model-checkpoint-contract.zh.md)
- 当前本地证据记录: `LOCAL.md`
- 物理诊断记录: `diagnostic-v6`, `controlled-prepare-v6-v4`, `diagnostic-v7` 和 `opaque-analysis-v7`
