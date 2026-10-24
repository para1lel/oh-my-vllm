# 算子 microbench 设计

本文给出算子开发的数值和性能验收规则.
[用例清单](../../../development/operators/cases.json)固定测量形状与模式.
[限值](../../../development/operators/limits.json)将数值和性能验收绑定到这些用例.

## 范围和边界

主要边界采用完整的生产操作:

每个边界最多有 5 个固定数值用例和 5 个固定性能用例.
每个用例固定 shape, 权重, 输入, 状态, 执行模式及采样设置.
选择推理中会出现的 shape.
不要将这些用例扩展为 flags, seeds 或层权重的笛卡尔积.
仅由 MTP 使用的 FA 变体通过 MTP 边界检查.
GDN recurrence 和 prefill 通过完整 GDN 用例检查.

不要为测量而拆开生产融合或额外导出中间张量.
完整操作失败后, 使用 profiling 定位内部成本.
初始套件不为每个 CUDA kernel 增加 benchmark 入口.
保留全模型和 HTTP 检查, 验证组合, 服务行为和完整推理成本.

## 数值参考与 fixture

数值参考使用直接的 PyTorch 公式实现.
固定输入和原始 checkpoint 权重精确转换后, 使用 FP64 算术.
参考实现按照[模型契约](model.zh.md)实现数学操作, mask, position, 状态转移和采样规则.
它不得调用生产 kernel, 也不得复制其分块和融合算法.
它不得使用旧 kernel 输出作为预期值.
Attention 分块仅用于控制参考实现的内存用量.

Fixture 的输入, 初始状态, 元数据和完整 PyTorch 预期输出保存在仓库内.
使用 `development/operators/fixtures/<case_id>/manifest.json`, 并在该目录存放无损压缩的 NumPy 分片.
每个分片不超过 32 MiB.
不要将 checkpoint 权重复制到 fixture.
通过张量名和现有 checkpoint 身份绑定外部 checkpoint 权重.

Manifest 记录张量 shape/dtype, 用例身份, 输入摘要, 预期输出摘要, reference 源码摘要, PyTorch 版本和精度.
使用固定公开输入, 或具有代表性幅度的确定性合成输入.
不要捕获用户的私有请求.
使用存储的数组作为 fixture 输入.
仅有 seed 不能替代已存储的参考输出.

常规测试加载 fixture, 只执行待测实现.
它们不导入 PyTorch, 也不生成参考输出.
Fixture 缺失或改变时返回错误.
仅在显式首次生成或经批准的 fixture 更新时使用 PyTorch.
将变更的 fixture, manifest 和参考实现变更一起暂存并审查.
测试期间不得重新生成或覆盖预期数组.

比较具名输出和指定状态写入的全部元素.
其中包括 BF16 KV, GDN recurrent/convolution checkpoint, 以及 hidden 导出.
按 sequence 和字段分别检查归一化 L2 误差及最大绝对误差.
参考值接近零时, 也使用绝对误差条件.
拒绝契约不允许的非有限值.

对整数索引, mask, slot 选择, shape 和不应修改的 guard 存储进行精确检查.
参考输出只包含指定的状态写入, 不附带第二份完整只读 cache.

数值门槛保存在版本化的 `development/operators/limits.json` 中.
Kernel 变更前, 根据直接 PyTorch 输出校准门槛.
按各字段的数据类型和数学行为应用经审查的[精度策略](precision.zh.md).
审查参考公式时检查可精确求解的简单用例.

不要将全模型门槛复制到所有算子字段.
不要仅通过当前 kernel 误差乘以一个系数来选择门槛.
缺失或未经审查的门槛阻止验收.
待测实现不得通过放宽门槛或更新 fixture 来使自身通过.

FFN 和 GDN Graph 用例在数值检查期间先使用零输入, 然后使用 fixture 输入.
FFN 输入为零时, 输出必须为零.
使用零输入重放后, fixture 输入的输出必须满足相同的数值门槛.
此检查检测前次重放的值. 它使用相同的 shape 和 reference fixture.

采样 fixture 包含完整分布和不使用生产代码计算的 seed/counter 预期值.
精确 token 检查使用远离数值歧义 CDF 边界的固定随机阈值.
通过固定用例检查 greedy 并列顺序, mask, rejection residual, bonus 选择和 RNG counter.

PyTorch 生成预期数值数组. 常规测试使用 NumPy 比较数组.
整数 RNG 映射使用不调用生产代码的确定性辅助实现.
仅计算 distribution 时不得选择 token 或推进 RNG 状态.
Worker 行为测试检查 stop 截断发生在有效状态选择之前.

## 理论延迟与性能门槛

使用与 [HTTP 计费](performance.zh.md#标称资源)相同的标称 B200 资源速率和资源时间公式.
算子计费和 HTTP 计费使用不同的版本标识符.
每个固定性能用例使用以下公式:

```text
t_lower = max(
    necessary_HBM_bytes / HBM_rate,
    FP8_work / FP8_rate + BF16_work / BF16_rate + TF32_work / TF32_rate,
    FP32_work / FP32_rate,
    FP64_work / FP64_rate,
)
ratio = measured_GPU_seconds / t_lower
pass: ratio <= case.max_ratio
```

分母来自用例的语义工作量和固定精度假设.
它不来自待测实现的 kernel 数量, 临时缓冲区, padding, 重复工作或实测延迟.
每次逻辑执行中, 共享数据只计一次.
与 HTTP 计费一样, 明确列出未计入的成本.
不要将实测 launch 常数加入理论下界.
分母为零属于计费错误, 不能通过.

使用明确声明的生产工作集和缓存策略.
不要将常驻 cache 的输入和权重计为 HBM 必读数据.
多层执行计入 HBM 前, 先从唯一读取字节数中扣除保守的 cache 容量.
整个读取工作集能够常驻 cache 时, HBM 计费为零.
在计费身份中记录容量来源和假设.
结果是保守的标称资源估计, 并非物理最小延迟的证明.

FA/GDN/FFN 和内部 vision block 测量固定的适用层权重序列, 其中 shape 相同.
MTP 和 output 使用其生产权重及声明的缓存策略.
校准和后续测试使用同一序列.
这样可避免用单个热缓存层代表流式访问权重的模型工作负载.

每个最大倍率根据当前正确实现校准.
执行 5 次新的测量会话, 每次会话中每个用例至少采集 30 个样本.
使用固定 seed 的 bootstrap, 重采样 10000 次, 计算每次会话中位倍率的 95% 置信上限.
原始建议值为各会话置信上限的最大值, 向上取整到小数点后 4 位.

经批准的 `max_ratio` 为该建议值乘以 `1.02`, 再向上取整到小数点后 4 位.
将原始建议值和固定的 2% 余量记录在门槛来源信息中.
Kernel 变更后, 该余量不再改变.

将这些门槛与用例身份, 计费版本, 执行模式, 设备规格和校准源码身份一起审查.
冻结经批准的门槛.
校准报告保存在仓库外, 其摘要记录在门槛来源信息中.
常规性能测试将自身倍率与该固定门槛比较.
它们不加载旧计时, 不执行旧实现, 也不做 baseline 与 candidate 的版本比较.
待测实现不得重新校准自身门槛.

这些门槛控制已接受的延迟预算.
略微变慢但仍低于冻结门槛的实现可能通过.
该套件不得声称能检测所有性能下降.

## 计时与判定

通过 GPU wrapper 使用一张空闲 B200.
使用调用者 stream 和生产 Eager/Graph 策略.
Vision 使用 Eager. Decode 使用适用的录制路径.
在该 stream 上使用 CUDA events 测量 GPU 时间.

准备, 录制, host 提交和临时存储大小分别报告.
不要将 host 时间与 GPU 资源下界比较.

GPU 执行计时不包含参考计算, 下载, 分配和 fixture 恢复.
每个独立样本前, 在固定地址恢复状态和 RNG.
Warmup 使用隔离状态.
录制不得消耗 serving RNG 状态.
仅当后续输入不变, 或 fixture 定义了有效序列时, 才能在同一个计时区间中重复调用.
否则每次恢复后只测量一次调用.

不要从混合计时中减去 reset 时间.

Warmup 后至少采集 30 个样本.
对可以安全重复的调用, 每个样本以至少 20 ms 为目标.
记录 GPU UUID, driver, 时钟, 功耗/温度状态, 源码, native 二进制, checkpoint 和用例身份.

拒绝并发 GPU 工作或不兼容的测量设置.

计算中位倍率及其 95% 置信区间.
仅当区间上端不超过 `max_ratio` 时通过.
区间下端大于 `max_ratio` 时失败.
区间跨越门槛时, 补足 120 个样本, 用全部样本重新计算.
仍无法判定时, 报告未决结果并以非零状态退出.

缺失主要用例同样阻止验收.
跨用例平均值不能掩盖单个用例失败.

计时样本和测试报告保存在仓库外.
固定数值 fixture, 参考源码, 用例定义, 计费规则和验收门槛纳入版本控制.

## Serving 边界

FA, GDN, FFN 和 vision 接口提供完整子层.
[阶段清单](../../../python/oh_my_vllm/operators/specification.py)给出内部张量契约和计算.
这些阶段遵循[模型契约](model.zh.md)和 [GPU 执行规则](../../execution.zh.md).
原生算子提供下表中的各个完整边界.
计时包含该边界的全部内部阶段.

| Serving 算子 | 完整计时操作 |
|---|---|
| `gdn` | GDN 输入, recurrent 或 prefill 计算及输出. |
| `fa` | FA 输入, attention, gate 及输出投影. |
| `ffn` | Gate/up 投影, activation 及 down 投影. |
| `output.hidden` | 最终归一化及 hidden 状态导出, 不进行词表投影. |
| `output.distribution` | 最终归一化, 词表投影, 惩罚, 约束和所需概率分布. |
| `output.sample` | 分布计算和 token 选择. |
| `mtp.context` | MTP 输入投影及 KV-only 变体的 FA 输入. |
| `mtp.draft` | 使用共享组件完成适用的 MTP 输入投影, 完整的一层 FA, FFN 及采样. |
| `mtp.accept` | 接受判断, 拒绝采样和候选状态 slot 选择. |
| `vision` | 一个视觉层. 首层包含 patch 输入. 末层包含空间合并. |

能降低完整操作耗时时, 将 producer 计算融合到 GEMM 输入加载或输出存储中.
小型 normalization, activation, quantization 和位置操作可以在其 producer 或 consumer kernel 中执行.
必要时分别使用小行数与 prefill 实现.
[GDN 入口](../../../python/oh_my_vllm/operators/gdn/cuda/gdn.cu)选择 recurrent 与 prefill 实现.
数值检查使用独立的 PyTorch FP64 全模型参考.

各输出接口共享必要的最终归一化计算.
Hidden 导出支持其完整输入行数, 独立于词表投影所选择的较少行数.
分布接口可以在不选择 token 的情况下执行.
这些语义接口可以使用融合的物理执行.

MTP 包包含其组合融合实现, 并控制其存储布局.
组件接口规定兼容的输入, 输出, 存储使用方式和生命周期.
它不根据其他算子的私有 buffer 下标推断兼容性, 也不复制该算子的完整 scratch 布局.

MTP acceptance 返回候选结果, 不返回已发布状态.
Worker 先对返回的 token 应用请求停止条件.
随后由停止条件处理后的接受数量通过现有状态提交过程控制状态选择.
处理停止条件前, GPU 候选 slot 不能替代这一决策.
保持 target, proposal 和 acceptance 各域的 RNG 状态与计数器规则.

## 依赖启动

小规模 FFN 和 GDN 操作在归一化与输入投影之间使用 [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
它们也在输出量化与输出投影之间使用 PDL.
65 到 1024 行的 FFN prefill 在这两个边界使用 PDL.
其融合输入投影包括激活和输出量化.
更大的 prefill 操作保留串行启动.
