# 算子性能计费

[Microbench 契约](operators.zh.md)定义固定用例, 校准, 计时和验收.
本文定义 `operator-reference-v3` 资源策略.
[计算实现](../../../development/operators/accounting.py)使用 HTTP 资源速率和资源时间公式.
它使用独立版本, 不改变 HTTP 计费.

## 内存计费

对完整的固定算子执行序列统计不重复的读取.
跨层共享的输入数组只计一次.
FA, GDN 和 output 输入包含 BF16 value 和 FP32 residual.
FFN 的这两个输入字段均使用 BF16.
各层不同的权重和初始状态各计一次.
矩阵 scales 和归一化向量按存储精度计费.

使用 [HTTP 计费](performance.zh.md#下界计算)中的串行 kernel 分组.
各组先取资源时间的最大值.
完整执行序列只使用一份 128 MiB cache 额度:

```text
raw = sum(group_resource_maximum)
compute_floor = sum(group_compute_maximum)
reference = max(compute_floor, raw - 128MiB / HBM_rate)
```

此 cache 预算不能将各组时间减少到计算时间下界以下.
不要对每个 kernel 单独扣除额度.
共享输入只在首次使用处获得内存额度.
Runner 通过 CUDA driver 读取设备 L2 容量.
拒绝 L2 容量更大或 GPU 型号不同的设备.

不给中间值, scratch, padding, 拷贝或重复读取 HBM 额度.
此策略还省略全部输出和持久状态写入.
这些省略可能降低下界.
它们不会在候选实现增加工作时提高计费额度.
数值测试仍然检查必要的状态写入.

四步 draft chain 的所有阶段共享层权重.
整个 chain 对这些权重和原始 cache 历史只计一次.
Chain 内产生的新 KV 值没有 HBM 额度.
Chain 也不给 embedding 读取额度, 因为生成的 token 可能共享 embedding 行.
其他固定 embedding 输入按不同的 token 行计费.

## 算术计费

每次矩阵乘加计两次运算.
全部固定 attention 用例使用 BF16 QK 和 PV 操作数.
每个历史 token 的 BF16 K/V cache 为 4096 字节.
候选实现的分派改变时, 这些假设保持固定.
候选实现不能选择更大的理论工作量.

文本投影, 归一化, FFN 和 GDN 递推按计算实现中的固定维度与公式计费.
GDN 使用 [HTTP 计费](performance.zh.md#算术)中固定的精度与分派规则.
私有 checkpoint 数量不增加内存计费额度.

Output distribution 和 sampling 包含归一化及完整词表投影.
Sampling penalties, 排序, 整数选择和随机数生成没有算术额度.
它们的成本仍在计时范围内.

Acceptance 统计尝试接受时的 FP32 概率比值.
对于概率 target, 还统计 residual 减法, mass 求和和 FP64 阈值乘法.
Greedy 整数 target 只保留比值计费.
因此很小的下界可能产生很大的延迟倍率.
不要添加经验性的 launch 时间下限.

Vision 的 patch, attention, MLP 和 merger 矩阵乘法按 BF16 精度计费.
其标量计费包括归一化, 残差加法, rotary 乘积, attention 缩放和求和.
Input 边界还统计像素归一化.
超越函数, 量化, 类型转换和省略的标量项没有计费额度.

## 输入与测量 identity

性能输入生成使用 NumPy 和已提交的数值 fixture.
它不执行 PyTorch reference.
较大用例使用固定的 shape, page table, slot 和有序层权重绑定.
生成的输入与初始状态数组具有合并的 SHA256 identity.
报告还绑定原型清单, 生成器, NumPy 版本和 checkpoint.

计费 identity 同时绑定算子计费源码及其导入的 HTTP 计算源码.
它包含 kernel 分组定义, 速率和 cache 策略.
批准的门槛还绑定完整资源总量和下界值.
批准的门槛绑定此 identity, 计时策略 identity, 用例, 生成的输入以及兼容的设备设置.
常规验收不加载校准耗时.
计费变化时, 按下式换算倍率, 保持每项已批准的最大延迟:

```text
new_max_ratio = old_max_ratio * old_reference / new_reference
```

在迁移来源信息中保留旧计费 identity, 资源值和倍率.
此换算不得提高已批准的最大延迟.

在每个独立样本前, 从准备好的设备副本恢复状态与 RNG.
在开始事件前完成每次恢复.
当 RNG 改变时, sampling, 单步 draft 和 acceptance 使用一次恢复后的调用.
可安全重复的操作使用足够的调用次数, 使每个样本至少持续 20 ms.

对于可安全重复的操作, 不纳入统计的预试以 25 ms 为目标, 选择重复次数.
整个样本集合保持该次数不变.
如果任何可安全重复的区间短于 20 ms, 则丢弃整个集合.
然后增加重复次数.

若 30 个样本的结果不确定, 则以相同重复次数扩展到 120 个样本.
扩展沿用相同的计时事件和测量 Graph.
固定次数的预试及扩展样本之前, 执行四次不纳入统计的预热调用.
若扩展中可安全重复的区间过短, 则重新采集完整的 120 个样本.
报告记录这些重试.

计时策略为 `operator-timing-v2`.

Eager 计时在调用者 stream 上的生产调用前后记录事件.
Graph 计时将同一个生产函数捕获为独立子图.
测量 Graph 在两个计时事件节点之间包含串行的子图副本.
Runner 在测量前上传此 Graph, 并保留其资源所有者直至销毁.

捕获必须保持全部输入和状态值不变.
独立的主机计时在恢复后使用原始生产 Graph replay.
测量 Graph 的构建时间单独报告.
策略 identity 绑定计时代码与 Graph 构建代码.

报告分别给出准备, recording, 主机提交和 GPU 时间.
临时存储大小是绑定, warmup 和 Graph 创建前后的设备内存差值.
恢复副本的存储量单独报告.
记录时钟, 功率, 温度, driver 和 GPU UUID.
若所选 GPU 存在其他计算进程, 或测量设置发生变化, 则拒绝测量.

校准命令启动五个不同的测量进程.
每个进程测量全部固定主用例.
重复 session, 缺失用例或输入与计费 identity 不同都会阻止门槛提议.
全部报告保存在仓库外.
