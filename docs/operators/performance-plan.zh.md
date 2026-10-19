# FA 和 GDN 性能计划

## 必需行为

每项测量使用一张 B200. 保持运行时及数值契约.
使用该 GPU UUID 的实测峰值. 保持负载权重和 roofline 公式不变.
必要工作量使用所选实现的操作数类型. 排除临时存储, padding 和重复工作.

24 种 HTTP 配置的每一种效率都必须至少为 50%.
每个关注的 FA 和 GDN shape 都必须至少与最快的适用公开基线一样快.

初始 shape 集合为完整的建模 FA 和 GDN 负载.
加入 HTTP 执行 trace 中额外的 attention shape.
分别比较 eager 执行和 CUDA Graph replay.
使用固定版本和源码身份的 FlashInfer 及 QwenLM/FlashQLA.
公开库仅用于隔离的开发环境.
推理 worker 不能导入它们.

## 比较规则

比较相同的操作边界, 输入, 类型, causal mask, scale, state 和必要输出.
公开 attention core 与完整 serving 边界是不同的测量.
状态布局转换, checkpoint 导出和其他必要适配必须明确计时范围.

计时前执行数值检查. 各次测量之间恢复可变状态.
每项比较使用相同的 cache 策略, 预热, 重复次数和计时方法.
不支持的 case 或无效的数值比较不能记为通过.

在两种语言的 README 中提供比较汇总和 HTTP 表.
列出覆盖数量, 严格性能通过数量, 中位及最差比值, 以及最差 shape.
将完整的逐 case 报告保存在仓库外. 明确写出每项未达到的性能要求.

## 实现任务

1. 建立可复现的公开基线及聚焦的 Nsight trace.
   在仓库外记录源码, 二进制, 依赖, 设备, shape 和测量身份.

2. 使用块计算及明确的状态依赖优化 GDN prefill.
   保留 FP32 live state, BF16 checkpoint 导出, 私有写入及保护存储.
   检查同时跨越一个完整计算块和 784-token checkpoint 的路径.

3. 优化 FA prefill 及共享 FP8 投影流水.
   保持规定的操作数类型, 缩放, 输出舍入及持久 cache 行为.
   GPU 实验前使用 CPU tile 模型.

4. 集成后测量完整服务.
   通过 GPU 时间线选择进一步的 host, workspace 或 launch 修改.
   执行全部 24 种配置, 使用精确输入及输出长度, 且不复用前缀缓存.

5. 执行适用的仓库检查和完整算子审计.
   获取独立代码及文档审查. 修正错误.

   更新两种语言的 README 和交接. 在 `main` 上启用 hooks 后提交及推送.
   停止临时服务, 并在审查后删除临时报告.

## 审查重点

检查零行, 短尾, 非零初始状态和可变序列长度.
检查 784-token 边界及 BF16 checkpoint 导出后的 FP32 carry.
代数变换改变 FP32 结合顺序时, 检查数值行为.
检查基线适用性及完整 shape 覆盖.
检查最终测量期间的来源稳定性.
