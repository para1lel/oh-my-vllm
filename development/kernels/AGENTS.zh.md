# 算子 harness

读取[内核契约](../../docs/kernels.zh.md), [需求](../../docs/requirements.zh.md)和[性能诊断](../../docs/profiling.zh.md).
从完整工作负载矩阵的合法调用静态推导正式案例. 相同配置去重.
保持独立数学参考, shape/dtype 契约, mutation witness 和数值容差.

FP64 不能替代规定的 FP32 量化或可观察的 BF16 cast.

## 测量

比较完整算子, 包含必需的复制和 reduction. 使用交错的 candidate/baseline 样本对.
profiling 与计时验收分开. 报告不可用计数器和失败尝试.
融合和 padding 不得抬高理论下界.

框架阶段时延是最终性能门槛; 算子时延用于诊断和选择候选.
保留源码/头文件/编译器身份, 正式案例, harness 代码和可移植摘要.
原始 trace 和临时调优放在外部存储. 每次运行后释放自有 GPU worker.
