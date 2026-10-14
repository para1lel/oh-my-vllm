# 验收证据索引

每个结果只适用于记录的源码, 环境, 配置和工作负载.
门槛见[需求](requirements.zh.md), 采集命令见[测试](testing.zh.md).
正式验证使用原始记录. 可移植摘要用于阅读和离线分析.

## 当前重构

已确认方案用仓库 CUDA 与原生库替换外部 Python 算子 provider.
原比较门槛也替换为独立数值契约, 以及框架阶段时延最多为下界 2 倍.
Prefix-hit prefill 保持 ratio 例外. 全部阶段 spread 门槛仍有效.

最终源码的性能矩阵及交付检查尚待完成.
[交接](handoff.zh.md)记录已完成里程碑和剩余检查.
不得依据历史测量或不完整矩阵声明最终验收.

## 已选 subgroup scan 检查点

[绑定源码的摘要](evidence/refactor-s8-scan/verification.json)标识已改动的 scan 和原始验证记录.
完整 CPU 套件通过 1592 个案例和 153 个 subtest. 完整非边界 GPU 套件通过 593 个案例.
选定 GPU 验证通过 99 个案例. 四个对齐及抵消案例通过 Compute Sanitizer, 错误为零.

全部 539 个算子数值案例通过. 未提交源码的报告保持 `diagnostic=true`, `passed=false` 和 `selected_passed=true`.
FP32 归约顺序改变. 数值容差和理论模型保持相同.
最终源码的容量, 服务, 一小时及框架性能验证尚待完成.

`f05a35d` 的[干净源码重跑](evidence/refactor-f05a35d/verification.json)全部 539 个算子数值案例通过.
ordinary 单请求 32768 输入/4096 输出尝试完成两次预热与五次测量, OOM 和重计算抢占均为零.
prefill 和 decode 比例为 4.436 和 2.320. 波动检查通过, 但两倍延迟检查失败.
失败记录保持不变. 这一组案例不能构成完整十五组框架矩阵.

## 优化观测

[可移植诊断摘要](evidence/optimization-observations/verification.json) 绑定当前 scan/GEMM 计数器, 隔离 cuBLAS 探针和三层 wavefront 实验.
它标识源码快照, 原始哈希, 样本, 失败尝试和验证限制.
这些采集只用于诊断. 它们没有选出生产修改, 也不能证明正式验收通过.
参见 [测量方法](profiling/measurement.zh.md) 和 [算法调研](kernels/optimization.zh.md).

## 源码 3324a7f

[可移植验证摘要](evidence/refactor-3324a7f/verification.json)记录源码哈希和原始报告身份.
它是供阅读的派生文件. 不变的原始记录及独立审查提供正式验证.

数值矩阵全部 539 个案例通过. 完整非边界 GPU 套件全部 589 个案例通过.
ordinary/MTP4 在 batch 1, 2, 4 的六组容量案例通过, 输入 258048 个 token, 输出 4096 个 token.
MTP 和 DSpark 各通过六类服务客户端和四阶段 allocator OOM 恢复.
`f5b13d7` 只修复 fixture, 完整 CPU 套件通过 1592 个案例和 153 个 subtest.

每种模式均有四个持久 OMP goal 会话运行一小时, OOM 和重启均为零.
独立审查验证采样资源边界, 最终释放, 正常 SSE/SDK 请求及 sandbox 删除.
MTP 完成 389 个正常流, DSpark 完成 76 个. 每种模式各有四次 controller 截止中断.

原始 MTP 采集器因生成/中断截止竞态而未通过 SSE 审计. 原失败结果保持不变.
原始 DSpark 采集器报告运行完成, `passed=false`, 尚待独立资源审查.
单独的独立审查绑定各组不变的原始摘要及 raw manifest.

每个外部 driver 均在服务退出后报告无法证明所收养进程身份, 且未保存这些身份.
已记录的服务, worker 和 sandbox 身份均已退出. 临时推理监听端口和 GPU 已释放.
历史未知收养进程身份无法重建.
采集中 worker RSS 增长. 证据不能证明内存平台或完整分配归因.

ordinary 单请求 32768 输入/4096 输出阶段尝试未通过下界两倍延迟门槛.
prefill 和 decode 比例为 5.419 和 2.329, 其波动检查通过.
理论公式未变. 完整十五组矩阵与最终交付尚待完成.

## 历史范围

重构前修复交付记录在 `b98a83a`.
其绑定源码的测量, 失败, 限制和已 review 的 collector 范围保留在 Git 历史与原始外部记录中.
这些记录不能证明重构源码或新 2 倍门槛通过验收.
过时比较摘要已从当前工作树删除.

## 采集身份

记录源码与可执行文件哈希, 依赖 pin, 已加载模块与头文件哈希, 编译器 flags 和 checkpoint 身份.
进程出生身份, worker epoch, GPU UUID 和采集时间记录在原始外部记录中.
可移植摘要不得包含主机路径和 secret, 并须标识原始记录哈希.

保留完整失败集合和中断尝试及原因. 不挑选有利样本.
编译活动, capture, cache 淘汰, 干扰或源码变化会使测量重复失效.
profiling 与正式计时分开.

## 服务小时范围

检查框架持续可用, 采样资源限制, 服务契约和自有 sandbox 清理.
OMP goal 完成, 任务进展, 工具错误和最终回答属于工作负载观察.
原始客户端失败保持原状态. 配置 deadline 和取消单独说明.

一小时结果不能证明内存平台, 完整分配归因或测量区间之后的运行.
报告 snapshot 延迟和请求排空尾部. 内存观察放在证据中, README 不放内存图.
