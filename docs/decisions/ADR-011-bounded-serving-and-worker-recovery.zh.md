# ADR-011: 服务资源上限与 worker 恢复

状态: 有效.

## 问题

长时 agent 任务会复用前缀, 建立不同形状的图并延续历史, 客户端可能中断或缓慢读取.
仅有数量上限无法限制历史副本及已完成 stream 缓冲的字节数.
GPU 故障可能发生在部分持久状态修改之后, 完整回复返回之前.

## 决策

准备请求或复制历史前先保留准入容量.
分别设置请求数量, 保留请求字节, response storage 及 stream buffer 上限.
请求输出和 stream drain 释放引用前, 持有准入 reservation.
准入, 排队和生成共用从准入时开始的绝对 deadline.

分配时暂存新的 prefix identity.
只有经过验证的结果提交对应状态后才发布 identity.
LRU 驱逐保护仍被引用的 block, 移除失败后未提交的 identity.
驱逐复用固定 GPU arena, 不将 arena 的 HBM 存储返还 allocator.

形状持续变化时, graph 和编译器存储不超过各自容量上限.
保留 fullgraph 编译, 手动 CUDA Graph, stream 依赖及 PDL 安全合同.
Shutdown 释放 graph pool, 编译 callable, 模型引用及请求状态.

Worker / device 故障后状态无法安全复用时, 让受影响请求失败, 不自动重放.
HTTP listener 继续存活, 关闭旧 worker, 清空逻辑缓存并加载新 worker.
恢复期间返回暂时不可用. 恢复尝试和 shutdown 等待均遵守既定上限.
使用新 worker 状态后再准入请求. 替换 worker 启动前使旧逻辑 KV identity 失效.

## 替代方案

部分状态修改后继续使用旧 worker, 需要证明每次状态修改均已恢复.
无法提供此证明时, 恢复路径使用新 worker.
自动重放请求可能重复 stream 输出或工具调用, 因而重试由客户端控制.
定期轮换 worker 可能掩盖资源泄漏. 正常负载测试要求意外重启次数为零.

## 验证

CPU 测试覆盖缓存发布, 未知回复 ID, 有界准入, deadline 及 stream drain.
真实模型测试覆盖 OOM, 部分状态修改, 共享前缀及恢复后再次服务.
在显存压力和形状变化下测试图驱逐及重新捕获.

分别使用 4 个 oh-my-pi agent 对 MTP4 和 DSpark 运行 soak test.
每种模式至少持续 1 小时. 请求数量和 agent 任务完成情况用于诊断, 不设最低数量门槛.
测量请求计时和内存, 不设 soak 性能门槛.
推理路径修改后重测 15 个阶段性能行.
日常测试默认跳过重型 soak test.

参见 [需求](../requirements/service.zh.md#req-serve-006-资源上限与缓存正确性) 和 [测试](../testing.zh.md).
