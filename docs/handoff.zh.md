# 当前状态与待办

更新时间: 2026-10-11.

## 实现

Rust 管理服务, 调度和逻辑 KV. Python 管理一块 B200 GPU.
使用块大小 784, 16 层 FA, 48 层 GDN 和 `mamba_cache_mode="align"`.
输入与输出的总长度上限为 262144 token.

普通解码, MTP4 和 DSpark 在启动时选择. Worker 保持所选模式.
Chat, Responses, streaming, tools, thinking 设置, 约束生成, 取消和前缀复用继续可用.

DSpark 使用固定权重和 target features `(5,19,33,47,61)`.
Markov proposal 与累计 confidence 选择零至七个候选. 初始阈值为 `0.2`.
随机接受使用 `min(1,p/q)`, 拒绝后使用归一化的 `max(p-q,0)`, 全部接受后使用 target bonus.
DSpark context boundary 测试及其验收要求已删除.
普通解码和 MTP4 保留 6 个总长度 262144 边界用例.

CUDA Graph, 多 CUDA stream 和 PDL 控制模型执行.
Native 构建使用 10 个算子头文件. GEMM 构建独立识别其根头文件.

全部 CUDA 后端代码文件满足 800 行上限. Pre-commit 钩子使用 clang-format 23.1.3.
头文件内容进入缓存 key, 加载 provenance, 正式合同和 profiler 检查.
构建与加载检查拒绝输入变化.

Prefix cache 在校验后的状态提交之后发布. 请求准入和已完成 stream 共用有界资源 lease.
HTTP supervisor 使旧逻辑 KV 失效, 重建失败 worker, 并且不重放请求.
Graph 压力驱逐释放未使用的 pool owner. 编译器和诊断 registry 有固定上限.

项目持有的 FX 源码和 FakeTensor metadata 设有数量与字节上限.
编译器退休在安全的主机边界执行 Python 垃圾回收. Prefill 图准入按计划存储需求计算, 各路线使用各自预算.
[服务合同](serving.zh.md) 规定资源预算, 恢复限制和请求计时.

[架构](architecture.zh.md) 定义所有权和缓存合同.
[算子开发](kernels.zh.md) 记录模块划分及已选配置.
[开发](development.zh.md) 记录环境和模型配置.

## 验证

15 个阶段性能行通过, 每项负载完整预热 2 次, 测量 5 次.
[README 表格](../README.zh.md#性能实测) 给出 prefill 时延, 下界倍率, decode TPS 及理论百分比.
Prefix-hit prefill 不设倍率门槛. 波动与 decode 检查通过.

317 个正式算子用例在一次完整采集中通过数值及速度检查.
另一 agent 重算全部轮次中位数及置信下界.
普通 / MTP4 的 6 个容量用例完整输出, 无 OOM, 重算抢占为零.
摘要从完整原始日志派生. 较大的失败采集保持原始状态.
MTP4 和 DSpark 通过 Chat 及 Responses 完成真实 oh-my-pi 工具循环, 正常关闭审计通过.

这些采集使用推理源码 `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb`.
这些采集早于当前服务资源修改. 新采集需验证修改后的推理源码.
[验收](acceptance.zh.md) 记录原始哈希, 失败尝试, 审查范围和限制.

之前的 CPU 测试通过 760 项和 94 个 subtest, 未纳入 442 项 GPU 测试.
GPU 测试通过 436 项, 该次采集未纳入最大上下文用例.
Rust workspace 测试, Clippy, 格式化, 行宽, Ruff 和文档检查通过.
教程构建, 10 项片段测试和 9 项浏览器测试通过.
该次采集的全部自有 GPU worker 退出. 临时服务端口释放.

另一 agent 检查 CUDA 拆分, 头文件身份, 加载竞争和格式化工具.
修正后的 Opus 5.5 审计未新增生产选择.
Packed-scale 和 pipeline 候选保留为外部诊断.
[性能分析](profiling.zh.md) 记录已选优化及实测结果.

### 服务资源验证

修改后源码的 CPU 采集通过 922 项测试和 99 个 subtest, 未纳入 445 项 GPU 测试.
Rust workspace 通过 143 项测试. 格式化, Clippy, 行宽, Ruff, CUDA 格式化和文档检查通过.
更新后的教程通过 10 项片段测试和 9 项浏览器测试.

GPU 采集通过 437 项测试, 两项使用旧测试合同的测试失败.
修正后的 graph pool 和编译 metadata 测试随后一起通过.
本次采集未纳入最大 context 案例. CUDA 算子实现没有变化.

使用目标模型的 MTP4 诊断通过两种 API 的真实取消和慢客户端完成检查.
最终请求状态和资源 lease 均为空. 逻辑 pool 仅有 null block 0 保持分配.
全部自有服务进程退出. 这项短时诊断不提供长时间运行验收.

MTP4 的 `after_commit` 诊断在流已经输出内容后触发 CUDA allocator OOM.
HTTP 保持存活, 流报告失败, 替换后的 worker 完成 Chat 和 Responses 请求.
共享前缀在恢复后首次请求为冷缓存, 下次请求命中. 最终请求状态为空, 全部自有进程退出.
本次诊断覆盖一个故障阶段. 完整的两模式故障, 阶段时延和持续测试验收尚待执行.

当前 CPU 测试套件通过 1133 项测试和 99 个 subtest, 未纳入 448 项 GPU 测试.
Rust workspace 通过 143 项测试. 全部 pre-commit 检查通过.

另一 agent 检查协议结束记录, 客户端消息, 服务端 metric 和断连后的 2 秒等待上限.
更新后的教程通过 10 项片段测试和 9 项浏览器测试.
源码 `a0109c9cdebbd142a5ae79109b54766dc3ee2464` 完成 MTP4 持续测试, 在 3670.64 秒内完成 5733 个正常请求.
MTP4 的可用性, 资源和内存检查全部通过.

DSpark 持续测试在 3643.13 秒内完成 964 个正常请求, 没有请求失败, OOM 或 worker 重启.
其 GPU 和 HTTP 内存检查通过. worker RSS 检查因后期中位数增长 12.32% 而失败.

同一源码通过八项 allocator-OOM 恢复用例, 每种推测模式覆盖四个阶段.
这些结果适用于下一轮编译器 metadata 和图准入修正之前的代码.
它们不提供这些修正的验收.
这些采集的全部任务自有 GPU worker 和临时服务端口均已关闭.
完整原始记录保留失败及停止的尝试.

修改后的 IR 通过编译器退休后的 CUDA Graph 重放和冷形状重新捕获.
两项图内存测试通过 allocator 回收和侧流 owner 释放检查.

独立的 DSpark batch-four 诊断在三轮执行中完成六次捕获, 十二次命中, 没有压力驱逐.
各轮保存的 FA/GDN 字节和生成 token 一致.
它在 `a0109c9cdebbd142a5ae79109b54766dc3ee2464` 之上使用已审查的准入文件, IR 仍为上一版本.
该诊断不提供完整修改后推理源码的验收.

## 待办

- 用当前源码的资源, allocator-OOM 和长时服务采集验证 `REQ-SERVE-006`, `REQ-SERVE-007` 和 `REQ-SERVE-008`.
- 验证编译器 metadata 和图准入修正后的 RSS 稳定性.
- 在稳定的 prefill 图重放下测量 DSpark batch-four 阶段时延.
- 使用 4 个 oh-my-pi agent, 对 MTP4 和 DSpark 分别测量至少 3600 秒和 500 个正常请求.
- 发布每种模式的请求散点图, 然后在修改后的推理源码上重测 15 个阶段性能行.
- 新采集通过前, 之前的测量仅适用于其记录的源码范围.
- 按 [审计](audit.zh.md) 的 `PY-04`, 在显存压力下测试图淘汰和重新捕获.
- 验证形状持续变化时编译器存储不超过指定容量上限, 测量 `PY-06` 主机重叠.
- 完成用户对教程的审阅.

## 教程预览

按用户要求, 教程预览继续使用端口 18084.
被忽略的 `LOCAL.md` 记录地址, 进程身份和维护命令.
英文 Markdown 及完整中文翻译记录当前代码合同. 教程包含 13 个完整中文章节.
