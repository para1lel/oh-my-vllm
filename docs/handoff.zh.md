# 当前状态与待办

更新时间: 2026-10-10.

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

全部后端代码文件满足 800 行上限. Pre-commit 钩子使用 clang-format 23.1.3.
头文件内容进入缓存 key, 加载 provenance, 正式合同和 profiler 检查.
构建与加载检查拒绝输入变化.

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
最后修改更新边界 collector, 测试, 文档和教程, 推理实现保持不变.
[验收](acceptance.zh.md) 记录原始哈希, 失败尝试, 审查范围和限制.

当前 CPU 测试通过 760 项和 94 个 subtest, 未纳入 442 项 GPU 测试.
GPU 测试通过 436 项, 该次采集未纳入最大上下文用例.
Rust workspace 测试, Clippy, 格式化, 行宽, Ruff 和文档检查通过.
教程构建, 10 项片段测试和 9 项浏览器测试通过.
全部自有 GPU worker 退出. 临时服务端口释放.

另一 agent 检查 CUDA 拆分, 头文件身份, 加载竞争和格式化工具.
修正后的 Opus 5.5 审计未新增生产选择.
Packed-scale 和 pipeline 候选保留为外部诊断.
[性能分析](profiling.zh.md) 记录已选优化及实测结果.

## 待办

- 按 [审计](audit.zh.md) 的 `PY-04`, 在显存压力下测试图淘汰和重新捕获.
- 验证形状持续变化时编译器存储有界, 测量 `PY-06` 主机重叠.
- 完成用户对教程的审阅.

## 教程预览

按用户要求, 教程预览继续使用端口 18084.
被忽略的 `LOCAL.md` 记录地址, 进程身份和维护命令.
英文 Markdown 及完整中文翻译记录当前代码合同. 教程包含 13 个完整中文章节.
