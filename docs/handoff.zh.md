# 当前状态与开放工作

更新日期: 2026-10-09.

## 当前实现

Rust 管理服务, 调度与逻辑 KV. Python 在单张 B200 上执行 GPU 计算.
独立运行时默认使用 CUDA, 保留固定 TileLang 作为比较参考.
Block size 为 784. 目标模型包含 16 个 FA 层及 48 个 GDN 层, 使用 `mamba_cache_mode="align"`.
输入与输出总长度上限为 262144 token.

在 worker 启动时选择普通解码, 原生 MTP4 或 DSpark.
普通解码保存 FP32 GDN state. 两种推测模式保存 BF16 GDN state 及回滚快照.
Chat, Responses, 流式输出, 工具, thinking 设置, 约束生成, 取消与前缀复用继续可用.

DSpark 固定 checkpoint 权重, 共享目标 embedding 及词表 head.
五层草稿使用零基层号 `(5,19,33,47,61)` 的目标特征, context 注入, Markov 及 confidence head.
只有保留的目标行注入持久草稿 context. 被拒绝的目标行回滚已计算进度及 GDN 快照.

DSpark 提出零至七个候选. 累积 confidence 初始阈值为 `0.2`.
小样本文本及工具历史采集使用不同于正式采集的输入, 提供阈值选择证据. 阈值设为 `0.0` 可调试固定候选数.
所有阈值下的候选数都可能受输出, context 及 grammar 限制而减少.
Greedy 验证保留连续匹配的草稿前缀及下一个目标 token.

随机验证使用完整条件分布, 按 `min(1,p/q)` 接受.
拒绝后从归一化的 `max(p-q,0)` 重采样. 全部接受后从目标分布抽取 bonus token.
Grammar mask 及采样设置应用于每个推测前缀.

Semantic IR 编译 prefill, 目标验证, 原生 MTP 及 DSpark 单元.
Manual CUDA Graphs 使用有界且不同的缓存族及 pool, 包含捕获恢复与显存保护.
DSpark CUDA 内核保留规定的 BF16 舍入及长位置 YaRN 精度.

环境选择使用显式 prefix, 已激活环境或仓库 `.venv`.
设置 `OH_MY_VLLM_MODEL` 或 `--model`, DSpark 另用 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model`.
命令见[开发指南](development.zh.md), 所有权及缓存契约见[架构](architecture.zh.md).

## 性能实现进行中

采集器采用十五项固定负载, 记录各请求的提交, 首 token 和末 token 边界.
Prefill 与 decode 独立检查墙钟时延和理论下界.
模型计入参数流量, 共享执行资源和语义依赖.

执行 trace 记录有效 query, KV 长度, 状态槽和草稿工作.
GPU event 区间仅用于诊断.
规范模型已完成独立数学审查, 并检查调度一致性, 缺失证据时拒绝验收.
完整十五项采集待完成.

执行采用 CUDA Graph, 多个 CUDA stream 和 PDL.
自有 group-scaled FP8 GEMM 保持 checkpoint FP32 scales, 控制 stream 与 launch.
GDN 准备汇合独立 gate 分支. 有界 prefill graph 恢复 FA 与 GDN 写入.
Allocator topology reuse 将诊断中的 prefill 捕获存储从 29.13 降至 7.82 GB.

大型 GEMM, 卷积, 量化与 gated RMS 调度使用已测量的候选.
[性能诊断](profiling.zh.md) 记录选择, 计数器与被排除的候选.
MLP gate/up 路径将残差 RMS 与 FP8 量化合并, 并使用之前的 GEMM. 单行使用之前的 CUDA 链.
其二十项 GPU 检查通过, 包括旧链输出精确一致, 图输入变化和非法存储拒绝.
扩展后的算子矩阵保留全部原有 230 项, 增加十六项融合投影与二十八项其余 FP8 投影用例.

离线注册向 Python 共享输出预算. Proposer 为调度器的 bonus token 预留额度.

当前 CPU collection 通过 687 项测试和 65 项 subtest, 取消选择 365 项 GPU 测试.
受影响 GPU collection 通过 104 项测试.
教程验证通过十项源码节选测试与九项浏览器测试.
短输出诊断使用输入 32768 和输出 16. 最新普通模式 prefill 中位数约为 1.39 s.
此区间仍超过当前约 0.453 s 下界的三倍.
正式性能测量使用输出 4096. 完整算子, 边界与服务回归待完成.

[验收索引](acceptance.zh.md) 保留算子, 边界, 成对比较和服务测量, 并标明源码范围.
274 项算子用例和九项 262144-token 边界用例继续有效.
完整服务验证继续有效.

## 文档与证据

英文 Markdown 遵循 ASD-STE100 Issue 9, 配套完整中文翻译.
文档和教程使用当前项目契约.
跟踪摘要保留实测值, 源码身份, 原始哈希和验证限制.
原始尝试与主机配置保存在被忽略的本地或外部目录.

用户要求的教程预览继续使用端口 18084.
被忽略的 `LOCAL.md` 记录地址, PID 和维护命令.

## 开放工作

- 后续实现变化后完成源码契约复审. 当前融合已完成独立审查.
- 完成十五行阶段时延测量, 优化失败路径后重新执行受影响的完整测量集.
- 后续源码变化后执行教程检查.
- 推理修改后执行适用的数值, 算子, 上下文和服务回归.
- 完成教程的用户验收.
- 按 [审计](audit.zh.md) 的 `PY-04`, 在实测内存压力下测试跨形状 262k 图淘汰 / 重捕获.
- 在形状无限变化时证明编译器存储有界.
- 以测量推进 `PY-06` 的主机重叠工作.
- 在完整操作证据显示增益前, 继续排除已否决的 `KRN-06` 候选.

阅读 [要求](requirements.zh.md), [测试](testing.zh.md) 和 [审计](audit.zh.md).
