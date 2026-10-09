# 当前状态与开放工作

更新日期: 2026-10-10.

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

## 正在实施的性能工作

收集器使用十五项固定负载, 逐请求记录提交, 首 token 和末 token 边界.
Prefill 与 decode 分别检查墙钟时延和理论下界.
规范模型计入必要参数读取, 共享资源, 语义依赖, 有效 query, KV 区间, state 与草稿.
独立数学审查和 fail-closed 操作检查通过. 当前源码的完整采集仍待执行.

执行使用 CUDA Graph, 多个 CUDA stream 和 PDL.
GDN 准备汇合独立 gate 分支. 有界 prefill 图恢复 FA 和 GDN 写入.
Allocator 拓扑复用使诊断 capture storage 从 29.13 降至 7.82 GB.
Prefill capture 仅在至少有 32 GiB 空闲显存时开始. Replay 命中保持原路径.

自有 FP8 GEMM 保持 checkpoint FP32 scales, 控制自身 stream 和启动.
大型 MN gate/up 使用 K256 tile, 三级流水和 swizzle 16. K-major 布局参考保持 K128 tile 与五级流水.
成对完整操作测量保持逐位输出.
在 32144 和 32290 行时, 平均耗时均减少约 0.487 ms.
[性能诊断指南](profiling.zh.md) 记录计数器, 已选配置和未采用的候选.

四项受影响完整操作用例通过数值与速度检查.
[K tile 与调度诊断](../bench/evidence/2026-10-10-fp8-k-tile-scheduling-diagnosis.json) 保留源码身份, 每次启动十五项计数器, 实际加载库的摘要和原始摘要.

五种投影在 624 到 2496 行时使用两个 block 的 CTA cluster 与 TMA multicast.
残差 RMS 和 GDN 输出在指定舍入点融合归一化与 FP8 量化.
小型归一化投影保持原 CUDA 整链.
GDN value 使用 packed 行的向量复制. FP32 递归 batch 2 和 4 使用实测选择的 row/warp tile.

持久化 MTP 和目标最终 prefill 移除未使用的历史 attention 与 MLP 输出.
它们保留必要 KV, 边界特征和验证样本.
不同 context graph 恢复 capture, 验证完整 replay metadata.
规范计算保持实际行, 不相交参数片段和绝对 MTP KV 区间.

仅在成功分配至少一页 prefill 后, 因对齐缩为一个 token 的 prefill 才会等待下一轮.
纯 decode 轮次和草稿验证保持正输入推进.
生产 token 预算仍为 32768. 四十七项调度器测试通过, 覆盖新 prefill, 连续 prefill, 四个/七个草稿和 checkpoint 推进.

源码 `9c336e9` 完成八项有效完整输出阶段负载.
四项的两个阶段通过: 普通解码 batch 1, MTP4 batch 1 和 2, DSpark batch 1.
普通解码 batch 2 和 4, MTP4 batch 4, DSpark batch 2 的 prefill 失败, decode 通过.
另一次 MTP4 batch-2 尝试未通过缓存审计. 保留该次尝试及其原因.

后续 dirty-source 调度诊断在公平性修正前完成五组完整输出负载.
DSpark batch 2 未通过外部进程保护. 这些尝试不提供当前源码验收.
上述负载中所有任务所属 worker 已退出.

当前 CPU 采集通过 748 项测试, 未选择 435 项 GPU 测试, 另有 65 项 subtest 通过.
二十项 GPU 投影检查通过, 覆盖 graph 输入和原始 scales 的修改, 布局一致性与存储检查.
Rust workspace 测试, 格式, 行宽, Clippy, Ruff 和文档检查通过.
教程通过十项节选测试和九项浏览器测试.

317 项算子, 十五项阶段负载, 九项容量负载和完整服务检查继续有效.
此前 dirty-source DSpark 容量结果无 OOM 或重算抢占, 但早于后续 kernel 修改.
[验收](acceptance.zh.md) 记录源码范围和原始尝试. 当前源码容量与服务采集仍待执行.

## 文档与证据

英文 Markdown 遵循 ASD-STE100 Issue 9, 配套完整中文翻译.
文档和教程使用当前项目契约.
跟踪摘要保留实测值, 源码身份, 原始哈希和验证限制.
原始尝试与主机配置保存在被忽略的本地或外部目录.

用户要求的教程预览继续使用端口 18084.
被忽略的 `LOCAL.md` 记录地址, PID 和维护命令.

## 开放工作

- 后续实现变化后完成源码契约复审. 当前选中输出与成本模型修改已完成独立审查.
- 完成十五行阶段时延测量, 优化失败路径后重新执行受影响的完整测量集.
- 后续源码变化后执行教程检查.
- 推理修改后执行适用的数值, 算子, 上下文和服务回归.
- 完成教程的用户验收.
- 按 [审计](audit.zh.md) 的 `PY-04`, 在实测内存压力下测试跨形状 262k 图淘汰 / 重捕获.
- 在形状无限变化时证明编译器存储有界.
- 以测量推进 `PY-06` 的主机重叠工作.
- 在完整操作证据显示增益前, 继续排除已否决的 `KRN-06` 候选.

阅读 [要求](requirements.zh.md), [测试](testing.zh.md) 和 [审计](audit.zh.md).
