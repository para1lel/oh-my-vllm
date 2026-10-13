# 推理草稿概念

## MTP

Draft row `p` 组合 target hidden row `p-1` 和 token row `p`.
Row 0 不存在.
统一 RoPE offset 保留相对位置.
Rust 分配 draft / state 容量; Python 不自行创建逻辑块.
784-token 边界 hidden state 在所需 page 不可用时等待.

MTP4 提出 4 个草稿, 对照 target sampling 验证.
Grouped verification 最多 5 个 query, 每个 query 有自己的 causal length.
Single-token decode 路径独立保留.
Worker 保留接受的 target output 和正确 GDN / convolution snapshot.
被拒绝的已调度草稿回滚 computed progress; 未调度草稿不需要回滚.
Grammar simulation 使用各 speculative prefix, 提交保留输出前先回滚.

## DSpark

可选模式通过项目内代码加载本地 DSpark checkpoint.
执行前校验配置, tensor 名称, shape, BF16 dtype 和稳定的源文件身份.
加载的配置和权重哈希标识实际提供 tensor 的文件字节.
不执行 checkpoint 内的 Python 代码.

草稿使用 5 层 GQA, 以及从 0 开始编号的 target 层输出 `(5,19,33,47,61)`.
每个 feature 是下一次 normalization 前的 BF16 层输出.
原生 MTP 和普通计算不提取这些 feature.
共享的 target embedding 和 vocabulary head 提供 DSpark token 表示及 base logits.
草稿将拼接的 target feature 投影为其 5 层的 context KV.

DSpark context page 使用相同的 Rust 分配 FA page ID 和 784-token layout, 保存在独立物理缓存中.
只有已提交的 target 输入行进入 context.
被拒绝的输入行可以保留在 committed cursor 之后的 target FA 中, 后续使用前会被覆盖.
DSpark context 只接收保留输入行.
7 个 proposal row 使用双向草稿 attention 和临时 KV, 不进入持久 context.

Markov head 用前一个 candidate token 修正每行 base logits.
Confidence head 通过累计条件 confidence 限制 candidate prefix.
`--dspark-confidence-threshold` 初始设置为 `0.2`.
首项 confidence 乘积低于阈值时返回 0 个 candidate, 该步使用一个目标 token.
设置为 `0.0` 可在固定数量实验中保留至多 7 个合法 row.
输出和上下文预算可以缩短这个前缀.

Draft block 7 和 training block 16 独立于 target block784.

Greedy verification 将 candidate 与 target argmax token 比较.
随机输出保留每个 candidate 的完整条件 proposal 分布 `q`.
Target 分布 `p` 和 proposal 分布 `q` 都使用配置的 temperature, penalty, top-k / top-p 和 grammar mask.
以概率 `min(1,p(x)/q(x))` 接受 candidate `x`.
拒绝后从归一化的 `max(p-q,0)` 采样, 全部接受后从 `p` 采样 bonus token.
草稿和 target 采样使用各自独立的随机数生成器.

既有 GDN snapshot / rollback 逻辑保留已提交输入后的状态, 验证行数至多 8.
推测状态为 BF16, target 和草稿保留规定的 BF16 舍入点.
Grammar simulation 先回滚, 再由已提交输出更新持久 grammar 状态.
取消和抢占释放 target 与草稿的请求状态.

worker 启动时选择一种模式. 在 worker 停止前保持该模式.
参见 [ADR-009](../decisions/ADR-009-dspark-optional-mode.zh.md).
