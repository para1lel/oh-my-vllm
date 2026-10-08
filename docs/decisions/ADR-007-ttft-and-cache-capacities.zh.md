# ADR-007: TTFT 与独立缓存容量

日期: 2026-09-22. 状态: accepted, 已实现.

## 决策

使用 batch 共同提交到首个保留 token 的 TTFT, 每请求使用单调时钟.
包括注册, 排队, 调度, 传输和采样.
2 次完整预热后, 取 5 次重复最大值的中位数.
12 行必须满足 95% 吞吐, 110% TTFT 和 10% spread 门槛.
在隔离环境中冻结基线源码 `e9f169d16b9408bb9ae44f75072b91a5521d733c`.

保持总上下文 262144, block784, state dtype 和数值容差.
增加 `--mamba-blocks`, 分开 Rust GDN 分配和 Python tensor 容量.
FA 和 GDN ID 具有独立 namespace.
未设置该选项时, 保留共享 pool 用于受控容量 / 抢占测试.
接纳检查两种容量, 拒绝无法容纳的请求.
Prefix hit 在新分配前 touch 两种 pool.

## 原因与影响

大共享 pool 在分配 FA 容量时同时分配了过多 recurrent state.
独立 1400 FA 和 128 GDN slot 减少闲置状态, 让长上下文驻留.
历史 `num_gpu_blocks` 单位保留; Python 返回 `floor(num_gpu_blocks/3)` 个 FA slot.
Block ID 直接寻址所属 group tensor.
未来自动容量计算必须保留接纳, 前缀, 引用和淘汰不变量, 逻辑分配仍由 Rust 负责.

两引擎都需让完整工作负载驻留, 无抢占.
匹配有效 token 容量, CPU affinity 和 GPU 型号 / 显存 / 驱动.
相同的原始块数量整数不代表容量等价.
记录完整配置和原始源码身份.
参见 [需求](../requirements.zh.md), [测试](../testing.zh.md) 和 [验收](../acceptance.zh.md).
后续 roofline 策略独立于当前有效协议.
