# ADR-003：MTP target 状态槽与精度

2026-09-19 接受，属于已授权的模块/精度实现范围。
在独立运行时下仍然有效：MTP 使用 BF16 GDN 状态，ordinary 使用 FP32。下文关于 vLLM 平台容量计算的内容描述的是最初动机。

draft 模型不含 GDN 层，但 target 验证 K 个 draft token 需要额外 K 个递归状态槽。Rust 预留这些槽，在 draft 被拒绝后复用，在大 prefill 分块后迁移，并保留此前已接受状态，直到下一次执行消耗它。worker 接收实际调度的 draft，返回实际被接受的输出和下一批 draft token ID。Rust 只回滚已调度且被拒绝的 draft。

当时安装的模型/后端中，默认 FP32 SSM 加四个推测槽，会让 vLLM 平台缓存定容将 block_size 从 784 增至 1568。项目要求 784，因此 MTP 显式使用 BF16 SSM 缓存存储；普通模式保留后端 auto 精度。配对 vLLM 基线采用完全相同选择。初始化断言 block_size 保持 784。这一精度选择属于基准配置，不代表对任意 draft 数量的 MTP 精度或兼容性作普遍保证。

实际路径测试观测 target GDN 融合 MTP 验证及全部 draft 状态，同时观测真实 GQA 和 prefill 调用；以相同舍入输入与 CPU FP64 参考比较。短中文生成和跨块中文生成均通过检查，输出连贯。状态误差容差及限制记录在 testing.md。完整模型 token 一致不属于验收门槛。
