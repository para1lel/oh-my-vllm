# ADR-003: MTP 状态 slot 与精度

日期: 2026-09-19. 状态: accepted, 在独立运行时中仍有效.

## 背景

Draft layer 没有 GDN, 但 target verification 需要为已调度草稿保留额外 recurrent state.
原后端的 FP32 状态容量计算在 4 个草稿时将块大小从 784 增至 1568.
项目合同固定块大小为 784.

## 决策

MTP 使用 BF16 GDN 状态, 普通执行使用 FP32.
Rust 保留 speculative slot, 保护前一已接受状态直到下一次执行消费它.
Worker 返回保留的 target token, 接受草稿数量和下一批草稿.
Rust 只回滚已调度且被拒绝的草稿.

## 影响

精度属于实测配置, 不作为通用 MTP 精度结论.
保持 block784 和实际路径 FP64 参考容差.
测试覆盖 target verification state, GQA/GDN, 分块边界, 前缀复用和连贯文本.
完整模型 token identity 不作为门槛.
参见 [精度需求](../requirements.zh.md#req-acc-001-数值精度) 和 [验收](../acceptance.zh.md).
