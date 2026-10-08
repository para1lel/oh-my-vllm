# ADR-005: 历史 V2 runner

日期: 2026-09-21. 状态: 由 [ADR-006](ADR-006-independent-runtime.zh.md) 替代.

## 原始决策

过渡 adapter 选择 V2 model runner, 保持持久请求状态和显式 graph 生命周期.
Rust 保持已接受历史, 缓存所有权和调度.
接纳 / 恢复时传完整已接受历史; 普通解码保持增量, 草稿独立.
迁移保留离线 / 服务功能, MTP, mask, 前缀复用和取消.

## 替代

独立 runner 通过项目代码保留这些状态和生命周期合同.
V2 adapter, vLLM scheduler 和 legacy runner 均已移除.
原 9 行验收属于历史, 由 12 行协议替代.
有效行为以 [架构](../architecture.zh.md), [需求](../requirements.zh.md) 和 [验收](../acceptance.zh.md) 为准.
