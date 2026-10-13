# 可移植证据

摘要保留测量值, 源码身份, 原始哈希和验证限制.
[验收索引](../../docs/acceptance.zh.md) 定义其源码范围.
当前重构验收尚未完成. 旧 provider 证据仅保留在 Git 历史中.

每份可移植记录包含 `artifact_kind="portable-derived-evidence"`, schema 1, 原始字节数和原始哈希 (SHA-256).
`data` 字段保留测量值及软件 / 硬件身份.
`removed_fields` 列出脱敏字段. 原始日志及主机配置保留在外部存储.

[原生 SDK 记录](2026-10-13-native-sdk-collector.json) 覆盖 CPU fixture, 不证明真实模型验收通过.
[解码调查](2026-10-13-constrained-decoding-investigation.json) 保留原始症状和条件测量.
它不能证明原始采样路径或 agent 任务完成.
[历史源码记录](2026-10-13-current-source-acceptance.json) 仅适用于其记录的源码.

## 导出

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

导出器保持输入不变, 拒绝相同输入 / 输出位置.
提交前检查摘要中的主机信息. 正式验证使用完整原始记录.
性能遵循[阶段协议](../../docs/testing/performance.zh.md#框架性能).
