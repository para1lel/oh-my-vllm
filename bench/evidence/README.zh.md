# 可移植证据

这些 JSON 摘要提供实测值, 源码身份和验证限制.
测量范围见 [验收索引](../../docs/acceptance.zh.md).

每份摘要包含 `artifact_kind="portable-derived-evidence"`, schema 1, 原始字节数和原始哈希 (SHA-256).
`data` 字段保留测量值, 源码哈希和软硬件版本.
`removed_fields` 列出已脱敏字段.
主机路径, GPU 身份, CPU core ID, 命令和原始日志保存在本地或外部目录.
键名的哈希后缀使脱敏后的字典键保持独立.

## 导出

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

导出器保持输入不变, 拒绝相同的输入 / 输出位置.
放入跟踪证据前, 检查摘要中的主机信息.
正式验证使用完整原始记录.
性能采用 [阶段时延协议](../../docs/testing.zh.md#框架性能).
