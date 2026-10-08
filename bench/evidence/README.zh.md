# 可移植历史证据

这些 JSON 是用于历史查阅和离线分析的派生记录.
它们不能证明其他服务器的性能.
源码范围和门槛见 [验收索引](../../docs/acceptance.zh.md).

每份摘要包含 `artifact_kind="portable-derived-evidence"`, schema1, 原始名称, 字节数和原始 SHA-256.
`data` 保留已有测量值, 源码哈希及软件 / 硬件版本.
`removed_fields` 标明删除或脱敏内容.
主机路径, GPU 身份, CPU core ID, 命令和原始日志已移除.
已有 CPU affinity count 会保留.
哈希后缀让脱敏 dictionary key 保持唯一.
`10.4.0.35` 是 nvidia-curand package version.

[originals.json](originals.json) 记录全部 66 份原跟踪文件, 包括 2 份日志.
被忽略的 `.local/evidence/` 保存字节相同的本地副本.
原始提交仍可从 Git 历史读取.
区分原始哈希, 派生文件哈希和嵌套外部采集器哈希.

## 导出

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

导出器不改变输入, 拒绝相同输出位置.
移入跟踪证据前审查摘要是否仍有主机数据.
任意 command/log payload 整段删除, 不做局部日志脱敏.

## 正式用途

`benchmarks/ttft.py` 和历史比较器拒绝派生 envelope.
正式比较需要完整原始记录, 完整重复集合及匹配 hardware/configuration.
新服务器按 [当前协议](../../docs/testing.zh.md#框架性能) 测量新基线.
历史 9 行基线只有 3 次重复, 无法通过当前样本数量门槛.
当前 12 行仍使用 95% 吞吐, 110% TTFT 和 10% spread 门槛, 直到独立 roofline 策略替代.
