# 可移植证据

这些 JSON 摘要提供实测值, 源码身份和验证限制.
测量范围见 [验收索引](../../docs/acceptance.zh.md).

每份摘要包含 `artifact_kind="portable-derived-evidence"`, schema 1, 原始字节数和原始哈希 (SHA-256).
`data` 字段保留测量值, 源码哈希和软硬件版本.
`removed_fields` 列出已脱敏字段.
主机路径, GPU 身份, CPU core ID, 命令和原始日志保存在本地或外部目录.
键名的哈希后缀使脱敏后的字典键保持独立.

`artifact_kind="derived_diagnostic_summary"` 的文件采用不同范围.
它们保存实验名称, 原始哈希, 所选测量值和验证限制.
其 `formal_acceptance=false`.
[自有算子与输出需求摘要](2026-10-10-owned-tuning-liveness.json) 使用这一格式.

[历史服务资源记录](2026-10-11-serving-stability.json) 给出十五项阶段负载和六项容量案例.
它还给出两项一小时 agentic 采集和八项 allocator-OOM 恢复阶段.
[README](../../README.zh.md#性能实测) 给出阶段表格和独立的 MTP4 / DSpark 请求图.
每项结果保留其实测源码和完整原始 SHA.

[原生 SDK collector 记录](2026-10-13-native-sdk-collector.json) 提供 CPU fixture 检查及原始结果哈希.
这些检查不提供模型验收.

[约束解码调查](2026-10-13-constrained-decoding-investigation.json) 保留已记录的循环症状及选定的 target 条件概率.
它也记录两项构造出的协议缺陷. 它不能确定原始采样路径, 也不提供正式验收.

[当前验收记录](2026-10-13-current-source-acceptance.json) 给出注明源码范围的 GPU, CPU, 恢复及服务结果.
[框架一小时记录](2026-10-13-framework-hour.json) 给出源码 `ae070e4` 上两个经独立复查的 3600 秒窗口.
它保留原始 native 审计失败及框架验收范围.
请求及内存图包含这些窗口和停止尾部. 资源采样不能证明内存平台或完整分配归因.

算子矩阵有 517 项数值通过及 3 项速度失败.
用户同意交付且暂不优化这些案例.

## 导出

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

导出器保持输入不变, 拒绝相同的输入 / 输出位置.
放入跟踪证据前, 检查摘要中的主机信息.
正式验证使用完整原始记录.
性能采用 [阶段时延协议](../../docs/testing.zh.md#框架性能).
