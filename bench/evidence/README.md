# Portable evidence

These JSON summaries give measured values, source identities, and verification limits.
Use the [acceptance index](../../docs/acceptance.md) for measured scope.

Each summary has `artifact_kind="portable-derived-evidence"`, schema 1, original byte count, and original hash (SHA-256).
The `data` field keeps measurements, source hashes, and software/hardware versions.
The `removed_fields` list identifies redacted fields.
Host paths, GPU identities, CPU core IDs, commands, and raw logs stay in local or external storage.
Hashed key suffixes keep redacted dictionary keys distinct.

Files with `artifact_kind="derived_diagnostic_summary"` use a different scope.
They keep experiment names, original hashes, selected measurements, and verification limits.
They have `formal_acceptance=false`.
The [owned-kernel and liveness summary](2026-10-10-owned-tuning-liveness.json) uses this format.

## Export

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

The exporter keeps its input unchanged and rejects identical input/output locations.
Examine the summary for host data before you put it in tracked evidence.
Formal verification uses full original records.
Performance uses the [phase-latency protocol](../../docs/testing.md#framework-performance).
