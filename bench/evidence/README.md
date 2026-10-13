# Portable evidence

Summaries keep measured values, source identities, original hashes, and verification limits.
The [acceptance index](../../docs/acceptance.md) defines their source scope.
Current refactor acceptance is pending. Obsolete provider evidence stays only in Git history.

Each portable record has `artifact_kind="portable-derived-evidence"`, schema 1, original byte count, and original hash (SHA-256).
Its `data` field keeps measurements and software/hardware identities.
Its `removed_fields` list identifies redacted fields. Raw logs and host configuration stay in external storage.

The [native SDK record](2026-10-13-native-sdk-collector.json) covers CPU fixtures, not target-model acceptance.
The [decoding investigation](2026-10-13-constrained-decoding-investigation.json) keeps original symptoms and conditional measurements.
It does not prove original sampled paths or agent task completion.
The [historical source record](2026-10-13-current-source-acceptance.json) applies only to its recorded source.

## Export

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

The exporter keeps its input unchanged and rejects identical input/output locations.
Examine summaries for host data before commit. Formal verification uses full original records.
Performance uses the [phase protocol](../../docs/testing/performance.md#framework-performance).
