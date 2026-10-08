# Portable historical evidence

These JSON files are derived records for historical reference and offline analysis.
They do not show performance on another server.
Use the [acceptance index](../../docs/acceptance.md) for source scope and gates.

Each summary has `artifact_kind="portable-derived-evidence"`, schema 1, initial name, byte count, and original hash (SHA-256).
Its `data` keeps measurement values, source hashes, and software/hardware versions where present.
`removed_fields` identifies omitted or redacted content.
Host paths, GPU identities, CPU core IDs, commands, and raw logs are removed.
CPU affinity count stays when available.

Hashed suffixes keep redacted dictionary keys distinct.
The value `10.4.0.35` is a nvidia-curand package version.

[originals.json](originals.json) records all 66 initial tracked files, with two logs.
Ignored `.local/evidence/` holds their byte-identical local copies.
The source commit stays available in Git history.
Keep its original hashes distinct from derived-file hashes and nested external collector hashes.

## Export

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

The exporter keeps its input unchanged and refuses an identical output location.
Review the summary for host data before you move it into tracked evidence.
It removes arbitrary command/log payloads instead of attempting partial log redaction.

## Formal use

`benchmarks/ttft.py` and the historical comparator reject derived envelopes.
Formal comparison must use full original records, full repetitions, and matching hardware/configuration.
New servers must have new baseline measurements with the [current protocol](../../docs/testing.md#framework-performance).
The historical nine-row baseline had only three repetitions and fails the current sample-count gate.
Current twelve-row limits are 95% throughput, 110% TTFT, and 10% spread until a new roofline policy replaces them.
