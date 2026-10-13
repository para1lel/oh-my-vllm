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

The [historical serving resource record](2026-10-11-serving-stability.json) gives fifteen phase rows and six capacity cases.
It also gives two one-hour agentic collections and eight allocator-OOM recovery stages.
The [acceptance index](../../docs/acceptance.md) records current scope and pending collection.
Each result keeps its measured source and full original hash.

The [native SDK collector record](2026-10-13-native-sdk-collector.json) gives CPU fixture checks and original result hashes.
These checks do not supply model acceptance.

The [constrained-decoding investigation](2026-10-13-constrained-decoding-investigation.json) keeps recorded loop symptoms and selected conditional target probabilities.
It also records two constructed protocol defects. It does not show original sampled paths or formal acceptance.

The [current acceptance record](2026-10-13-current-source-acceptance.json) gives the source-scoped GPU, CPU, recovery, and service results.
The [framework-hour record](2026-10-13-framework-hour.json) gives two independently reviewed 3600-second windows on source `ae070e4`.
It keeps the original native audit failures and the framework-only acceptance scope.
The request and memory plots include those windows and their stop tails. Resource samples do not prove a memory plateau or full allocation attribution.

The operator matrix has 517 numerical passes and three speed failures.
The user gave approval for delivery without optimization of these cases.

## Export

```bash
scripts/with-env.sh python scripts/export_evidence.py --input "$EVIDENCE_DIR/original.json" --output "$EVIDENCE_DIR/summary.json"
```

The exporter keeps its input unchanged and rejects identical input/output locations.
Examine the summary for host data before you put it in tracked evidence.
Formal verification uses full original records.
Performance uses the [phase-latency protocol](../../docs/testing/performance.md#framework-performance).
