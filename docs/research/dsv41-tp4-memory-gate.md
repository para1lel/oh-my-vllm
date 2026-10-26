# DSV4.1 TP4 Physical Memory Gate

Date: 2026-10-26

## Decision

The DSV4.1 TP4 physical memory gate is unresolved.

Keep the current safety gate. Do not increase the 1 GiB limit. Do not subtract pre-load opaque occupancy. Do not run ordinary compute, CUDA Graph, or memcheck work before an independent resolution.

The gate rejects a rank when conservative kernel bytes plus opaque occupancy exceeds 1 GiB. This rule protects physical memory ownership during model preparation. It is not a performance limit.

## Recorded facts

The current local evidence record reports these TP4 preparation values:

| Evidence | Recorded value | Meaning |
|---|---:|---|
| Typed arena on each rank | `79,177,028,608` bytes | Model state allocation lower bound |
| Load peak | `79,178,595,328` bytes | Highest recorded load allocation |
| Unit D callback charge lower bound | `175,558,569` bytes | Known callback charge only |
| Total device memory in the v6 record | `191,505,498,112` bytes | Device total reported at the guard |
| Free device memory in the v6 record | `111,210,004,480` bytes | Free memory reported at the guard |
| Allocated in the v6 record | `79,177,028,608` bytes | Explicit allocation counter |
| Used in the v6 record | `80,295,493,632` bytes | Driver-used counter |
| Opaque occupancy in the v6 record | `1,118,465,024` bytes | Conservative opaque counter |
| Kernel bytes in the v6 record | `0` bytes | Conservative kernel counter |
| v6 excess over 1 GiB | `44,723,200` bytes | Gate rejection margin |

The v7 diagnostic adds a boundary sequence:

- Pre-load opaque occupancy was `1,115,422,720` bytes.
- Load peak opaque occupancy was `1,118,995,456` bytes.
- Post-load opaque occupancy was `1,118,465,024` bytes on rank 0.
- The load increase was `3,572,736` bytes.
- The post-load increase was `3,042,304` bytes on rank 0.
- Ranks 2 and 3 had the same load peak and a post-load increase of `2,976,768` bytes.
- Rank 1 produced no boundary record before group abort.

The v6 record therefore exceeds 1 GiB even with `kernel_bytes=0`. The v7 pre-load value also exceeds 1 GiB. The load delta is small compared with the pre-load excess. These facts rule out "weight loading alone caused the first excess" as a sufficient explanation.

The checkpoint contract records `203,073,076,240` bytes for the twelve Engram tensors across the checkpoint. It also records row sharding for MP4 and CUDA parameter loading in the reference loader. The contract does not record runtime opaque residency, allocator categories, or a pinned host-memory transfer contract. Checkpoint size alone cannot resolve this gate.

## Budget status

The table separates measured values, incomplete estimates, and unknown budgets.

| Budget category | Status | Amount or reason |
|---|---|---|
| Typed model-state allocation | Measured lower bound | Typed arena is `79,177,028,608` bytes per rank. It includes model state beyond weights. |
| Weight-only resident checkpoint | Unknown | No isolated weight-only measurement exists. |
| Complete post-load resident model state | Unknown | Execution stopped before a weights-complete record. |
| Logical checkpoint total | Measured logical total, not a resident estimate | `510,286,023,000` bytes across the checkpoint. |
| Temporary workspace | Unknown | No prepared record exists. |
| Communicators and collectives | Unknown | The opaque total is not reconciled to these owners. |
| CUDA Graph reservations | Not measured | Graph capture did not run. |
| Pinned-host Engram residency | Unknown | The reference source defines no pinned-host contract. |
| Engram logical tensor total | Measured logical total, not a resident estimate | `203,073,076,240` bytes across twelve tensors. |
| Transfer buffers | Unknown | No transfer-lifetime record exists. |
| Available KV/cache budget | Unknown | No runtime cache was created. |

No available-budget claim is made. The logical checkpoint and Engram totals do not show per-rank residency or available KV/cache capacity.

## Why the gate is unresolved

The current records show a safety violation, but they do not identify the opaque owners. The records do not reconcile the opaque total with CUDA library, communication, allocator, registration, or model-preparation categories.

The evidence is also incomplete across ranks. Rank 1 has no v7 boundary record. No record proves that all four ranks had the same pre-load state or the same load delta.

The preparation stopped at the unchanged guard. It produced no weights-complete, prepared, compute, or close record. The available result cannot prove safe post-load residency, execution behavior, or release behavior.

The repository rules set physical ownership as explicit and bounded. The execution rules also keep resource ownership explicit during preparation and graph setup. Relaxing the gate without an ownership ledger would remove the safety evidence that currently blocks TP4.

## Smallest safe next measurement

Run one controlled TP4 prepare-only diagnostic with the frozen inputs and the unchanged 1 GiB gate.

Record the same bounded fields for each rank at three points:

1. Before weight loading.
2. Immediately after weight loading.
3. At guard evaluation.

Record the rank, stage, allocated bytes, used bytes, free bytes, total bytes, kernel bytes, opaque bytes, and the accounting source. Stop at the unchanged gate. Do not run model compute, CUDA Graph capture, memcheck, or shutdown experiments.

This single run can answer the first decision question. Does each rank show the same pre-load excess? Does weight loading add material opaque occupancy? The gate stays unresolved if any rank lacks a record or if the opaque total cannot be reconciled to bounded owners.

## Sources

- [GPU execution rules](../execution.md)
- [Runtime contracts](../requirements.md)
- [DSV4.1 model and checkpoint contract](dsv41-model-checkpoint-contract.md)
- Current local evidence record: `LOCAL.md`
- Physical diagnostic records: `diagnostic-v6`, `controlled-prepare-v6-v4`, `diagnostic-v7`, and `opaque-analysis-v7`
