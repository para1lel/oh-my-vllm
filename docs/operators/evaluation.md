# Numerical evaluation

Use the [model contracts](model.md) and [development commands](development.md).
The [tile design](../tile-execution.md) controls execution coverage.
Only the [HTTP comparison](../performance.md) has a performance gate.

## Numerical scope

Use the [fixed numerical cases](../../development/reference/cases.py) for numerical acceptance.
Run each case in eager mode and a CUDA Graph.
Do not run CPU FP64 references for large modeled cases.
A small row count alone does not limit attention reference work. The context length also controls this work.

| Operator | Numerical cases |
|---|---|
| FFN | Rows 1, 4, 5, 20, 65, 129, and 144 |
| GDN | Default rows 1, 4, 5, and 20. One embedding row. 65 rows with two media rows |
| GDN boundaries | Default 65 rows at position 0. 129 rows at position 770. Empty input guards |
| FA | Rows 1, 4, 5, 20, and 33 with context 256. Five rows with context 786 |
| Sampling | All modeled cases: rows 1, 4, 5, and 20, with the full vocabulary |
| MTP context | Four rows in one request. 16 rows in four requests |
| MTP draft | Rows 1 and 4 with context 256. The `full` and `query_only` variants |
| Vision | Two frame groups of 196 patches. `input`, `default`, and `merge` variants |

The shifted GDN case has checkpoint rows 13 and 128.
It crosses a 784-token checkpoint, then processes a full 64-token block and a short tail.
FA checks include the prefill path and a cache page boundary.
Vision checks include changes to one frame group and unchanged outputs in the other group.

The 65-row FFN case uses the prefill projection path.

Each provider states the elements that its reference checks.
Unchecked elements have no numerical evidence.
Missing providers or failed checks prevent correctness acceptance.
Do not use a reference implementation in place of a missing provider.
