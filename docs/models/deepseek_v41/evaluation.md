# Numerical acceptance

Full model acceptance uses an independent, frozen MP4 text reference with a recorded quantizer synchronization patch.
Expected arrays cover target vocabulary vectors, hidden endpoints, DSpark5 proposals, confidence, verification, and accepted-state restoration.
The [quantization contract](quantization.md) defines the model's numerical order.

The original external reference tree is missing.
The full fixture manifest and endpoint bytes are not available in this checkout.
There is no maintained repository command that restores and runs the full independent model check.
Full-model numerical acceptance cannot be repeated with the current artifacts.

## Frozen contract

The original reference driver sets these limits before candidate execution:

| Output | Limit |
|---|---|
| Full target and proposal logits | Relative L2 error at most 0.01 |
| Hidden endpoints | Relative L2 error at most 0.01 |
| Confidence | Absolute error at most 0.01 |
| Greedy, proposal, and acceptance IDs | Exact equality |

Restoration uses a fresh replay of accepted tokens only.
Truncating state after rejected computation is not an independent restoration reference.
The reference includes forced acceptance counts from zero through five, corrections, and subsequent target and DSpark endpoints.
Expected arrays and limits stay fixed during candidate acceptance.
Operator checks do not replace this contract.

The frozen objects have these SHA256 identities:

| Object | SHA256 |
|---|---|
| `manifest.json` | `2c40ce592665c14a7334657be7fdcf88fd7ea29c9a276c6fa7b159f5cdd8bbe4` |
| `endpoints.safetensors` | `3089f8e3d0327a07f166c744b25d7751fc65175963fbbd588533d34643568093` |
| `execution-manifest.json` | `82c8f49a4c6341d68716099825e10fdbeddd206295f302f2738db6507fe7579d` |
| Reference driver | `c4ac8b03345dd465c1a20b34e7cfbfe2a4891b6bb99b8a920e4f05d89900c909` |

Recovered external records contain the execution manifest, driver, and partial fixture-manifest fields.
Those records supply these identities and limits.
Partial fields do not replace the original manifest or endpoint bytes.
Host artifact locations belong in ignored `LOCAL.md`.

## Restore independent fixtures

1. Get the original manifest, endpoint file, execution manifest, source files, synchronization patch, driver, and checkpoint digest lists.
2. Put these artifacts in an external directory.
3. Check the original manifest and endpoint SHA256 values against the frozen identities.
4. Check the source, patch, driver, tokenizer, checkpoint, compiler, and runtime identities against the full manifest.
5. Use the released MP4 checkpoint conversion only when the recorded reference driver uses converted shards.
6. Keep converted weights separate from the small endpoint fixtures.
7. Compare full frozen endpoint vectors with a separate candidate execution.
8. Keep the comparison report and candidate identities external to the repository.

If an original artifact is unavailable, report the full-model check as blocked.
Matching shapes, token IDs, or numerical tolerances cannot prove restoration of the original bytes.
Do not replace the frozen SHA256 values with hashes of newly generated arrays.
A new reference is a separate reviewed reference change.

## Reference generation

Use an isolated development environment for the released TileLang reference.
Do not add TileLang to inference dependencies.
TileLang 0.1.8 can reuse input shared storage before all quantizer input stores complete.
The recorded patch adds `T.sync_threads()` immediately before `T.copy(y_local, y_shared)` in `act_quant_kernel` and `fp4_quant_kernel`.
It changes no equation, cast, or checkpoint value.

The patch SHA256 is `a2fcc1970f543f1a360a5f7554e37b21a1a272cfcab463228d6c4377194f4968`.
The modified quantizer SHA256 is `6206c3bad60de34247481e9c1ab8c487c7930ffca0f773f8c296ba05db879084`.
The [quantization contract](quantization.md) gives the unmodified quantizer identity.
A description of two barriers does not replace the original patch bytes.

Record source, patch, compiler, runtime, tokenizer, checkpoint, input cases, and expected-array identities in the reference manifest.
Check packed and in-place formats with Compute Sanitizer Racecheck before expected-array generation.
Freeze arrays only when repeated target and DSpark endpoints are bit-exact.
Identify this backend as a reference with a synchronization patch.
Do not import candidate modules to generate expected arrays.

## Supporting checks

The [released-operand check](../../../development/operators/check_deepseek_released.py) compares captured operator outputs bit for bit.
The [projection check](../../../development/model/check_quantized_gemm.py) compares an immutable native baseline.
The [batch check](../../../development/model/check_deepseek_batch.py) checks candidate execution and state behavior.
These checks have different scopes from the independent full-model reference.
They do not prove full HTTP, AgentX, long-context, or DEP4 acceptance.
