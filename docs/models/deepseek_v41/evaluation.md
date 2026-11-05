# Numerical acceptance

Full model acceptance uses an independent, frozen MP4 text reference with a recorded quantizer synchronization patch.
Expected arrays cover target vocabulary vectors, hidden endpoints, DSpark5 proposals, confidence, verification, and accepted-state restoration.
The [quantization contract](quantization.md) defines the model's numerical order.

The [repository checker](../../../development/model/check_deepseek_reference.py) reads the frozen fixtures and runs the candidate Worker through native operators.
The SHA256 values of the recovered fixture manifest and endpoint bytes are equal to the frozen values.
The fixture package stays external to the repository.
The execution package from the initial reference is not full.

## Frozen contract

The frozen reference driver sets these limits before candidate execution:

| Output | Limit |
|---|---|
| Full target and proposal logits | Relative L2 error at most 0.01 |
| Hidden endpoints | Relative L2 error at most 0.01 |
| Confidence | Absolute error at most 0.01 |
| Greedy, proposal, and acceptance IDs | IDs must be equal |

Restoration uses a new replay of accepted tokens only.
Truncating state after rejected computation is not an independent restoration reference.
The reference includes forced acceptance counts from zero to five, corrections, and subsequent target and DSpark endpoints.
Expected arrays and limits stay fixed during candidate acceptance.
Operator checks do not replace this contract.

The frozen objects have these SHA256 identities:

| Object | SHA256 |
|---|---|
| `manifest.json` | `2c40ce592665c14a7334657be7fdcf88fd7ea29c9a276c6fa7b159f5cdd8bbe4` |
| `endpoints.safetensors` | `3089f8e3d0327a07f166c744b25d7751fc65175963fbbd588533d34643568093` |
| `execution-manifest.json` | `82c8f49a4c6341d68716099825e10fdbeddd206295f302f2738db6507fe7579d` |
| Reference driver | `c4ac8b03345dd465c1a20b34e7cfbfe2a4891b6bb99b8a920e4f05d89900c909` |

The frozen driver and synchronized source reproduced the endpoint file in two executions.
In these executions, all 143 tensors and the full endpoint files were equal byte for byte.
The fixture manifest was reconstructed from the initial field order and recorded identities.
The SHA256 of the full manifest is equal to its frozen value.
This does not show that the initial manifest file was found.

Execution records from this recovery, checkpoint digest lists, and the reconstructed patch have different identities in the reviewed recovery package.
The initial patch bytes, checkpoint digest-list files, and preparation and compiled-cache attachments are not available.
Installed library hashes do not show which GPU libraries were mapped during execution.
The recovered execution manifest from the initial reference does not supply these missing attachments.
Host artifact locations belong in ignored `LOCAL.md`.

## Run the independent check

Set `DEEPSEEK_REFERENCE_ROOT`, `MODEL_ROOT`, and `ARTIFACT_ROOT` from local configuration.
Set `CHECKPOINT_DIGESTS` to the digest record for the 48 input shards from this reference conversion.
Keep converted MP4 weights in a different directory from the endpoint fixtures.

First, do a CPU check of the frozen identities, full tensor schema, and acceptance fields:

```bash
scripts/with-env.sh python -m development.model.check_deepseek_reference \
  --reference "$DEEPSEEK_REFERENCE_ROOT" \
  --output "$ARTIFACT_ROOT/deepseek-reference-validation" \
  --validate-only
```

Set `NUMERICAL_GPU_UUIDS` to four ordered GPU UUIDs.
Use a new output directory for each execution:

```bash
OH_MY_VLLM_GPU_COUNT=4 OH_MY_VLLM_GPU_UUIDS="$NUMERICAL_GPU_UUIDS" \
  scripts/with-env.sh scripts/with-gpu.sh \
  python -m development.model.check_deepseek_reference \
  --reference "$DEEPSEEK_REFERENCE_ROOT" \
  --model "$MODEL_ROOT" --strategy tp4 \
  --checkpoint-digests "$CHECKPOINT_DIGESTS" \
  --evidence "$DEEPSEEK_REFERENCE_ROOT/bundle-manifest.json" \
  --output "$ARTIFACT_ROOT/deepseek-reference-tp4"
```

Do the executions again with `--strategy tep4` and `--strategy dep4` in different output directories.
Each execution compares candidate checkpoint bytes with the inputs of this reference conversion.
It records source, native binary, configuration, tokenizer, runtime, strategy, and ordered GPU identities.
The digest record from this recovery does not replace the missing digest-list files from the initial reference.

The default selection includes all nine cases and 143 frozen tensors.
The cases include short target calls, the window boundary, authentic proposals, and forced acceptance counts from zero to five.
The checker compares each full vocabulary vector and hidden endpoint with the frozen expected arrays.
It uses the limit for relative L2 from the frozen contract.
Greedy, proposal, verification, acceptance, and correction IDs must be equal to the frozen expected IDs.
Confidence uses the limit for maximum absolute error from the frozen contract.

The checker also runs six-token native verification.
It then does a state commit for the accepted endpoint.
It restores that endpoint through the candidate cache.
It compares the restored state with a new replay of accepted tokens only.
It compares subsequent target and DSpark endpoints with the frozen expected arrays.

The new replay does not generate or change expected arrays.
Raw arrays, state records, identities, and comparison reports stay in the external output directory.

The checker uses Eager execution with one request (`max_num_seqs=1`) and batch decode disabled.
DEP4 uses rank zero as the dense request owner and three ranks for expert work only.
Its expert exchange uses the production Worker configuration.
This scope does not include more than one request owner or batches with more than one request.

It does not select a different strategy after a failure.
A check with a `--cases` subset does not include all cases.
This entry does not show CUDA Graph, Rust service, HTTP, AgentX, or long-context acceptance.

A missing fixture or a different frozen SHA256 stops the check.
Do not replace frozen identities or limits during candidate acceptance.
A new reference must have a new review.

## Reference generation

Use an isolated development environment for the released TileLang reference.
Do not add TileLang to inference dependencies.
TileLang 0.1.8 can reuse input shared storage before all quantizer input stores complete.
The recorded patch adds `T.sync_threads()` immediately before `T.copy(y_local, y_shared)` in `act_quant_kernel` and `fp4_quant_kernel`.
It changes no equation, cast, or checkpoint value.

The patch SHA256 is `a2fcc1970f543f1a360a5f7554e37b21a1a272cfcab463228d6c4377194f4968`.
The modified quantizer SHA256 is `6206c3bad60de34247481e9c1ab8c487c7930ffca0f773f8c296ba05db879084`.
The [quantization contract](quantization.md) gives the unmodified quantizer identity.
A description of two barriers does not replace the initial patch bytes.

Record source, patch, compiler, runtime, tokenizer, checkpoint, input cases, and expected-array identities in the reference manifest.
Do a check of packed and in-place formats with Compute Sanitizer Racecheck before expected-array generation.
Freeze arrays only when target and DSpark endpoints from two executions are equal bit for bit.
Identify this backend as a reference with a synchronization patch.
Do not import candidate modules to generate expected arrays.

## Supporting checks

The [released-operand check](../../../development/operators/check_deepseek_released.py) compares captured operator outputs bit for bit.
The [projection check](../../../development/model/check_quantized_gemm.py) compares an immutable native baseline.
The [batch check](../../../development/model/check_deepseek_batch.py) compares candidate batch execution and state behavior with individual candidate execution.
These checks have different scopes from the independent full-model reference.
They do not show full HTTP, AgentX, long-context, or DEP4 acceptance.
