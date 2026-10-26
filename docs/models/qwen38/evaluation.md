# Full-model numerical evaluation

Use the [model contract](model.md) and [development commands](README.md#numerical-commands).
The [GPU execution rules](../../execution.md) control execution.

## Numerical scope

The default case uses a fixed 16-token text prompt and one decode iteration.
The prompt is `The quick brown fox jumps over the lazy dog. Explain why the sky is blue`.
The [capture code](../../../development/model/capture.py) specifies its token IDs.
The comparison includes all 248320 logits from embedding through all 64 layers and final vocabulary projection.
It does not compare intermediate operators or apply single-operator tolerances.
The [operator microbench suite](operators.md) supplies additional local checks with stored PyTorch fixtures.

The [reference generator](../../../development/model/reference.py) uses direct PyTorch FP64 computations.
It decodes checkpoint E4M3 weights with their original 128-by-128 scales into FP64.
The change from BF16 checkpoint values to FP64 has no rounding error.
All subsequent arithmetic, activations, normalization, attention, and recurrent state use FP64.
The reference does not use activation quantization, BF16 state storage, or intermediate low-precision rounding.

It operates independently of inference kernels and old operator outputs.

The reference first computes the prompt and keeps its FP64 state.
Its highest prompt logit selects the fixed input token for the one decode iteration.
The implementation uses that same token after its own prompt computation.

Thus, the decode comparison includes accumulated prompt error and the full decode computation.
The comparison uses Eager prefill and CUDA Graph decode.
The report records the execution policy.
There is no generated-token loop.

Use `--mode prefill` to compare the same prompt's last row when prefill changes.
The stored artifact contains the two endpoint vectors and their checkpoint identity.
The added cases below check long-context, MTP, and vision computation.
Use different checks for protocols, state ownership, and numerical results.

## Added fixed cases

The [case definitions](../../../development/model/cases.py) specify stored inputs and complete vocabulary outputs.
Usual tests load repository fixtures without PyTorch.

| Case | Fixed input | Vocabulary outputs |
|---|---|---|
| `mtp-restored` | 784-token prefix, five verification rows, two selected rows | All five verification rows, restored decode, first MTP draft, next MTP draft |
| `image-video` | 178-token prefix, one image, three video frames | Prefill, decode, first MTP draft |
| `context-32769` | 32768-token prefix and one decode token | Prefill and decode |

The restoration case clears the request's private state reference before verification.
Verification reads the logical 784-token checkpoint.
State selection commits the first two input rows.
The next target computation reads that selected state.
MTP context uses the stored boundary hidden row and the selected verification rows.
Its two draft inputs use fixed tokens.

The media case checks the image and video transformer, merger, language embedding replacement, multiaxis positions, and MTP embedding replacement.
The three-frame video checks final-frame duplication.
The long case checks complete causal attention and accumulated recurrent state before decode.

The [added reference generator](../../../development/model/reference_cases.py) uses direct PyTorch FP64 formulas.
Query tiles limit attention memory while each query keeps its complete causal key range.
The generator stores inputs, complete endpoint arrays, checkpoint identity, and reference provenance in compressed shards.
Each shard is at most 32 MiB.
The fixture index binds each manifest by SHA256.

Set `MODEL_ROOT` to the checkpoint directory.
Set `LOGITS_REPORT` to a new external JSON path.
Run one added case:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --case mtp-restored --report "$LOGITS_REPORT"
```

Use the other case names for their fixed checks.
Each command compares all listed endpoints.
The verification comparison checks each row independently.

For explicit fixture generation, set `MODEL_FIXTURES` to a new external directory.
Use the separate reference environment:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference_cases --model "$MODEL_ROOT" --output "$MODEL_FIXTURES"
```

Examine the generator, manifest hashes, and complete endpoint arrays before copying fixtures into the repository.
Do not generate expected arrays during acceptance.

## Evidence and limits

Keep the frozen [FP64 artifact](../../../development/model/reference.npz) in the repository.
Record its SHA256, checkpoint identity, reference source identity, PyTorch version, and input token IDs.
Reject changed artifacts, checkpoint files, inputs, nonfinite logits, or incomplete vocabulary vectors.

Do not replace the reference with the inference output.
Do not overwrite artifacts during reference generation.

The report gives NRMSE, maximum absolute error, cosine similarity, and the two top-1 token IDs.
NRMSE is the L2 norm of the logit difference divided by the L2 norm of the FP64 logits.
Cosine similarity uses the full vectors without centering.
Top-1 equality is a diagnostic because small errors can change the order of almost equal logits.

The [comparison policy](../../../development/model/compare.py) supplies the fixed full-model limits and artifact identity.
All three limits must pass for each endpoint row. A failed comparison exits with an error.

| Case | Maximum NRMSE | Minimum cosine | Maximum absolute error |
|---|---|---|---|
| Short prompt | 0.05 | 0.999 | 0.5 |
| `mtp-restored` | 0.09 | 0.996 | 1.6 |
| `image-video` | 0.06 | 0.998 | 1.3 |
| `context-32769` | 0.07 | 0.998 | 1.1 |

The short case keeps its original limits.
The added limits use [vLLM commit 86129933](https://github.com/vllm-project/vllm/commit/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af) and the selected project implementation, with the same checkpoint, stored inputs, and FP64 references.
For each case, take the largest NRMSE and absolute error, and the smallest cosine across all its endpoint rows.
Round these values outward to steps of 0.01, 0.1, and 0.001, respectively.
Set the limits independently for each case. Keep the larger error limits and the smaller cosine limit from these two implementations.

The selected project implementation uses FP32 FFN residuals, beta-weighted GDN inverse columns, and scaled BF16 GDN attention coefficients.
This calibration fixes the limits before operator migration. It does not change the reference arrays or operator numerical limits.

The vLLM calibration uses the default BF16 path, FP8 projections, FP32 recurrent state, and O2 compilation.
Its numerical adapter supplies disjoint cache blocks for each cache group.
For media MTP, it compiles the model with the stored three-axis positions.
This adapter checks the model computation.
It does not check the default one-axis V2 proposal buffer.

Keep the adapter, environment identities, input hashes, generated-code hashes, and full reports in external calibration artifacts.

These limits control the fixed cases. They do not prove accuracy for other inputs.
Do not adjust those limits automatically after an implementation change.
Keep inference vectors and reports external to the repository.
