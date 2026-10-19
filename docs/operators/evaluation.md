# Full-model numerical evaluation

Use the [model contract](model.md) and [development commands](development.md).
The [GPU execution rules](../execution.md) control execution.

## Numerical scope

Use a fixed 16-token text prompt and one decode iteration.
The prompt is `The quick brown fox jumps over the lazy dog. Explain why the sky is blue`.
The [capture code](../../development/model/capture.py) specifies its token IDs.
The comparison includes all 248320 logits from embedding through all 64 layers and final vocabulary projection.
It does not compare intermediate operators or apply single-operator tolerances.

The [reference generator](../../development/model/reference.py) uses direct PyTorch FP64 computations.
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
These short text checks do not prove long-context, MTP, or vision numerical behavior.
Extend the full-model input and endpoint when those paths change.
Use different checks for protocols, state ownership, and numerical results.

## Evidence and limits

Keep the frozen [FP64 artifact](../../development/model/reference.npz) in the repository.
Record its SHA256, checkpoint identity, reference source identity, PyTorch version, and input token IDs.
Reject changed artifacts, checkpoint files, inputs, nonfinite logits, or incomplete vocabulary vectors.

Do not replace the reference with the inference output.
Do not overwrite artifacts during reference generation.

The report gives NRMSE, maximum absolute error, cosine similarity, and the two top-1 token IDs.
NRMSE is the L2 norm of the logit difference divided by the L2 norm of the FP64 logits.
Cosine similarity uses the full vectors without centering.
Top-1 equality is a diagnostic because small errors can change the order of almost equal logits.

The [comparison policy](../../development/model/compare.py) supplies the fixed full-model limits and artifact identity.
The limits are NRMSE at most 0.05, cosine similarity at least 0.999, and maximum absolute error at most 0.5.
All three limits must pass. A failed comparison exits with an error.

These limits control this short case. They do not prove accuracy for other inputs.
Do not adjust those limits automatically after an implementation change.
Keep inference vectors and reports external to the repository.
