# DeepSeek FP64 reference construction

## Scope

The [phase plan](plan.md) controls reference construction before native operator tuning.
The current tools supply checkpoint identities, decoded weights, Engram hashes, and small FP64 mathematical functions.
They do not supply full-model endpoint vectors or numerical acceptance at this time.
The [checkpoint guide](checkpoint.md) specifies identity and row-loading checks.

Keep the reference separate from inference code and native operators.
Use the pinned PyTorch environment from `requirements/model-reference.lock`.
The inference runtime must not import PyTorch or these development modules.

## Arithmetic

Decode original BF16, FP8, and packed FP4 checkpoint values directly to FP64.
Multiply their original scales in FP64.
Use FP64 for projections, activations, normalization, residuals, attention, and persistent mathematical state.
Do not add runtime activation quantization or BF16/FP32 intermediate rounding to this reference.

The [mathematical functions](../../development/model/deepseek_v41/reference_math.py) implement hyper-connections, adjacent-pair rotary transforms, sparse attention sinks, and routing.
Hyper-connection residual coefficients have source-copy and destination-copy axes, in that order.
RMS weights multiply the normalized values without a Qwen-style weight offset.
Finite Sinkhorn iteration includes the source epsilon at each specified normalization.
The attention sink contributes to the softmax denominator but has zero value.

Exact expert-score ties select lower global expert IDs first.
The local `torch.topk` call does not specify a stable tie order.
This explicit project rule prevents rank placement from changing tied expert selection.
Correction bias selects experts.
Unbiased scores determine routing weights.

The [source contract](model.md) and [state contract](state.md) specify the other model equations.
Operator and full-model limits must be fixed before tuning.
These mathematical functions do not set or relax those limits.

## Layer-streamed construction

Keep original packed weights in checkpoint files.
Load one projection or selected expert at a time through the checked weight reader.
Release temporary FP64 weight storage before the next projection.
Keep accepted activations and mathematical cache state separate from temporary weights.

Read embedding and Engram rows by their selected IDs.
Stream vocabulary-head rows in bounded pieces, then assemble all 129280 logits.
Do not substitute top-k logits for a full endpoint vector.
No full FP64 model copy is necessary.

Compute all 40 target layers for the short prefill reference.
Keep its FP64 state for the next decode token.
Use the highest prefill logit to select that fixed decode input.
Record the selected token with the two full endpoint vectors.
Draft, restoration, media, and long-context references are separate cases.

Bind each artifact to checkpoint, tokenizer, reference source, dependency versions, input tokens, and output array identities.
Reject existing output paths instead of overwriting reference artifacts.
Keep raw generation records external to the repository.
Add reviewed endpoint fixtures only after the generator and identities pass independent review.

## Mathematical checks

Set `REFERENCE_PYTHON` to the Python executable in the pinned reference environment.
The following small checks use CPU tensors only:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" -m unittest \
  development.model.deepseek_v41.test_reference_math
```

The checks use hand-calculated values for mixing direction, normalization, rotary pairs, attention sinks, and routing bias.
They are not full-model numerical evidence.
The normal CPU contract hook does not import PyTorch.
Use the [GPU execution rules](../execution.md) for later GPU reference generation.
