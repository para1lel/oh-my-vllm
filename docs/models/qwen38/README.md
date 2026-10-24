# Qwen adapter

Qwen3.8-27B-FP8 runs on one NVIDIA B200.
The adapter implements the [shared service functions](../../requirements.md#service-functions) for text, images, and video.
Use ordinary decode or MTP4, with at most 262144 input and output tokens together.

The [Python adapter](../../../python/oh_my_vllm/models/qwen38) owns computation, weights, cache, codecs, and MTP4.
The [Rust policy](../../../crates/zmq-worker/src/models/qwen38/policy.rs) owns defaults and readiness validation.
Its checkpoint declares `Qwen3_5ForConditionalGeneration` and `qwen3_5`.
Keep those original identifiers.

Use block size 784, 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
The default scheduler has four request slots and a 32768-token step budget.
The default public model name is `qwen3.8-27b-fp8`.

## Guides

- [Model mathematics and precision](model.md)
- [Cache capacity, prefixes, and host offload](cache.md)
- [Full-model numerical acceptance](evaluation.md)
- [Operator cases](operators.md), [numerical limits](precision.md), and [performance accounting](operator-performance.md)
- [HTTP procedure](performance.md), [AgentX procedure](agentx.md), and [latest HTTP results](results.md)

Use the [shared service guide](../../serving.md) for commands and API fields.
Use `--speculative-mode mtp --num-speculative-tokens 4` before the subcommand for MTP4.
Use `--speculative-mode none` for ordinary decode.

## Media

The worker downloads and decodes media.
Each encoded item has a limit of 64 MiB.
An image has a decoded limit of 67108864 pixels.
Sampled video frames have a limit of 512 MiB.
Source height and width must be at most 65536 pixels.

Video sampling uses two frames per second, with at most 768 frames.
Each temporal pair has a timestamp in the prompt.
Media tokens use three-axis RoPE positions.
Media requests have separate prefix cache identities.

## Numerical commands

Do the CPU evidence and accounting checks:

```bash
scripts/with-env.sh python -m unittest development.model.test_identity development.model.test_logits development.serving.test_roofline
```

Comparisons use the frozen [FP64 reference](../../../development/model/reference.npz) in the repository.
Set `MODEL_ROOT` to the checkpoint directory.
Set `LOGITS_REPORT` to a new external JSON path.
Run the full-model comparison:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --report "$LOGITS_REPORT"
```

The command also saves inference logits in a `.npz` file with the same stem.
Use `--mode prefill` for the short prompt's last row.
The [evaluation procedure](evaluation.md) specifies the input, precision, metrics, and scope.

Generation is necessary only for an explicit reference change.
Install `requirements/model-reference.lock` in a different environment.
Use the [official PyTorch packages](https://pytorch.org/get-started/previous-versions/).
The inference worker does not use PyTorch.
Set `LOGITS_REFERENCE` to a new external `.npz` path.
Generate the file with the reference environment:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference --model "$MODEL_ROOT" --output "$LOGITS_REFERENCE"
```

The generator does not overwrite files.
Update the frozen file and its SHA256 only after review of the checkpoint, inputs, generator, and numerical policy.

## Operator commands

Set `OPERATOR_REPORT` to a new external directory.
Run all fixed numerical cases in the inference environment:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators numerics --model "$MODEL_ROOT" --report "$OPERATOR_REPORT"
```

This command loads repository fixtures and approved limits without PyTorch.
Missing fixtures, changed identities, or failed limits cause an error exit.

For an explicit fixture update, set `OPERATOR_FIXTURES` to a new external staging directory.
Generate fixed arrays in the reference environment:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.reference --model "$MODEL_ROOT" --output "$OPERATOR_FIXTURES"
```

Set `OPERATOR_LIMITS` to a new external JSON path.
Propose limits with the approved [precision policy](precision.md):

```bash
scripts/with-env.sh python -m development.operators.budgets --fixtures "$OPERATOR_FIXTURES" --output "$OPERATOR_LIMITS"
```

Examine arrays, manifests, generator identity, and proposed limits before updating repository fixtures, case hashes, and [limits](../../../development/operators/limits.json).

Set `OPERATOR_PERFORMANCE` to a new external directory.
Run all fixed performance cases with the approved ratio limits:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators performance --model "$MODEL_ROOT" --report "$OPERATOR_PERFORMANCE"
```

The [resource and timing policy](operator-performance.md) defines the measured operation and theoretical lower bound.
Failed or unresolved cases cause an error exit.

For initial calibration, set `OPERATOR_CALIBRATION` to a new external directory.
Collect five complete sessions:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators calibrate --model "$MODEL_ROOT" --report "$OPERATOR_CALIBRATION"
```

This command proposes limits in its external report.
It does not change repository limits.
Review the complete reports and provenance before copying approved performance entries into the limits file.

Use the [HTTP benchmark](performance.md#measurement-command) for complete inference timing.
Use the same `--prompt-seed` value when you compare measurements before and after a change.
The HTTP artifact records this seed and the execution policy.
