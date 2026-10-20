# Kernel development

Use the [model contract](model.md), [numerical checks](evaluation.md), and [GPU execution rules](../execution.md).
The inference worker loads native modules. The placeholder applies only to `--model stub`.

## Tools

Use CUDA 13.1. Keep host paths in ignored `LOCAL.md`.
Set `CUDA_HOME` to the CUDA installation directory.
Install the kernel development packages:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

The inference bindings are in `requirements/kernels.txt`.
Development packages are not worker dependencies.
Set `CUTLASS_ROOT` to an external dependency directory.
Get the specified headers:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

The build accepts only clean Git revision `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
Set `OH_MY_VLLM_KERNEL_CACHE` to an external build directory.
Build the native modules:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build
```

The inference loader does not use development imports or a compiler.
It checks source and binary identities against the build manifest.
Each binary has an immutable copy and manifest in the external cache.

## Numerical commands

Do the CPU evidence and accounting checks:

```bash
scripts/with-env.sh python -m unittest development.model.test_identity development.model.test_logits development.serving.test_roofline
```

Comparisons use the frozen [FP64 reference](../../development/model/reference.npz) in the repository.
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

Examine arrays, manifests, generator identity, and proposed limits before updating repository fixtures, case hashes, and [limits](../../development/operators/limits.json).

Set `OPERATOR_PERFORMANCE` to a new external directory.
Run all fixed performance cases with the approved ratio limits:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators performance --model "$MODEL_ROOT" --report "$OPERATOR_PERFORMANCE"
```

The [resource and timing policy](performance.md) defines the measured operation and theoretical lower bound.
Failed or unresolved cases cause an error exit.

For initial calibration, set `OPERATOR_CALIBRATION` to a new external directory.
Collect five complete sessions:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators calibrate --model "$MODEL_ROOT" --report "$OPERATOR_CALIBRATION"
```

This command proposes limits in its external report.
It does not change repository limits.
Review the complete reports and provenance before copying approved performance entries into the limits file.

Use the [HTTP benchmark](../performance.md#measurement-command) for complete inference timing.
Use the same `--prompt-seed` value when you compare measurements before and after a change.
The HTTP artifact records this seed and the execution policy.
