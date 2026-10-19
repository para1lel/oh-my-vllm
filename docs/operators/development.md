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

Use the [HTTP benchmark](../performance.md#measurement-command) for performance measurement.
Use the same `--prompt-seed` value when you compare measurements before and after a change.
The HTTP artifact records this seed and the execution policy.
