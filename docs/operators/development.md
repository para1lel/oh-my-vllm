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

Install `requirements/model-reference.lock` in a separate development environment for the PyTorch FP64 reference.
Use the [official PyTorch packages](https://pytorch.org/get-started/previous-versions/).
The inference worker does not use PyTorch.
Set `MODEL_ROOT` to the checkpoint directory.
Set `LOGITS_REFERENCE` to an external `.npz` path.
Generate the PyTorch FP64 reference with that environment:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference --model "$MODEL_ROOT" --output "$LOGITS_REFERENCE"
```

The command rejects output paths that already contain a file.
Record its SHA256 in the comparison policy before numerical acceptance.
Set `LOGITS_REPORT` to a new external JSON path.
The comparison also saves the inference logits in a `.npz` file with the same stem.
Run the full model comparison with the inference development environment:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --reference "$LOGITS_REFERENCE" --report "$LOGITS_REPORT"
```

Use `--mode prefill` for the short prompt's last row.
The [evaluation procedure](evaluation.md) specifies the input, precision, metrics, and scope.
The command has no timing or efficiency gate.
Use the [HTTP benchmark](../performance.md#measurement-command) for performance acceptance.
Use Nsight Systems and Nsight Compute when a trace or counters can identify a bottleneck.
Keep their artifacts external to the repository.

Use the same `--prompt-seed` value when you compare HTTP measurements before and after a change.
The HTTP artifact records this seed and the execution policy.
