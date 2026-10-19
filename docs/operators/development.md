# Kernel development

Use the [model contract](model.md), [numerical checks](evaluation.md), and [tile migration plan](../tile-migration.md).
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

Do the CPU checks:

```bash
scripts/with-env.sh python -m unittest development.reference.test_contracts development.reference.test_reference development.serving.test_roofline
```

Set `MODEL_ROOT` to the checkpoint directory.
Run the small GPU references:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.reference --model "$MODEL_ROOT"
```

Use `--operator` to select `ffn`, `fa_attention`, `gdn_attention`, `vision_block`, `sample`, `mtp_context`, or `mtp_draft`.
The command checks eager execution and CUDA Graph replay.
It restores mutable state before each check and closes native workspaces after use.
It has no timing or efficiency gate.

Each provider states its numerical sample scope.
FFN checks all residual elements and samples output rows.
GDN checks private state writes. FA and MTP check KV writes and unchanged guard storage.
Sampling checks seeded selection and accepted state slots.
Vision checks raw patches, output samples, and frame isolation.

These checks do not prove accuracy for elements beyond their stated scope.

Use the [HTTP benchmark](../performance.md#measurement-command) for performance acceptance.
Use Nsight Systems and Nsight Compute when a trace or counters can identify a bottleneck.
Keep their artifacts external to the repository.
