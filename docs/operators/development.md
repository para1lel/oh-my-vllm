# Kernel development

Use the [operator specification](model.md), [fusion plan](fusion.md), and [evaluation procedure](evaluation.md).
The FFN provider is a baseline. The other serving providers are not available.
The worker still uses its placeholder backend.

## Tools

Use CUDA 13.1, Nsight Systems, and Nsight Compute.
Keep their paths in the ignored `LOCAL.md` file.
Set `CUDA_HOME` to the CUDA installation directory.

Install the kernel development packages:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

The inference binding packages are in `requirements/kernels.txt`.
The development packages are not worker dependencies.

Set `CUTLASS_ROOT` to an external dependency directory.
Get the specified CUTLASS headers:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

The build accepts only Git revision `098de2a652cf8f00fd70b2df54051c7eccbb855a` from a clean dependency checkout.
Set `OH_MY_VLLM_KERNEL_CACHE` to an external build directory.

Build the native modules:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels build
```

The inference loader uses these modules. It does not use the compiler or development tools.
The loader rejects source or binary identities that do not agree with its manifest.

## Evaluation commands

Get the workload plan:

```bash
scripts/with-env.sh python -m development.kernels plan
```

Do the CPU contract checks:

```bash
scripts/with-env.sh python -m unittest development.kernels.test_contracts development.kernels.test_reference
```

Set `MODEL_ROOT` to the checkpoint directory.
Set `ARTIFACT_ROOT` to an external artifact directory.
Measure the FFN provider:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode graph --output "$ARTIFACT_ROOT/ffn-graph.json"
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode eager --output "$ARTIFACT_ROOT/ffn-eager.json"
```

Get a Systems trace and Compute counters for one shape:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels profile --model "$MODEL_ROOT" --operator ffn --rows 1 --mode graph --output "$ARTIFACT_ROOT/ffn-profile"
```

Before the command, select a profile directory path that does not exist.
Profiling does not supply acceptance timing.

Use `--rows` for a subset of the modeled rows.
A subset is not sufficient for the operator coverage gate.
The missing providers prevent acceptance of the full implementation.

The FFN check examines all output elements for finite values.
It compares all residual elements for exact BF16 equality.

The FP64 projection reference compares all columns in the first, middle, and final rows.
The report identifies this sample scope. It gives no tolerance evidence for the other output rows.

Keep raw reports and profiler files external to the repository.
Do not put measurement results in project documents.
