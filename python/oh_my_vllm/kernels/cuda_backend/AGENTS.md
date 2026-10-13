# Native CUDA backend

Use owned CUDA C++ with optional PTX for B200. Native CUTLASS, CuTe, CUB, and CUDA libraries are permitted.
Python bindings may load repository kernels. Do not use a Python operator library as a provider.
Keep each `.cu`, `.cuh`, and `.py` backend file at most 800 physical lines.

Use pinned clang-format through the CUDA hook.

## Contracts

Keep caller streams, alignment, strides, graph replay, state writes, and numerical tolerances.
Fusion must keep observable BF16 rounding and state boundaries.
Bind source, headers, compiler, configuration, and loaded modules to evidence identities.

Keep dependency sources independent of Python operator wheels and development-tool updates.
Use independent Torch references for mathematical contracts. Measure whole operations and the framework before selection.
Keep formal cases and summarized evidence in the repository. Keep temporary tuning and raw traces in external storage.
