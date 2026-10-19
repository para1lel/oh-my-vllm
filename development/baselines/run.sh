#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "usage: development/baselines/run.sh ENVIRONMENT CACHE COMMAND [ARGS...]" >&2
    exit 2
fi
baseline_environment=$(realpath "$1")
baseline_cache=$(realpath -m "$2")
shift 2
: "${CUDA_HOME:?Set CUDA_HOME from LOCAL.md}"
: "${OH_MY_VLLM_GPU_UUID:?Set the selected GPU UUID from LOCAL.md}"
export OH_MY_VLLM_CONDA_PREFIX="$baseline_environment"
export OH_MY_VLLM_WORKER_PYTHON="$baseline_environment/bin/python"
export OH_MY_VLLM_KERNEL_CACHE="$baseline_cache/own"
export FLASHINFER_WORKSPACE_BASE="$baseline_cache/flashinfer"
export TRITON_CACHE_DIR="$baseline_cache/triton"
export TILELANG_CACHE_DIR="$baseline_cache/tilelang"
export CUTE_DSL_CACHE_DIR="$baseline_cache/cute"
export CUTE_DSL_KEEP="ptx,cubin"
export CUTE_DSL_NO_CACHE="1"
export CUTE_DSL_DUMP_DIR="$baseline_cache/cute-dumps"
export OH_MY_VLLM_PUBLIC_CUBINS="$baseline_cache/public-cubins"
mkdir -p "$CUTE_DSL_DUMP_DIR" "$OH_MY_VLLM_PUBLIC_CUBINS"
export XDG_CACHE_HOME="$baseline_cache/xdg"
export CUDNN_FRONTEND_COMPILED_CACHE="$baseline_cache/cudnn"
export CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1"
export CUDA_CACHE_PATH="$baseline_cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$baseline_cache/torchinductor"
export CUDNN_FRONTEND_CUDART_LIB_NAME="$CUDA_HOME/lib64/libcudart.so"
export FLASHINFER_EXTRA_CFLAGS="-I$baseline_cache/headers/cutlass/include -I$baseline_cache/headers/cutlass/tools/util/include -I$baseline_cache/headers/spdlog/include"
export FLASHINFER_EXTRA_CUDAFLAGS="$FLASHINFER_EXTRA_CFLAGS"
export MAX_JOBS="${MAX_JOBS:-8}"
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-10.0a}"
if [[ "$TVM_FFI_CUDA_ARCH_LIST" != "10.0a" ]]; then
    echo "The native core requires TVM_FFI_CUDA_ARCH_LIST=10.0a." >&2
    exit 2
fi
for header in cutlass spdlog cccl; do
    if [[ ! -f "$baseline_cache/headers/$header/oh-my-vllm-public-baselines.json" ]]; then
        echo "Run development.baselines.bootstrap for this cache first." >&2
        exit 2
    fi
done
export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CFLAGS -I$baseline_cache/headers/cccl/cub -I$baseline_cache/headers/cccl/thrust -I$baseline_cache/headers/cccl/libcudacxx/include"
export FLASHINFER_EXTRA_CUDAFLAGS="$FLASHINFER_EXTRA_CFLAGS"
exec scripts/with-env.sh scripts/with-gpu.sh "$@"
