#!/usr/bin/env bash
# Start a command with the configured project environment.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${OH_MY_VLLM_CONDA_PREFIX:-${VIRTUAL_ENV:-${CONDA_PREFIX:-$REPO_ROOT/.venv}}}"

if [[ ! -d "$ENV_PREFIX/bin" ]]; then
	echo "with-env.sh: environment does not exist: $ENV_PREFIX" >&2
	echo "  set OH_MY_VLLM_CONDA_PREFIX, activate an environment, or create .venv" >&2
	exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
# uv otherwise prefers an inherited active environment over PATH.
export UV_PYTHON="$ENV_PREFIX/bin/python"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OH_MY_VLLM_WORKER_PYTHON="${OH_MY_VLLM_WORKER_PYTHON:-$ENV_PREFIX/bin/python}"
# Keep independently built kernels separate from other frameworks' artifacts.
RUNTIME_CACHE="${OH_MY_VLLM_RUNTIME_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-vllm/tilelang-ffi012}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$RUNTIME_CACHE}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_CACHE/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$RUNTIME_CACHE/tilelang}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$RUNTIME_CACHE/native-cuda}"
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-10.0a}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec "$@"
