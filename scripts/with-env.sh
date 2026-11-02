#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${OH_MY_VLLM_CONDA_PREFIX:-}"

if [[ -z "$ENV_PREFIX" ]]; then
	echo "Set OH_MY_VLLM_CONDA_PREFIX to the oh-my-vllm Conda environment." >&2
	exit 1
fi
if [[ "$(basename -- "$ENV_PREFIX")" != "oh-my-vllm" || ! -d "$ENV_PREFIX/conda-meta" ]]; then
	echo "OH_MY_VLLM_CONDA_PREFIX must name the oh-my-vllm Conda environment." >&2
	exit 1
fi
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "$CONDA_DEFAULT_ENV" != "oh-my-vllm" ]]; then
	echo "CONDA_DEFAULT_ENV must be oh-my-vllm for repository commands." >&2
	exit 1
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
	echo "The oh-my-vllm Conda environment has no executable Python." >&2
	exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
if [[ "$1" == "python" || "$1" == "$ENV_PREFIX/bin/python" ]]; then
	export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OH_MY_VLLM_WORKER_PYTHON="${OH_MY_VLLM_WORKER_PYTHON:-$ENV_PREFIX/bin/python}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

exec "$@"
