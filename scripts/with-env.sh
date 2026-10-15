#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${OH_MY_VLLM_CONDA_PREFIX:-${CONDA_PREFIX:-${VIRTUAL_ENV:-$REPO_ROOT/.venv}}}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
	echo "Activate the development environment or set OH_MY_VLLM_CONDA_PREFIX." >&2
	exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export UV_PYTHON="$ENV_PREFIX/bin/python"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OH_MY_VLLM_WORKER_PYTHON="${OH_MY_VLLM_WORKER_PYTHON:-$ENV_PREFIX/bin/python}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

exec "$@"
