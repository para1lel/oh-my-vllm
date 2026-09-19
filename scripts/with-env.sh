#!/usr/bin/env bash
# Run a command with the project's conda environment on PATH.
#
# Git hooks inherit whatever environment the committer happens to be in, which is
# usually not the `oh-my-vllm` conda env where cargo/rustfmt/clippy/ruff live.
# Every pre-commit hook goes through here so the toolchain is resolved the same
# way regardless of the caller's shell.
#
# Override the location with OH_MY_VLLM_CONDA_PREFIX.
set -euo pipefail

ENV_PREFIX="${OH_MY_VLLM_CONDA_PREFIX:-/data0/shared/dongwu.chen/conda-envs/oh-my-vllm}"

if [[ ! -d "$ENV_PREFIX/bin" ]]; then
	echo "with-env.sh: no conda env at $ENV_PREFIX" >&2
	echo "  set OH_MY_VLLM_CONDA_PREFIX to the env holding cargo and ruff" >&2
	exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
# Keep build artifacts inside the repo: the root disk has little room left.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(git rev-parse --show-toplevel)/target}"
export PYTHONPATH="$(git rev-parse --show-toplevel)/python${PYTHONPATH:+:$PYTHONPATH}"
export OH_MY_VLLM_WORKER_PYTHON="${OH_MY_VLLM_WORKER_PYTHON:-/data0/shared/dongwu.chen/conda-envs/vllm/bin/python}"

exec "$@"
