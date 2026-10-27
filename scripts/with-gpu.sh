#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OH_MY_VLLM_GPU_EXCLUDE_INDICES="${OH_MY_VLLM_GPU_EXCLUDE_INDICES-0}"
exec python3 "$repo_root/development/serving/gpu.py" "$@"
