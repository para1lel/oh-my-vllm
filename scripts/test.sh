#!/usr/bin/env bash
# One pytest entry point; GPU mode acquires and pins an idle B200.
set -euo pipefail

mode=${1:-}
if [[ "$mode" != cpu && "$mode" != full ]]; then
    echo "usage: scripts/test.sh {cpu|full} [pytest arguments...]" >&2
    exit 2
fi
shift
if [[ "$mode" == full && -z "${OH_MY_VLLM_MODEL:-}" ]]; then
    echo "set OH_MY_VLLM_MODEL before full GPU tests" >&2
    exit 2
fi

scripts/with-env.sh cargo build -p oh-my-vllm-zmq-worker --bin oh-my-vllm-zmq-worker
export OH_MY_VLLM_TEST_BINARY="${CARGO_TARGET_DIR:-$(git rev-parse --show-toplevel)/target}/debug/oh-my-vllm-zmq-worker"
if [[ "$mode" == cpu ]]; then
    exec env CUDA_VISIBLE_DEVICES='' scripts/with-env.sh env \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -m 'not gpu' "$@"
fi
exec scripts/with-gpu.sh scripts/with-env.sh env \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MAX_JOBS=8 python -m pytest tests -q "$@"
