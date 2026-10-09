#!/usr/bin/env bash
# Wait for an idle B200; retain a cooperative lock until the command exits.
set -euo pipefail
exec {command_stdin}<&0
if (( $# == 0 )); then
    echo "usage: scripts/with-gpu.sh COMMAND [ARGS...]" >&2
    exit 2
fi
while true; do
    inventory=$(nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits)
    requested_present=false
    while IFS=, read -r uuid name memory utilization; do
        uuid=${uuid// /}
        [[ -z ${OH_MY_VLLM_GPU_UUID:-} || "$uuid" == "$OH_MY_VLLM_GPU_UUID" ]] || continue
        requested_present=true
        if [[ -n ${OH_MY_VLLM_GPU_UUID:-} && "$name" != *B200* ]]; then
            echo "requested GPU is not a B200" >&2
            exit 2
        fi
        [[ "$name" == *B200* ]] || continue
        [[ "$memory" =~ ^[[:space:]]*[0-9]+$ && "$utilization" =~ ^[[:space:]]*[0-9]+$ ]] || continue
        (( memory <= 64 && utilization == 0 )) || continue
        [[ "$processes" != *"$uuid"* ]] || continue
        exec {gpu_lock}>"/tmp/oh-my-vllm-${uuid}.lock"
        if flock -n "$gpu_lock"; then
            # Recheck after locking: another cooperating test may just have exited.
            processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits)
            if [[ "$processes" != *"$uuid"* ]]; then
                export CUDA_VISIBLE_DEVICES="$uuid"
                export OH_MY_VLLM_RUN_ID="${OH_MY_VLLM_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}"
                echo "$(date -u +%FT%TZ) run=$OH_MY_VLLM_RUN_ID gpu=$uuid" >&2
                # exec keeps the lock fd open in the command and its children.
                exec "$@" <&"$command_stdin"
            fi
        fi
        exec {gpu_lock}>&-
    done <<< "$inventory"
    if [[ -n ${OH_MY_VLLM_GPU_UUID:-} && "$requested_present" == false ]]; then
        echo "requested GPU UUID is not in the device inventory" >&2
        exit 2
    fi
    echo "$(date -u +%FT%TZ) waiting for an idle B200" >&2
    sleep 10
done
