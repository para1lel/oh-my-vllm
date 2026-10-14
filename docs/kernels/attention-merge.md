# Attention merge contract

The [owned attention source](../../python/oh_my_vllm/kernels/cuda_backend/operators/decode_attention.cuh) combines partial attention outputs with their weights.
Keep FP32 accumulation, BF16 boundaries, caller streams, and project tolerances. Empty heads produce exact zero.

The [merge tests](../../tests/numerical/test_attention_merge_replay.py) use an independent FP64 reference.
They check cancellation, dominant and empty splits, output guards, immutable inputs, and changed-input graph replay.
Use the [acceptance index](../acceptance.md) for source-scoped numerical reports and pending framework checks.
