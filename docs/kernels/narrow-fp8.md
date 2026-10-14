# Narrow FP8 projection contract

[Projection implementation](../../python/oh_my_vllm/kernels/fp8.py) keeps checkpoint FP8 weights, FP32 block scales, and BF16 output.
[Native bindings](../../python/oh_my_vllm/kernels/cuda_backend/groupwise.py) validate shapes, strides, alignment, capacity, and caller streams.
Read scales on each call and changed-input graph replay. Keep partial buffers and output guards in their allocated storage.

Use [numerical requirements](../requirements/kernels.md), [candidate safety](cuda-safety.md), and [performance checks](../testing/performance.md).
The [acceptance index](../acceptance.md) identifies source-scoped verification records.
