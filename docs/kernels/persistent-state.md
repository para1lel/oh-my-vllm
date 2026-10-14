# GDN state contract

[Python GDN](../../python/oh_my_vllm/kernels/gdn.py) owns tensor validation and GPU computation.
Rust selects committed read slots and separate speculative write slots. Rejected candidates cannot change a committed prefix state.
Keep FP32 persistent state and observable BF16 boundaries.

The [native scan](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_chunk.cuh) must keep storage, stream, and replay contracts.
The [state tests](../../tests/numerical/test_gdn_prefill_persistent.py) check cancellation, empty sequences, padded and misaligned rows, input bytes, and output guards.
Use [numerical requirements](../requirements/kernels.md) and the [acceptance index](../acceptance.md) for verification scope.
