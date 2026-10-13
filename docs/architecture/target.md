# Target model concepts

## Target computation and attention

The target combines full-attention layers with recurrent GDN layers.
Attention can read the applicable KV prefix. GDN carries history in a compact recurrent state.
Their different access patterns determine why the framework coordinates two cache groups.

[Model configuration and forward units](../../python/oh_my_vllm/models/qwen.py) define dimensions, weights, and layer order.
Do not copy their validation or implementation thresholds into this guide.

FP8 projections combine quantized values with explicit block scales and FP32 accumulation.
BF16 conversion points form part of the numerical contract. A fused chain must keep those points observable.
Read [projection semantics](../../python/oh_my_vllm/ir/fp8.py) and
[normalization/projection semantics](../../python/oh_my_vllm/ir/normalized_linear.py).

RoPE changes Q/K coordinates using token positions. Correct phase precision matters at long context.
Q/K preparation and cache writes must refer to the same logical positions.
[Attention semantics](../../python/oh_my_vllm/ir/attention.py) define the reference behavior.
[attention source](../../python/oh_my_vllm/kernels/attention.py) defines execution selection.

[GDN semantics](../../python/oh_my_vllm/ir/gdn_prefill.py) define recurrent updates and checkpoint outputs.
Causal convolution and state restoration must match accepted token history.
Read [GDN execution](../../python/oh_my_vllm/kernels/gdn.py) and
[convolution](../../python/oh_my_vllm/kernels/convolution.py) for implementation details.

Formal tests check outputs, written state, untouched storage, page boundaries, and graph replay.
[Accuracy requirements](../requirements/core.md#req-acc-001-numerical-accuracy) define the unchanged tolerances.
