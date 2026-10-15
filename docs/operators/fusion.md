# Operator fusion plan

Use the [stage inventory](../../python/oh_my_vllm/kernels/specification.py) for the tensor contracts of the 18 internal semantic stages.
Use CUDA C++ and CUTLASS for this implementation.
The [runtime contracts](../requirements.md) also allow CuTe DSL.
CUTLASS, cuBLAS, CUB, and native CUDA libraries are permitted.

Use only project CUDA operators and the native CUDA libraries permitted here.
Do not use other external operator libraries, such as DeepGEMM or FlashInfer.
Python can allocate tensors and launch the CUDA implementations.

## Serving boundaries

Use the 7 serving boundaries in `SERVING_OPERATORS` in the [workload model](../../development/kernels/workload.py) for implementation and acceptance.
The internal stages are parts of these boundaries. They are not serving or acceptance boundaries.

| Serving operator | Full measured operation |
|---|---|
| `gdn_attention` | GDN input, recurrent or prefill computation, and output |
| `fa_attention` | FA input, attention, gate, and output projection |
| `ffn` | Gate/up projection, activation, and down projection |
| `sample` | Vocabulary projection and sampling. MTP acceptance, rejection sampling, and selection of accepted state slots if applicable |
| `mtp_context` | MTP input projection and FA input with the KV-only variant |
| `mtp_draft` | MTP input projection if applicable, one full FA layer, FFN, and sampling |
| `vision_block` | One vision layer. The first layer has patch input. The last layer has spatial merge |

The first MTP draft uses `query_only`. The other 3 drafts use `full`.
The first draft uses the context endpoint that is already available.

Native providers supply these boundaries. Full workload acceptance is necessary.
The implementation plan gives no evidence of the 60% performance gate.

## Internal language stages

| Internal stage | Computation |
|---|---|
| `gdn_input` | Embedding and media gather if applicable, residual RMSNorm, FP8 projection, BF16 gates, convolution, SiLU, Q/K norm, decay and beta |
| `gdn_recurrent` | Decay, delta update, output, private verification states |
| `gdn_prefill` | Chunk scan, output, aligned state checkpoints |
| `gdn_output` | Head RMSNorm, SiLU gate, activation quantization, output projection |
| `gqa_input` | Residual RMSNorm, Q/gate/K/V projection, head norm, MRoPE, paged KV writes |
| `gqa_attention_output` | Causal GQA, sigmoid gate, activation quantization, output projection |
| `ffn_gate_up` | Residual RMSNorm, activation quantization, packed gate/up projection |
| `ffn_down` | SiLU product, activation quantization, down projection |
| `output_sample` | Final norm, vocabulary projection, penalties, grammar, greedy or stochastic selection |
| `mtp_input` | Embedding gather, two norms, concatenation through the projection |
| `mtp_accept_commit` | Conditional acceptance, rejection sampling, accepted state selection |

MTP uses the same FA and FFN operators as the target model.
A second implementation of those operators is not necessary.

## Internal vision stages

| Internal stage | Computation |
|---|---|
| `vision_patch_input` | Resize, normalization, patch projection, bias, position interpolation, patch order |
| `vision_qkv` | Residual LayerNorm, Q/K/V projection, bias, spatial RoPE |
| `vision_attention_output` | Frame-local noncausal attention, output projection, bias |
| `vision_ffn_up` | Residual LayerNorm, first projection, bias, GELU |
| `vision_ffn_down` | Second projection and bias |
| `vision_merge_up` | Final residual LayerNorm, spatial gather, first projection, bias, exact GELU |
| `vision_merge_down` | Second projection and bias |

## Implementation selection

The stage tables give the necessary computation inside the serving boundaries.
An implementation can fuse stages. Each measured operation must supply all stages of its serving boundary.

If this reduces the time for the full operation, fuse producer work into GEMM input loads or output stores.
Use different small-row and prefill implementations when necessary.

Use small norm, activation, quantization, and position operations only in the full serving operators.
Keep their reference functions in the evaluation tools only.

A full operation can have more than one launch.
Include all launches, copies, reductions, and temporary storage in its measurement.

Compare a fusion with its full unfused chain with the same inputs and state.
Keep only fusions that are correct and reduce the time for the full operation.

Do not remove a launch from the measured operation to increase its efficiency.
Use [the evaluation procedure](evaluation.md) for acceptance.
