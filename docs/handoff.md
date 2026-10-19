# Work state

Use `main` for work.
The worker uses [operator scheduling](execution.md): Eager for Execute messages with prefill, and CUDA Graph replay for decode-only messages.
The native builder is `development.native_build`.
FA keeps BF16 cache state and derived FP8 operands with the same logical page ownership.
See the [model contract](operators/model.md#intermediate-precision) for attention precision.

The [numerical tools](operators/evaluation.md) compare full-model logits with the frozen [PyTorch FP64 reference](../development/model/reference.npz).
Keep its identity checks and numerical limits.
Use the 24 [HTTP configurations](performance.md) to measure time and nominal-resource efficiency.
Keep measurement artifacts external to the repository.

Limit further optimization to changes with a clear performance benefit.
Keep numerical precision, state ownership, and service behavior.
Extend full-model inputs when changes affect MTP, vision, or long contexts.
Keep host settings in ignored `LOCAL.md`.
