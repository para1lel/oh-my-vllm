# Operator implementation plan

Use the [model contracts](model.md), [fusion boundaries](fusion.md), and [evaluation procedure](evaluation.md).
Keep the six request profiles. Do not change their weights during optimization.
Develop on `main`. Keep raw measurements external to this repository.

## Shared interfaces

Runtime modules use native FFI bindings and tensors allocated before execution.
Development providers supply `run()`, `reset()`, and `check(execute=True)`.
The benchmark includes all launches in each serving boundary.
Reset mutable state before numerical checks and each measurement.

1. Add shared BF16 projections and residual normalization.
   Use `cuda_backend/linear.cuh` and `normalization.cuh`.
   Track each native module's recursive source dependencies in `native.py` and the build tools.
   Check transitive cache invalidation and missing headers.

2. Optimize the full FFN in `cuda_backend/ffn.cu`.
   Compare small-row and prefill projection variants with the same CPU reference and cold-weight policy.
   Keep useful activation and quantization fusions.

3. Implement FA and GDN in `fa.py`, `gdn.py`, and their native modules.
   Check paged causal attention, MRoPE, private writes, recurrent carry, and 784-token checkpoints.
   Check all persistent writes and unchanged guard storage.

4. Implement vision in `vision.py` and its native modules.
   Include raw image preparation in the first block and spatial merge in the last block.
   Check patch order, frame isolation, normalization, activation, and output values.

5. Implement sampling in `sample.py` and MTP in `mtp.py`.
   Include vocabulary projection, penalties, masks, seeded selection, acceptance, rejection, and accepted state slots.
   Reuse FA and FFN for draft computation.
   Check ties, masked rows, accepted prefixes, rejection, and the bonus token.

6. Connect all providers to the evaluation CLI.

   Bind each native binary and checkpoint shard to each report.
   Measure all modeled cases in eager and graph modes in separate runs.
   Collect profiles after numerical checks. Optimize the measured bottlenecks.
   Measure each serving operator with the [performance calculation](evaluation.md#performance-calculation).

## Delivery

Use few behavior checks. Before you add a check, tell the user why it is necessary.

Keep the numerical sample scope explicit. Do not claim accuracy for unchecked elements.
Missing providers or failed checks prevent acceptance.

Get an independent review. Correct the errors. Update the handoff. Commit the change. Push each milestone.
