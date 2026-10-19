# FA and GDN performance plan

## Required behavior

Use one B200 for each measurement. Keep the runtime and numerical contracts.
Use measured peaks for that GPU UUID. Keep the workload weights and roofline formula unchanged.
Use the selected implementation's operand types for necessary work. Exclude temporary storage, padding, and repeated work.

Each of the 24 HTTP configurations must have efficiency of at least 50%.
Each FA and GDN shape of interest must be at least as fast as the fastest applicable public baseline.

The initial shape set is the complete modeled FA and GDN workload.
Include additional attention shapes from the HTTP execution trace.
Compare eager execution and CUDA Graph replay separately.
Use FlashInfer and QwenLM/FlashQLA with fixed versions and source identities.
Use public libraries only in an isolated development environment.
The inference worker must not import them.

## Comparison rules

Compare the same operation boundary, inputs, types, causal mask, scale, state, and necessary outputs.
Public attention cores and full serving boundaries are different measurements.
State layout conversion, checkpoint export, and other necessary adaptation must have explicit timing scope.

Do numerical checks before timing. Restore mutable state between measurements.
Keep the cache policy, warmup, repetitions, and timing method the same for each comparison.
An unsupported case or an invalid numerical comparison cannot count as a pass.

Put comparison summaries and the HTTP table in the two README files.
Give coverage counts, strict parity pass counts, median and worst ratios, and the worst shape.
Keep complete reports for each case external. State each unmet performance requirement.

## Implementation tasks

1. Prepare reproducible public baselines and focused Nsight traces.
   Record the source, binary, dependency, device, shape, and measurement identities externally.

2. Optimize GDN prefill with block computation and explicit state dependencies.
   Keep FP32 live state, BF16 checkpoint exports, private writes, and guard storage.
   Check a path that crosses a full computation block and a 784-token checkpoint.

3. Optimize FA prefill and the shared FP8 projection pipeline.
   Keep the specified operand types, scaling, output rounding, and persistent cache behavior.
   Use CPU tile models before GPU experiments.

4. Measure the complete service after integration.
   Use GPU timelines to select any further host, workspace, or launch changes.
   Run all 24 configurations with exact input and output lengths and uncached prefixes.

5. Run the applicable repository checks and the complete operator audit.
   Get an independent code and document review. Correct the errors.

   Update the two README files and the handoff. Commit and push on `main` with hooks active.
   Stop temporary services and remove temporary reports after review.

## Review focus

Check zero rows, short tails, nonzero incoming state, and variable sequence lengths.
Check the 784-token boundary and FP32 carry after BF16 checkpoint export.
Check numerical behavior when an algebraic transformation changes FP32 association.
Check baseline eligibility and complete shape coverage.
Check source stability during the final measurements.
