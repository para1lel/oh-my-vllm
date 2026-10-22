# Test scope

Use the [contribution procedure](../CONTRIBUTING.md) before you add tests.
Agents do the tests for the maintainer.
Do the hooks in the [development commands](development.md#git-hooks).
Use the [nightly validation plan](deepseek-v41/validation.md) for model-specific scope and document-only changes.

The Rust tests include checks of the service, scheduler, logical KV, and worker lifecycle.
The HTTP tests use the Python placeholder through the process boundary.
They do not measure model inference or GPU performance.

CPU checks cover checkpoint and native identities, full-model report validation, canonical HTTP accounting, and manifest validation.
The constraint test checks schema masks and separate speculative grammar states.
Rust checks streamed JSON tool frames and prefix cache isolation for media requests.

The [execution tests](../tests/test_execution.py) check the execution decision for the full Execute message and graph resource release.
They check recording failures, replay on the caller's stream, interruption, and closure order.
They do not supply model numerical or performance evidence.

The [draft sampling check](../development/operators/check_draft_chain.py) compares CUDA Graph sampling with sequential sampling.
It checks probabilities, RNG offsets, and sampling history after each graph replay.
Run it through the GPU wrapper:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.check_draft_chain
```

Use the [full-model evaluation](operators/evaluation.md) for numerical checks.
Compare all vocabulary logits after a fixed short prompt and one decode iteration with an independent PyTorch FP64 model.
Do not use sampled intermediate outputs or the old low-precision path as the numerical reference.
Use the same reference artifact for subsequent changes.

Use the fixed MTP restoration, image/video, and long-context cases for changes to those paths.
Their repository fixtures contain complete PyTorch FP64 endpoint logits.
Usual tests do not run the reference generator.
The [development commands](operators/development.md#numerical-commands) generate and compare these artifacts.

Use the [operator microbench design](operators/microbench.md) for fixed operator numerical and performance cases.
Keep numerical fixtures and expected PyTorch outputs in the repository.
Normal numerical tests load these fixtures without reference generation.
Performance acceptance uses frozen theoretical-latency ratio limits.

Keep service contracts for state selection, constraints, media order, and request isolation.
Keep memory safety and graph resource checks separate from model numerical checks.
Use Compute Sanitizer for changed device memory or synchronization behavior.

Use the [HTTP benchmark](performance.md#measurement-command) for full inference timing.
It checks exact input and output lengths, uncached prefixes, and measured MTP acceptance.
Use the GPU wrapper for GPU work.
After each run, stop the processes that you started.
Keep measurement artifacts external to the repository.

Use the [AgentX procedure](agentx.md) for trace replay with prefix reuse.
Its approved CSV and figure can be stored in the repository.
Keep its raw records and logs external to the repository.
CPU checks cover its fixed configuration and request data validation.

Use the nightly validation plan for approved performance documents instead of new README result tables.
Do not put unrelated test results in project documents.
