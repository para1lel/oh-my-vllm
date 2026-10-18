# Test scope

Use the [contribution procedure](../CONTRIBUTING.md) before you add tests.
Agents do the tests for the maintainer.
Do the hooks in the [development commands](development.md#git-hooks).

The Rust tests include checks of the service, scheduler, logical KV, and worker lifecycle.
The HTTP tests use the Python placeholder through the process boundary.
They supply no acceptance evidence for model inference or GPU performance.

The kernel contract tests include checks of the workload paths and the performance calculation.
The reference tests include checks of low-precision rounding, GDN carry state, and MTP short tails.
The constraint test checks schema masks and separate speculative grammar states.
Rust checks streamed JSON tool frames and prefix cache isolation for media requests.

Use the [kernel development commands](operators/development.md) for GPU numerical checks and measurements.
The FFN output reference uses selected rows. The report gives the sample scope.
Each provider states its numerical sample scope.

Sampling checks seeded selection and MTP state slot selection.
FA, GDN, and MTP check persistent writes and unchanged guard storage.
Vision checks patch order and frame isolation.
Incomplete coverage or failed checks prevent full operator acceptance.

Use the [HTTP benchmark](performance.md#measurement-command) for full inference timing.
It checks exact input and output lengths, uncached prefixes, and measured MTP acceptance.
Use the GPU wrapper for GPU work.
After each run, stop the processes that you started.
Keep raw artifacts external to the repository.

Update the required performance tables in the two README files before each delivery.
Do not put other test results in project documents.
