# Test scope

Use the [contribution procedure](../CONTRIBUTING.md) before you add tests.
Agents do the tests for the maintainer.
Do the hooks in the [development commands](development.md#git-hooks).

The Rust tests include checks of the service, scheduler, logical KV, and worker lifecycle.
The HTTP tests use the Python placeholder through the process boundary.
They supply no acceptance evidence for model inference or GPU performance.

The kernel contract tests include checks of the workload paths and the performance calculation.
The reference tests include checks of low-precision rounding and GDN carry state.

Use the [kernel development commands](operators/development.md) for GPU numerical checks and measurements.
The FFN output reference uses selected rows. The report gives the sample scope.
Missing providers prevent full operator acceptance.

Use the GPU wrapper for GPU work.
After each run, stop the processes that you started.
Keep raw artifacts external to the repository.
Do not put test results in project documents.
