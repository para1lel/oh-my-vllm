# Project principles

Build a common inference framework with model adapters.
Keep Qwen and DeepSeek as peer adapters. Put model-specific rules in their [model directories](docs/models).
Use Rust for the service, the scheduler, and the logical KV cache.

Use Python for GPU computation. Use a ZMQ DEALER socket and msgpack for process communication.
Obey the [runtime contracts](docs/requirements.md).

Read the Git status, the [handoff](docs/handoff.md), and the applicable directory instructions before work.
Obey [CONTRIBUTING.md](CONTRIBUTING.md).
Codex uses `nightly`.
Use the [GPU execution rules](docs/execution.md) for scheduling and resource ownership.

Keep the root guides short. Put local instructions in the related directory.
Write all English documents in ASD-STE100. Supply a full Chinese translation for each English document.

Use clear names and small functions. Add a comment only when the code cannot give the necessary information.
Keep the tests few. Before you add a test, tell the user why it is necessary.

Do the applicable tests in the repository. Use the [reporting policy](docs/testing.md#reporting) for performance documents.
Do not put unrelated test results in project files.
Use the [test scope](docs/testing.md).

Get a code review from another agent before each delivery. Correct the errors.
Keep the hooks active. Commit only the files for the change.

Push the working branch to GitHub after each delivery.
Before work in other repositories or other publication, get instructions from the maintainer.

Put host settings in the ignored `LOCAL.md` file. Stop the workers and temporary services that you start.
