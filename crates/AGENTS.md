# Rust code

Use Rust for the service, the scheduler, and the logical KV cache.
The Python process prepares tokens and supplies worker results.

Use the DEALER socket and msgpack protocol for process communication.

Validate the request and output before the state commit.
Publish validated prefixes only. Allocate related model state groups in one transaction.

Cancellation and worker replacement must release the related state.

Obey the runtime contracts for the API, CLI, stream events, and resource limits.
Use small functions and clear names. Put blank lines between functions and module definitions.

Keep the Rust line width at most 100 columns.
Put the safety information in a `// SAFETY:` comment before each unsafe block.

Before you add a test, tell the user why it is necessary.
Do the applicable tests and hooks. Do not put test results in project files.
