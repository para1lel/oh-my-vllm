# Rust ownership

Read [architecture](../docs/architecture.md) before scheduler, cache, or service changes.
Rust controls logical slots and accepted request history. GPU tensor ownership stays in Python.
Keep validation before state commit, coordinated FA/GDN allocation, and committed-only prefix publication.

Cancellation and worker replacement must release reservations and invalidate the correct worker epoch.

## Code and checks

Use Rust edition 2024, rustfmt, and Clippy. Keep all physical lines at most 100 characters, with macros and strings.
Each `unsafe` block must have a `// SAFETY:` explanation.
Keep concise comments for lifetime, concurrency, or numerical contracts that names cannot express.

Separate functions and definition blocks with blank lines. Keep imports and module declarations grouped.
Use critical state and failure tests. Normal service paths belong in integration tests.
Run workspace tests, format, line-width, and Clippy checks through the environment wrapper.
