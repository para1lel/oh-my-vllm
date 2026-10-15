# Development tools

This subtree owns syntax parsers and other development-only helpers.
Keep dependency versions pinned and lock files tracked. Do not import these tools from inference code.
Keep compiler and parser dependencies separate from runtime packages.

The Rust spacing checker uses syntax spans for parsed definitions, attributes, comments, and nested methods.
The JavaScript checker uses Acorn and Node's TypeScript stripping support.
Do not replace syntax parsing with textual guesses.

Keep the native Rust formatter and hard 100-column limit active here.
Use the [contribution procedure](../CONTRIBUTING.md) for tests and delivery.
