# Development tools

Put syntax parsers and other development tools in this directory tree.
Pin the dependency versions. Track the lock files.

Do not import these tools from inference code.
Use different dependency files for development tools and runtime packages.

The Rust spacing checker uses syntax spans for definitions, attributes, comments, and nested methods.
The JavaScript checker uses Acorn and the TypeScript type removal function in Node.

Do not use text patterns to find code definitions.

Use the Rust formatter here. Keep the line width at most 100 columns.
Obey the [contribution procedure](../CONTRIBUTING.md) for tests and delivery.
