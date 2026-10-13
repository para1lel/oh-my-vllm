# Architecture

Rust owns service, scheduling, and logical cache. Python owns GPU tensors and computation. Read only the topic needed for the change.

- [Runtime ownership](architecture/runtime.md)
- [Scheduling and cache concepts](architecture/scheduler.md)
- [Target model concepts](architecture/target.md)
- [Speculative decoding concepts](architecture/speculation.md)
- [Compilation and graph concepts](architecture/compilation.md)
- [Measurement and extension boundaries](architecture/extensions.md)
