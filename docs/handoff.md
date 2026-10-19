# Work state

Use `nightly` for work.
The [tile execution design](tile-execution.md) controls the GPU migration.
The [implementation plan](tile-migration.md) gives the next tasks.

The runtime still uses the native FFN, FA, GDN, vision, sampling, and MTP implementations.
Rust supplies HTTP, scheduling, logical KV, and worker lifecycle controls.
Python supplies model execution through the existing ZMQ DEALER and msgpack boundary.
No tile engine or new HTTP accounting implementation is available yet.

The final target covers all GPU tensor computation, also during preparation and initialization.
Migration stages have no performance threshold.
Final acceptance combines full task coverage, numerical and state correctness, and all 24 nominal-resource HTTP performance cases.

Start with independent HTTP accounting and extraction of shared build/reference tools.
Then implement the task engine and complete ordinary decode through vocabulary projection.
Continue with prefill, MTP/sampling, vision, and preparation/auxiliary computation.
Remove old implementations after their replacements pass the applicable correctness checks.

Keep the original model, state, and process contracts.
Keep host paths and artifact locations in ignored local configuration.
