# Work state

Rust supplies the HTTP APIs, CLI, scheduler, logical KV cache, and worker lifecycle controls.
The service modes are ordinary decode and MTP4.
HTTP input can contain text, images, and video.

The Python worker is a placeholder. It does not load model weights or use a GPU.
Rust sends media references through the inference worker's process and protocol boundary.

The [operator contracts](operators/model.md) have 18 internal stages and 7 serving boundaries.
The workload has six request profiles and one decode context per profile.

CUDA C++ and CUTLASS providers supply the seven boundaries through the [kernel development tools](operators/development.md).
The build tools are different from the inference loader.
Each provider supplies numerical checks and restores mutable state before measurement.
Reports bind checkpoint shards, source files, binaries, and build dependencies.
Vision MLP activation uses projection output stores.

The [numerical contract](operators/model.md#intermediate-precision) follows the pinned vLLM source types and rounding points.
The project has no vLLM runtime dependency.
Named tolerances apply to specific outputs. Cache checks and state selection keep their separate requirements.

The full operator implementation is not completed. More work is necessary for the 60% performance requirement.

Next work: optimize the measured bottlenecks.
Complete workload acceptance for each operator.
Then connect model computation to the worker.
Use the [implementation plan](operators/implementation.md).
Keep numerical checks and full-boundary measurements with each provider.

Use `main` for work.
Keep host paths and artifact locations in ignored local configuration.
Keep the [runtime contracts](requirements.md) and [process boundary](architecture.md).

Keep all English documents in ASD-STE100. Do not put test results in this document.
