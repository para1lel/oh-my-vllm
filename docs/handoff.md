# Work state

Rust supplies the HTTP APIs, CLI, scheduler, logical KV cache, and worker lifecycle controls.
The service modes are ordinary decode and MTP4.
HTTP input can contain text, images, and video.

The Python worker is a placeholder. It does not load model weights or use a GPU.
Rust sends media references through the inference worker's process and protocol boundary.

The [operator contracts](operators/model.md) have 18 internal stages and 7 planned serving boundaries.
The workload has six request profiles and one decode context per profile.

A CUDA C++ and CUTLASS FFN baseline is available through the [kernel development tools](operators/development.md).
The build tools are different from the inference loader.
The other serving providers are not available.

The full operator implementation is not completed. More work is necessary for the 60% performance requirement.
Next work: optimize FFN, add the other providers, and connect model computation to the worker.
Keep numerical checks and full-boundary measurements with each provider.

Use [refactor and Issue branches](collaboration.md) for work.
Keep host paths and artifact locations in ignored local configuration.
Keep the [runtime contracts](requirements.md) and [process boundary](architecture.md).

Keep all English documents in ASD-STE100. Do not put test results in this document.
