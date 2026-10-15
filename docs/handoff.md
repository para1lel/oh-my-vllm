# Work state

The Rust service has the HTTP APIs, CLI, scheduler, logical KV cache, and worker lifecycle controls.
The service modes are ordinary decode and MTP4.
The Python placeholder uses the same process and protocol boundary as the inference worker.

It does not load model weights or use a GPU.

Next step: add model computation to the Python worker.
Keep the [runtime contracts](requirements.md) and [process boundary](architecture.md).

Keep all English documents in ASD-STE100. Do not put test results in this document.
