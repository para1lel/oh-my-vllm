# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Performance measurements

Use the [operator evaluation](docs/operators/evaluation.md) for numerical checks and full operator measurements.
Use the [HTTP comparison](docs/performance.md) for complete request measurements.
The two procedures are necessary for performance acceptance.
Keep measurement reports and source identities external to the repository.
