# oh-my-vllm

A Rust inference framework for Qwen3.8-27B-FP8 on one B200 GPU.
Rust controls HTTP service, request scheduling, and logical KV cache.
Python controls model computation through third-party GPU libraries and project kernels.
The processes exchange msgpack messages through ZMQ DEALER sockets.

The target has 48 GDN layers and 16 FA layers.
The runtime uses 784-token blocks and has a 262144-token maximum for total input and output.
It supplies ordinary decoding, MTP4, DSpark, prefix reuse, recompute preemption, and constrained generation.
The HTTP service supplies Chat Completions and Responses APIs, with streams and function tools.

## Start

Install the environment and set the model location as specified in [development](docs/development.md).
Use an absolute checkpoint location that exists on your server:

```bash
export OH_MY_VLLM_MODEL="/path/to/Qwen3.8-27B-FP8"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker serve
```

The service default is `127.0.0.1:8000`.
See [service contracts](docs/serving.md) for limits, tools, and request examples.
See [tests](docs/testing.md) for correctness and performance procedures.

DSpark uses a different draft checkpoint and the same target model:

```bash
export OH_MY_VLLM_DRAFT_MODEL="/path/to/Qwen3.8-27B-DSpark"
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --speculative-mode dspark serve
```

See [architecture](docs/architecture.md) for the draft algorithm and target verification.

## Read

- [Requirements](docs/requirements.md): scope and acceptance criteria.
- [Architecture](docs/architecture.md): ownership, data flow, and implementation decisions.
- [Acceptance](docs/acceptance.md): measured results and source identities.
- [Open work](docs/handoff.md): current status and next tasks.
- [Document index](docs/README.md): guides and decision records.
- [Interactive tutorial](code-journey/README.md): thirteen Chinese chapters with project source and scheduler experiments.
- [Contribution rules](CONTRIBUTING.md): checks and commit procedure.

The [acceptance index](docs/acceptance.md) identifies each measurement's source and test scope.
Framework performance uses fifteen prefill/decode rows and theoretical resource limits.
Project builds, tests, and inference use no vLLM implementation.
Formal verification uses full original records.

Keep server-specific setup in ignored `LOCAL.md`.

[中文](README.zh.md)
