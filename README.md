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

## Measured performance

The current fifteen-row matrix uses ordinary decoding, MTP4, MTP4 prefix-hit, and DSpark.
Each group uses batches 1/2/4 and 4096 output tokens. New collection is pending.

Ordinary decoding and DSpark use 32768 input tokens.
MTP4 uses 32768, 131072, and a 32768 prefix-hit group with 32144 reused tokens.

Each phase uses two full warmups and five measurements.
The three-times lower-bound limit and 10% spread checks stay active, with the specified prefix-hit prefill ratio exception.
See [requirements](docs/requirements.md) for timing and [acceptance](docs/acceptance.md) for measured source scope.

## Long-duration agentic service

The current test uses four fixed oh-my-pi sessions per mode for at least one hour.
Two sessions use Chat and two use Responses. Session IDs and conversation histories stay the same.
There is no request-count minimum or separate warmup. First-request compilation enters the measurements.

OMP uses the `qwen3` tokenizer, context window `262144`, and runtime provider usage to select automatic compaction.
Task and summary calls use `max`, mapped to service `xhigh`. Model calls share a four-request concurrency limit.

The new persistent-session collection and resource review are pending.
The final figures show framework processing start and output tokens divided by processing wall time, with reasoning tokens.
The figures identify task and compaction purpose without API color legends.

This heavy test has no performance gate. Daily commands skip it.
See [testing](docs/testing.md#long-duration-agentic-service) for the procedure and memory review.

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
