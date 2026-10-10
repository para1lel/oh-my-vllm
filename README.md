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

These fifteen workloads use source `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb` on one B200 per workload.
Each request keeps 4096 output tokens. Each workload uses two full warmups and five measurements.
Prefix-hit requests reuse 32144 tokens. Other rows use cold prefixes.

Prefill covers request submission through the first token. Decode covers the first through the last token.
Each repetition uses the longest request interval for each phase.
Decode TPS is `4095 * batch / median(decode_wall_seconds)`.
The theoretical percentage is `100 * median(decode_bound_seconds) / median(decode_wall_seconds)`.
It gives the measured rate as a percentage of the theoretical maximum rate.

| Mode | Input | Batch | Prefill (s) | Prefill ratio | Decode TPS | Theoretical (%) | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| ordinary | 32768 | 1 | 1.268106 | 2.871 | 115.1 | 43.5 | pass |
| ordinary | 32768 | 2 | 2.555269 | 2.892 | 219.4 | 45.0 | pass |
| ordinary | 32768 | 4 | 5.224045 | 2.956 | 375.2 | 44.4 | pass |
| mtp4 | 32768 | 1 | 1.310482 | 2.883 | 395.2 | 45.2 | pass |
| mtp4 | 32768 | 2 | 2.615408 | 2.877 | 630.4 | 45.5 | pass |
| mtp4 | 32768 | 4 | 5.328181 | 2.930 | 1010.4 | 43.5 | pass |
| prefix | 32768 | 1 | 0.037446 | 3.723 | 115.5 | 43.7 | pass |
| prefix | 32768 | 2 | 0.061593 | 3.062 | 219.7 | 45.0 | pass |
| prefix | 32768 | 4 | 0.120962 | 3.007 | 399.7 | 47.3 | pass |
| ordinary | 131072 | 1 | 7.059579 | 2.501 | 104.1 | 48.1 | pass |
| ordinary | 131072 | 2 | 14.201501 | 2.516 | 166.5 | 48.0 | pass |
| ordinary | 131072 | 4 | 28.825513 | 2.553 | 219.2 | 44.2 | pass |
| dspark | 32768 | 1 | 1.279684 | 2.862 | 174.4 | 38.8 | pass |
| dspark | 32768 | 2 | 2.540217 | 2.841 | 246.0 | 38.8 | pass |
| dspark | 32768 | 4 | 5.292443 | 2.959 | 430.4 | 41.5 | pass |


All active phase gates must pass. Prefix-hit prefill records its ratio without a three-times gate.
Its 10% spread check and decode gate stay active.
See [acceptance](docs/acceptance.md) for original hashes, failed attempts, independent review, and verification limits.

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
