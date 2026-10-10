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

These fifteen workloads use source `706f1a6e7ce2b0188f07794cab574cbbf60f2927` on one B200 per workload.
Each request keeps 4096 output tokens. Each workload uses two full warmups and five measurements.
Prefix-hit requests reuse 32144 tokens. Other rows use cold prefixes.

Prefill covers request submission through the first token. Decode covers the first through the last token.
Each repetition uses the longest request interval for each phase.
Decode TPS is `4095 * batch / median(decode_wall_seconds)`.
The theoretical percentage is `100 * median(decode_bound_seconds) / median(decode_wall_seconds)`.
It gives the measured rate as a percentage of the theoretical maximum rate.

| Mode | Input | Batch | Prefill (s) | Prefill ratio | Decode TPS | Theoretical (%) | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| ordinary | 32768 | 1 | 1.265679 | 2.865 | 115.5 | 43.7 | pass |
| ordinary | 32768 | 2 | 2.524639 | 2.858 | 219.8 | 45.0 | pass |
| ordinary | 32768 | 4 | 5.180368 | 2.932 | 376.1 | 44.5 | pass |
| mtp4 | 32768 | 1 | 1.314763 | 2.892 | 392.8 | 44.9 | pass |
| mtp4 | 32768 | 2 | 2.625294 | 2.888 | 626.7 | 45.3 | pass |
| mtp4 | 32768 | 4 | 5.386242 | 2.962 | 997.5 | 43.0 | pass |
| prefix | 32768 | 1 | 0.038641 | 3.842 | 115.5 | 43.7 | pass |
| prefix | 32768 | 2 | 0.062813 | 3.123 | 219.8 | 45.0 | pass |
| prefix | 32768 | 4 | 0.122282 | 3.040 | 399.4 | 47.2 | pass |
| ordinary | 131072 | 1 | 7.111592 | 2.520 | 104.0 | 48.0 | pass |
| ordinary | 131072 | 2 | 14.215455 | 2.518 | 166.7 | 48.1 | pass |
| ordinary | 131072 | 4 | 28.724454 | 2.544 | 219.2 | 44.2 | pass |
| dspark | 32768 | 1 | 1.261098 | 2.820 | 174.4 | 38.8 | pass |
| dspark | 32768 | 2 | 2.524386 | 2.823 | 250.4 | 39.5 | pass |
| dspark | 32768 | 4 | 5.181812 | 2.897 | 429.1 | 41.4 | pass |


All active phase gates must pass. Prefix-hit prefill records its ratio without a three-times gate.
Its 10% spread check and decode gate stay active.
See [acceptance](docs/acceptance.md) for original hashes, failed attempts, independent review, and verification limits.

## Long-duration agentic service

These tests use four oh-my-pi agents and at most four model requests at the same time.
Each mode completed at least one hour and 500 normal requests. Warmup and intentional probes have separate records.
Request resources and late-period host/GPU memory checks passed, with no unexpected OOM or worker restart.

| Mode | Normal requests | Agent tasks | Normal duration (s) | Result |
|---|---:|---:|---:|---|
| MTP4 | 5135 | 2430 | 3675.338 | pass |
| DSpark | 3104 | 1461 | 3665.792 | pass |

Each point uses the framework processing start, when request preparation starts.
Its rate is output tokens divided by wall time from processing start to generation completion. Queue time is separate.
The full raw review checks SSE, SDK messages, tool results, and server records for each point.

### MTP4

![MTP4 agentic serving](bench/evidence/mtp-agentic-soak.svg)

### DSpark

![DSpark agentic serving](bench/evidence/dspark-agentic-soak.svg)

This heavy test has no performance gate. Daily commands skip it.
See [testing](docs/testing.md#long-duration-agentic-service) for the explicit command and memory checks.
See [acceptance](docs/acceptance.md) for full results, failed attempts, source scope, and original hashes.

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
