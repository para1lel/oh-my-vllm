# Service guide

The Rust service uses a Python placeholder worker through the ZMQ/msgpack process boundary.
The placeholder does not load model weights or use a GPU.

## Start

Use the [development environment](development.md).
Install the runtime packages:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/runtime.txt
```

Build the executable:

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

Start the service:

```bash
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm.ipc serve
```

Use `--max-num-batched-tokens` to set the token budget per scheduler step. The default is `32768`.
Use `--max-num-seqs` to set the maximum number of running requests. The default is `4`.
Put these options before `serve`, `run`, or `bench`.

The scheduler selects decode requests first.
It divides the available token budget between prefill requests.
It adjusts prefill counts for the cache boundaries and supplies unused tokens to other prefill requests.

Use a different `--socket` path for each service process.
Use `serve --listen` to set the address. The default address is `127.0.0.1:8000`.

The service has no authentication. Keep host addresses in the ignored local configuration.
Use Ctrl-C or SIGTERM to stop the service and its worker.

## HTTP APIs

| Method and route | Function |
|---|---|
| `GET /v1/models` | Model list |
| `POST /v1/chat/completions` | Chat JSON or SSE, tools, reasoning, and usage |
| `POST /v1/responses` | Responses JSON or typed SSE, tools, reasoning, and history |
| `GET /v1/responses/{id}` | Stored response retrieval |
| `DELETE /v1/responses/{id}` | Stored response removal |
| `GET /diagnostics` | Service, resource, cache, and worker status |

Send inputs with the model ID `qwen3.8-27b-fp8`.
The service supplies one completion per request.

User messages can contain text, images, and video in input order.
Chat content parts use `text`, `image_url`, and `video_url`.
Responses content parts use `input_text`, `input_image`, and `input_video`.

Supply an HTTP URL, HTTPS URL, or base64 data URI for each image or video.
The data URI media type must be the same as the image or video type.
Image detail can be `auto`, `low`, or `high`.
Each request history can contain at most 16 media items.

The service sends media references to the worker.
The placeholder uses each reference's digest to calculate the prompt length.
It does not download or decode media.

The service returns validation errors for fields and input types that it does not accept.
Use `previous_response_id` for a stored Responses continuation.

A Responses response uses storage unless `store=false`.

For a sample request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"reasoning_effort":"off"}'
```

## Offline commands

```bash
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker run --tokens 1 2 3 --max-tokens 8
scripts/with-env.sh target/release/oh-my-vllm-zmq-worker bench --input-len 785 --output-len 8 --warmup 0 --repetitions 1
```

Run and Bench keep fixed output lengths.
The HTTP placeholder uses a random output length that is at most the output budget.

Use `--speculative-mode` with `none` or `mtp` for the simulated draft traffic.
The modes do not run model computation.

The placeholder supplies random text, sample tool calls, and simple JSON samples.
The placeholder has no model tokenizer, grammar masks, or GPU profiler.

Placeholder timing does not measure inference performance.
The [runtime contracts](requirements.md) still control the inference implementation.
