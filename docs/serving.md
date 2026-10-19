# Service guide

The Rust service uses a Python inference worker through the ZMQ/msgpack process boundary.
The worker loads the local checkpoint. It uses the repository CUDA modules.

## Start

Use the [development environment](development.md).
Install the runtime packages:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/runtime.txt -r requirements/kernels.txt
```

Build the executable:

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

Build the native modules with the [kernel setup](operators/development.md#tools).
Set `MODEL_ROOT` to the checkpoint directory.
Start the service:

```bash
scripts/with-env.sh scripts/with-gpu.sh target/release/oh-my-vllm-zmq-worker --model "$MODEL_ROOT" --socket /tmp/oh-my-vllm.ipc serve
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

The worker downloads and decodes media. Each encoded item has a limit of 64 MiB.
An image has a decoded limit of 67108864 pixels. Sampled video frames have a limit of 512 MiB.
Source height and width must be at most 65536 pixels.
Video sampling uses 2 frames per second, with at most 768 frames.

Each temporal pair has a timestamp in the prompt.

Media tokens use three-axis RoPE positions. Media requests have separate prefix cache identities.

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
Add `--model "$MODEL_ROOT"` before the subcommand for model computation.

Use `--speculative-mode mtp --num-speculative-tokens 4` before the subcommand for MTP4.
Use `--speculative-mode none` for ordinary decode.

## Output controls

Use `min_tokens` to prevent EOS and stop strings before the minimum output length.
The minimum must be at most the output limit.
Use `temperature`, `top_k`, `top_p`, and the penalty fields for sampling.
Use `repetition_guard=false` to disable the generated-token cycle guard.

Use `response_format` for JSON output in Chat. Use `text.format` in Responses.
JSON Schema and enabled tools cannot be used together.

Tool choice can be `auto`, `none`, `required`, or a named function.
Set `parallel_tool_calls=false` for at most one call.
The worker applies grammar masks to target and proposal distributions.
Only accepted tokens change the request grammar state.

Decode-only batches use CUDA Graph execution. Other batches use ordinary execution.
The worker does not use PDL.
Use the [end-to-end procedure](performance.md) for inference performance measurements.

Temporary model buffers use a [CUDA memory pool](../python/oh_my_vllm/kernels/cuda_backend/scratch_allocator.cuh#L19)<!-- source: inline cudaMemPool_t scratch_pool(int device) { -->.
The worker prepares storage before it reports readiness.
The size uses the token budget and at most half the free GPU memory.
The pool keeps this storage for later work. The `gpu_reserved_bytes` statistic includes this storage.

## Placeholder

Use `--model stub` for service tests without a GPU.
The placeholder supplies random text, sample tool calls, and simple JSON samples.
It does not load the checkpoint or do model computation.
Placeholder timing does not measure inference performance.
A model initialization or execution error does not select the placeholder.
