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

## Cache memory

FA cache capacity has fixed defaults: `1824` blocks for ordinary decode and `1608` blocks for MTP4.
These defaults target 85% to 88% total GPU memory use during text inference on one B200 with the default scheduler settings.
The worker does not calculate the block count from available memory.

Use `--num-gpu-blocks` or `OH_MY_VLLM_NUM_GPU_BLOCKS` to change this capacity.
The service uses the command option if supplied.
Otherwise, it uses the environment variable if set. Otherwise, it uses the mode default.
Put the option before `serve`, `run`, or `bench`.

At startup, the worker checks the selected capacity against an 88% tensor budget.
This check includes memory already in use, FA and GDN storage, prefill, and stored decode and sampling workspaces.
The memory snapshot includes loaded weights. The check does not count them again.
The worker rejects a capacity that does not fit. It does not reduce the requested or default block count.

Additional media storage is reported separately. It does not reduce the default text cache capacity.
Media, CUDA Graph, and runtime allocations use the available memory.
Large media workloads can exceed available memory. Use a lower block count in this case.
The startup check does not set a strict limit on total process memory.

The [host cache](kv-offload.md) keeps interrupted requests and reusable prefixes in pinned memory.
Request restoration keeps recurrent state, media, pending drafts, and sampling offsets.
Other resident requests can execute during cache transfers.

Use `--host-cache-bytes` to set a host storage limit in bytes.
The default limit is calculated at runtime. Physical host storage grows only at safe idle points.
Use a zero limit to disable host offload for comparisons.

The default logical GDN capacity is `128` slots.
Private request states and temporary checkpoints use additional physical slots.
Use `--mamba-blocks` to change the logical GDN capacity.
The worker reports resolved capacities and memory calculations in Ready, the service log, and the execution trace identity.
Rust creates its logical pools and scheduler after Ready.
Replacement workers must supply the same capacities and satisfy the current memory budget.

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

## Prefix cache

Requests can use cached full blocks and prefixes with a final block that is not full.
For a prompt with a final block that is not full, the scheduler can record a checkpoint before the last token.
The checkpoint contains FA, GDN, convolution, and MTP hidden state for the same position.
The scheduler makes a checkpoint available to other requests only if its checks of the worker output find no errors.

For a media request, the scheduler records only full block checkpoints.

A request with the same prompt can continue from this checkpoint and do computation for its last token again.
A longer prompt can continue from the same recorded position.
The cache cannot continue from a position that has no GDN checkpoint.

Before computation continues, the worker makes a copy of the shared FA tail in a private block.
The copy contains K and V from the target and MTP layers.
GDN reads the checkpoint and writes to a different slot.
The scheduler keeps the source blocks until its checks of the worker result find no errors.

Without sufficient capacity for these private blocks, the cache uses full blocks.

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
Use `ignore_eos=true` to count EOS tokens without stopping generation.
This option does not mask EOS or disable stop strings.
Use `temperature`, `top_k`, `top_p`, and the penalty fields for sampling.
Use `repetition_guard=false` to disable the generated-token cycle guard.

Use `response_format` for JSON output in Chat. Use `text.format` in Responses.
JSON Schema and enabled tools cannot be used together.

Tool choice can be `auto`, `none`, `required`, or a named function.
Set `parallel_tool_calls=false` for at most one call.
The worker applies grammar masks to target and proposal distributions.
Only accepted tokens change the request grammar state.

The [execution rules](execution.md) select Eager for Execute messages with prefill and CUDA Graph replay for decode-only messages.
Selected FFN and GDN kernels use the dependent launches specified in those rules.
Use the [end-to-end procedure](performance.md) for inference performance measurements.

Temporary model buffers use a [CUDA memory pool](../python/oh_my_vllm/operators/runtime/cuda/scratch_allocator.cuh#L19)<!-- source: inline cudaMemPool_t scratch_pool(int device) { -->.
The worker prepares storage before it reports readiness.
The size uses the token budget and at most half the free GPU memory.
The pool keeps this storage for later work. The `gpu_reserved_bytes` statistic includes this storage.

## Placeholder

Use `--model stub` for service tests without a GPU.
The placeholder supplies random text, sample tool calls, and simple JSON samples.
It does not load the checkpoint or do model computation.
Placeholder timing does not measure inference performance.
A model initialization or execution error does not select the placeholder.
