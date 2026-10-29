# DeepInfer comparison

Date: 2026-10-29

This note compares the DeepSeek adapter with DeepInfer. It also checks the
CUDA Graph design against current vLLM and SGLang patterns.

## Main findings

The largest current gap is execution shape. The adapter sends each decode
request to the target model as a separate call. DeepInfer sends a variable
length decode batch to one model step. This difference adds Python work,
kernel launch work, and repeated communication.

The adapter also used a 384 row default prefill chunk. DeepSeek native
operators accept up to 8192 rows. The default is now 8192. The environment
variable `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` can lower this value.

DSpark verification now sends the accepted prefix and all draft tokens in one
target call. It projects only the required logits. It stores accepted target
hidden states in one MTP seed operation. This removes one target forward call
per draft token.

These changes improve execution shape. They do not make the adapter a
varlen engine. The next major step is per-request position metadata.

## Scheduling

DeepInfer keeps MTP state in the scheduler. It records draft confidence,
accepted tokens, and schedule counters. Its `MTPVarlen` policy selects a
total query budget from a measured decode curve. It assigns extra query rows
to requests with the highest cumulative confidence.

The scheduler runs prompt and decode work in a C++ path. It can overlap CPU
post-processing with the next decode forward through zero-overhead scheduling.
Its CUDA path corrects query lengths on the device after the previous sample.

This repository schedules token IDs and cache ownership in Rust. The Python
adapter receives one synchronous execute message and returns one result.
Prefill rows with the same start and length are grouped. Decode rows are
otherwise executed one at a time. DSpark acceptance is performed on the CPU
after the target logits return.

The Rust scheduler already supports draft rows and cache transactions. It
does not yet provide the per-request query length and position arrays needed
by a DeepInfer-style varlen decode step.

Sources: [DeepInfer MTP scheduler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/mtp_varlen.hpp),
[DeepInfer worker overlap](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/worker.py),
[repository scheduler](../../crates/scheduler/src/lib.rs).

## Computation and operators

DeepInfer uses FlashMLA for attention and DeepGEMM for FP8 and FP4 matrix
operations. It uses DeepEP for expert communication and TileLang kernels for
metadata, cache, MTP, and sampling work. It stores quantized KV data in the
cache format used by attention kernels.

The adapter has native paths for routing, SwiGLU, FP8 and FP4 projection,
hyper-connections, sparse attention, and transforms. Several high-cost paths
still use PyTorch operations. They include low-rank projections, tensor
parallel embedding and head reductions, `einsum` output projection, and
expert output reduction.

The sparse native path packs the window and compressed KV tensors during each
call. DeepInfer stores quantized KV data in the cache and passes the cache
layout to FlashMLA. The adapter also uses TEP4 as the DEP4 fallback. TEP4
computes local routed experts and uses all-reduce. DeepInfer uses expert
parallel token exchange and expert load balancing.

The operator work should follow this order:

1. Add per-request varlen metadata and batch decode.
2. Keep quantized KV in the runtime cache format.
3. Replace repeated PyTorch projection and reduction paths with prepared native workspaces.
4. Replace the TEP4 fallback with stable expert owners and token exchange.
5. Add GPU-side MTP acceptance and confidence scheduling.

Sources: [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md),
[DeepInfer model runner](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/model_runner.py),
[DeepInfer attention backend](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44b6aafeaf945337/deepinfer/model_executor/layers/attention_backend.py).

## CUDA Graph review

The shared Qwen Graph path follows the important ownership rules. It uses a
separate capture stream and waits for caller and transfer streams. It uploads
the graph before replay and keeps buffer addresses fixed. It releases the
graph before its workspace owners. Its cleanup path keeps resources when CUDA
cleanup fails.

DeepInfer follows the same broad pattern. It captures padded decode buckets,
keeps reusable input and output buffers, uses a graph memory pool, and
captures DFlash graphs after the main graphs. It destroys graphs before
process group shutdown because collectives can keep graph dependencies alive.

vLLM makes graph keys and fallback selection explicit. SGLang checks graph
support for ragged verification, padding, speculative mode, and attention
parallel alignment before replay. These checks prevent a graph from running
with incompatible metadata.

DeepSeek should stay eager until its model call accepts static device metadata
for each request. The current model uses a scalar `start_pos` and scalar
sequence length assumptions. It also builds cache indices in Python. Capturing
the current loop would capture one request shape or replay wrong positions.
A correct DeepSeek Graph design has these parts:

- fixed capture buckets for total query rows.
- device buffers for query lengths, positions, cache slots, and block tables.
- a graph eligibility check for each rank and collective path.
- separate capture and replay buffers with stable addresses.
- an eager fallback for unsupported shape, cache, or collective states.
- explicit graph destruction before process group destruction.

Sources: [vLLM CUDA Graph design](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md),
[vLLM GPU model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py),
[SGLang decode Graph runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py),
[DeepInfer Graph runner](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/cuda_graph_runner.py).

## Long context

The DeepSeek AgentX command accepts `1048576` as the context limit. The
service descriptor also advertises the one million token model limit. Physical
GPU blocks and host cache capacity are separate runtime settings. A full
one million token trace depends on enough configured cache resources. The
current AgentX test checks that the command keeps the one million token limit
and concurrency 16.

## Decision

Keep the shared Qwen Graph implementation. Keep DeepSeek eager until varlen
metadata and rank-safe graph eligibility exist. Use the new batched DSpark
verification and 8192 row prefill defaults as the first performance changes.
Measure the complete service after each later varlen or operator change.
