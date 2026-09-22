# Design: oh-my-vllm

## Overview

oh-my-vllm splits inference work along a clear boundary: Rust owns scheduling
policy and KV cache bookkeeping; Python owns the GPU. The two sides share no
memory — they exchange msgpack-encoded messages over a ZMQ DEALER socket at each
inference step.

## ZMQ protocol

Both sides use `zmq.DEALER` / `DealerSocket`. The Rust process binds on
`ipc://<socket_path>`; the Python process connects. Because the topology is
always one Rust client to one Python worker, DEALER↔DEALER is equivalent to
a synchronous request-response channel.

All messages are msgpack dicts with a `"type"` key.

**Rust → Python:**

```
{"type": "init",
 "model_path": str,
 "num_gpu_blocks": int,
 "block_size": int,            # 784 for Qwen3.8
 "tensor_parallel_size": int,
 "max_model_len": int,
 "num_speculative_tokens": int}  # 0 = MTP disabled

{"type": "register",
 "request_id": int,
 "prompt_token_ids": list[int]}

{"type": "execute",
 "step_id": int,
 "scheduled": [
   {"request_id": int,
    "token_ids": list[int],
    "num_computed_tokens": int,
    "prefill_token_ids": [int],  // admission/resumption only: full accepted history
    "new_block_ids_to_zero": [int],  // fresh logical allocations; excludes cache hits
    "fa_block_table": list[int],
    "mamba_block_table": list[int]}
 ],
 "finished_request_ids": list[int],
 "preempted_request_ids": list[int],
 "num_batched_tokens": int}

{"type": "abort",  "request_id": int}
{"type": "shutdown"}
```

Cancel requests only between completed execution steps. `Scheduler::abort`
releases Rust ownership and queues a `finished_request_ids` notification for the
next execute, including when no requests remain scheduled. The driver must flush
that final notification. Python clears registrations, sampling, accepted-state and MTP progress. The legacy standalone `abort` message
uses the same finished-only path without a reply; it does not change Rust state.

**Python → Rust:**

```
{"type": "ready", "logical_num_blocks": int}

{"type": "execute_result",
 "outputs": [
   {"request_id": int,
    "token_ids": list[int],  # empty prefill, one normal token, or accepted MTP outputs
    "num_accepted_draft_tokens": int,  # MTP: accepted draft count
    "new_draft_token_ids": list[int]}  # actual next drafts from the project MTP proposer
 ]}

{"type": "error", "message": str}
```

## KV cache data structures

### Block layout

Qwen3.8-27B has two attention groups requiring separate block tables:

- **Group 0 (full attention):** 16 layers, standard block layout,
  `block_size=784` tokens per block. Prefix cache enabled.
- **Group 1 (GatedDeltaNet / Mamba):** 48 layers in `mamba_cache_mode="align"`.
  Running/checkpoint state, a temporarily protected previous state and K
  speculative state slots in MTP mode. Null placeholders preserve block positions.
  Cached checkpoints may outlive requests until the shared pool evicts them.

Both groups draw from a single shared `BlockPool`. This means a request that
holds many full-attention blocks and a Mamba checkpoint all compete for the same
logical pool. Python allocates separate per-layer tensors at logical capacity;
block IDs directly address their slots. The historical CLI capacity unit is kept:
ready exposes floor(num_gpu_blocks/3), matching the frozen baseline. There is no
vLLM physical-stride remapping; ADR002 describes that superseded implementation.

### BlockPool

`BlockPool` owns all block metadata as a flat `Vec<Block>` (indexed by block id)
and a hand-written doubly-linked LRU free queue (`FreeKVCacheBlockQueue`). When
the pool is constructed with `enable_caching=true`, freed blocks with a non-null
hash are retained as cached entries until evicted by a new allocation. This
matches vLLM's `KVCacheBlock` design.

### Prefix cache

Chain-hash: each block's hash is `SHA-256(parent_hash || token_ids || extra_keys)[0..8]`
truncated to 64 bits. Each hash depends on all preceding token content; it does not guarantee that
earlier blocks are still resident. The FA finder requires a contiguous resident
prefix, which is then reconciled with an available Mamba checkpoint.

The coordinator's `find_longest_cache_hit` returns the longest consistent prefix
hit across both groups. For the `is_simple_hybrid=True` case (Qwen3.8):
full-attention hit length is computed first, then Mamba hit is bounded by that
value, and the reconciled length is `min(fa_len, mb_len)`.

### Two-phase block allocation (vLLM issue #33775)

The `allocate_slots` method follows the exact ordering vLLM uses to prevent
double-counting blocks that move from the prefix cache into the free queue:

1. `remove_skipped_blocks` — free Mamba blocks no longer needed for the current
   step (the one two steps back in align mode).
2. `get_num_blocks_to_allocate` — count capacity with read-only pool access.
3. Check pool availability including watermark. Return `None` if insufficient —
   this is the preemption signal.
4. `add_local_computed_blocks` touches/dequeues all reused cached blocks before
   any group allocates new blocks.
5. `allocate_new_blocks` for each group.
6. `cache_blocks` — register newly-full blocks in the prefix cache.

## Scheduler

The scheduler lives in `crates/scheduler/src/lib.rs` and runs a two-phase loop
per step:

1. **Running queue** — walk existing requests, try to allocate slots for their
   next tokens. If `allocate_slots` returns `None`, preempt by recompute: free
   blocks, reset `num_computed_tokens=0`, move back to waiting queue front.
2. **Waiting queue** — admit new requests up to `max_num_seqs` and
   `max_num_batched_tokens`. Per-step token budget comes from
   `max_num_batched_tokens`; each request takes `min(remaining, budget)` tokens,
   followed by explicit Mamba-aligned prefill splitting so checkpoints are
   materialized at reusable boundaries.

Preemption is by recompute only (no CPU swap). The last-admitted running request
is the first evicted.

## MTP speculative decoding

With num_speculative_tokens=4, the independent Worker verifies scheduled drafts
against target samples and returns retained tokens plus explicit next-draft IDs.
Intermediate prefills return empty outputs; accepted lists are never collapsed.
Rust reserves recurrent-state slots, migrates committed sources after chunks, and
rolls back only scheduled rejected drafts. Prefix admissions and recompute resumes
carry complete accepted history; running updates append incremental suffixes.
Finished IDs are flushed even when no model tokens remain. BF16 MTP state retains
block784, matching the frozen baseline. There is no vLLM runner or draft handler.

MTP caches are indexed by their input token with first valid position 1; boundary
hidden features restore a prefix without mutating shared pages. When all three
further writes fit allocated pages, a proposal graph generates four greedy drafts
without intermediate host synchronization. Otherwise, stepwise generation trims
at allocation/context boundaries. Target verification and draft-head warmup share
KV reads across a request's queries, preserving independent causal masks.

Fresh logical allocation IDs remain a wire compatibility hint. The owned kernels
fully overwrite committed recurrent destinations, initialize new requests from
immutable zero state, and mask FA reads by valid lengths; they do not require the
old V2 physical-stride remapping or new_block_ids_to_zero call. Graph warmup and
capture save and restore every mutated FA/state destination before actual replay.

## Serving protocol extension (2026-09-21)

`prepare` carries request_id and a normalized request object (messages, tools,
effort/template options, output format, max_tokens, sampling, stop). Python validates,
compiles constraints and registers the request, replying `prepared` with
prompt_token_ids, or `error` before Rust admission. Legacy `register` remains
fixed-length greedy for Run/Bench.

Serving execute outputs optionally add text (incremental decoded text), finish_reason
(stop/length or null), and cumulative reasoning_tokens. Token IDs remain the
authoritative Rust scheduling/KV input. Grammar masks stay entirely in Python and
are applied by the owned target sampler, including speculative verification rows.
Rust detects worker process exit while awaiting replies and uses the existing
finished_request_ids cleanup path for EOS, length and cancellation.
