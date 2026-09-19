//! Request scheduling for oh-my-vllm.
//!
//! Port target: vLLM's `vllm/v1/core/sched/scheduler.py`. Not yet implemented —
//! the crate exists so the workspace layout is settled while [`oh_my_vllm_kv_cache`]
//! is built out underneath it.
//!
//! Shape this will take, per the design notes:
//!
//! - `running` / `waiting` queues, FCFS to start.
//! - A per-step token budget; chunked prefill falls out of
//!   `remaining.min(token_budget)` with no dedicated branch, exactly as in vLLM.
//! - Preemption by recompute (drop KV, requeue at the front) before swap.
//! - Speculative decoding accounting via `num_tokens_with_spec`, with rollback of
//!   `num_computed_tokens` when the worker rejects draft tokens.
