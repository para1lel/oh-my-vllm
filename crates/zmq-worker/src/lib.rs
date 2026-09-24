//! Rust side of the Rust ↔ Python ZMQ boundary.
//!
//! This crate launches the Python worker process, connects a ZMQ DEALER socket,
//! and provides [`WorkerClient`] — the async handle the scheduler calls to
//! execute one inference step.
//!
//! # Protocol
//!
//! All messages are msgpack-encoded dicts with a `"type"` key:
//!
//! Rust → Python:
//! - `{"type":"init", "model_path":str, "num_gpu_blocks":u32,
//!    "block_size":u32, "tensor_parallel_size":u32, "max_model_len":u32,
//!    "num_speculative_tokens":u32}`
//! - `{"type":"register", "request_id":u64, "prompt_token_ids":[u32]}`
//! - `{"type":"prepare", "request_id":u64, "request":{...}}`
//! - `{"type":"execute", "step_id":u64, "scheduled":[
//!      {"request_id":u64, "token_ids":[u32], "num_computed_tokens":u32,
//!       "fa_block_table":[u32], "mamba_block_table":[u32],
//!       "prefill_token_ids":[u32]|null, "new_block_ids_to_zero":[u32]}],
//!    "finished_request_ids":[u64], "preempted_request_ids":[u64],
//!    "num_batched_tokens":u32}`
//! - `{"type":"abort", "request_id":u64}`
//! - `{"type":"shutdown"}`
//!
//! Python → Rust:
//! - `{"type":"ready", "logical_num_blocks":u32}` (after init)
//! - `{"type":"prepared", "prompt_token_ids":[u32]}` (after prepare)
//! - `{"type":"execute_result", "outputs":[{"request_id":u64,
//!    "token_ids":[u32], "num_accepted_draft_tokens":u32,
//!    "new_draft_token_ids":[u32],
//!    "text":str|null, "finish_reason":str|null, "reasoning_tokens":u32}]}`
//! - `{"type":"error", "message":str}`

pub mod client;
pub mod error;
pub mod protocol;

pub use client::WorkerClient;
pub use error::{Error, Result};

pub mod serving;
