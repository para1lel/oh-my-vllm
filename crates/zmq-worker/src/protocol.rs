//! msgpack message types shared between Rust and Python.

use serde::{Deserialize, Serialize};

// ── Rust → Python ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RustMessage {
    Init(InitMsg),
    Register(RegisterMsg),
    Execute(ExecuteMsg),
    Abort(AbortMsg),
    Shutdown,
}

#[derive(Debug, Serialize)]
pub struct InitMsg {
    pub model_path: String,
    pub num_gpu_blocks: u32,
    pub block_size: u32,
    pub tensor_parallel_size: u32,
    pub max_model_len: u32,
    pub num_speculative_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct RegisterMsg {
    pub request_id: u64,
    pub prompt_token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct ExecuteMsg {
    pub step_id: u64,
    pub scheduled: Vec<ScheduledRequestMsg>,
    pub finished_request_ids: Vec<u64>,
    pub preempted_request_ids: Vec<u64>,
    pub num_batched_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ScheduledRequestMsg {
    pub request_id: u64,
    pub token_ids: Vec<u32>,
    pub num_computed_tokens: u32,
    pub fa_block_table: Vec<u32>,
    pub mamba_block_table: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct AbortMsg {
    pub request_id: u64,
}

// ── Python → Rust ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PythonMessage {
    Ready { logical_num_blocks: u32 },
    ExecuteResult(ExecuteResultMsg),
    Error(ErrorMsg),
}

#[derive(Debug, Deserialize)]
pub struct ExecuteResultMsg {
    pub outputs: Vec<RequestResultMsg>,
}

#[derive(Debug, Deserialize)]
pub struct RequestResultMsg {
    pub request_id: u64,
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub num_accepted_draft_tokens: u32,
    #[serde(default)]
    pub new_draft_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ErrorMsg {
    pub message: String,
}

// ── encode / decode helpers ───────────────────────────────────────────────────

pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(msg)
}

pub fn decode(bytes: &[u8]) -> Result<PythonMessage, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}
