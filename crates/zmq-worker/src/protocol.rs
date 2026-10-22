use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RustMessage {
    Init(InitMsg),
    Register(RegisterMsg),
    Prepare {
        rpc_id: u64,
        request_id: u64,
        request: serde_json::Value,
    },
    Execute(ExecuteMsg),
    Abort(AbortMsg),
    Stats {
        rpc_id: u64,
    },
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitMsg {
    pub model_path: String,
    pub num_gpu_blocks: Option<u32>,
    pub mamba_blocks: Option<u32>,
    pub block_size: u32,
    pub tensor_parallel_size: u32,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub max_num_seqs: u32,
    pub num_speculative_tokens: usize,
    pub speculative_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_profile: Option<BenchmarkProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProfile {
    pub path: String,
    pub skip_steps: usize,
    pub steps: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterMsg {
    pub request_id: u64,
    pub prompt_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteMsg {
    pub rpc_id: u64,
    pub step_id: u64,
    pub scheduled: Vec<ScheduledRequestMsg>,
    pub finished_request_ids: Vec<u64>,
    pub preempted_request_ids: Vec<u64>,
    pub num_batched_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScheduledRequestMsg {
    pub request_id: u64,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_token_ids: Option<Vec<u32>>,
    pub num_computed_tokens: u32,
    pub fa_block_table: Vec<u32>,
    pub mamba_block_table: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fa_block_copies: Vec<(u32, u32, u32)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_state_slot: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_checkpoint: Option<(usize, u32)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AbortMsg {
    pub request_id: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerMessage {
    Ready {
        logical_num_blocks: u32,
        mamba_blocks: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_memory: Option<serde_json::Value>,
    },
    ExecuteResult(ExecuteResultMsg),
    Prepared {
        rpc_id: u64,
        prompt_token_ids: Vec<u32>,
    },
    Error(ErrorMsg),
    StatsResult {
        rpc_id: u64,
        stats: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteResultMsg {
    pub rpc_id: u64,
    pub outputs: Vec<RequestResultMsg>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestResultMsg {
    pub request_id: u64,
    #[serde(default)]
    pub error: Option<String>,
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub num_accepted_draft_tokens: u32,
    #[serde(default)]
    pub new_draft_token_ids: Vec<u32>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorMsg {
    pub message: String,
    #[serde(default)]
    pub rpc_id: u64,
    #[serde(default)]
    pub kind: WorkerErrorKind,
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerErrorKind {
    Validation,
    #[default]
    Internal,
    Oom,
    Device,
}

pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(msg)
}

pub fn decode(bytes: &[u8]) -> Result<WorkerMessage, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}
