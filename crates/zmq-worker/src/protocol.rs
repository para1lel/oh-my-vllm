pub use oh_my_vllm_model_contract::{MODEL_DESCRIPTOR_VERSION, ModelDescriptor};
use serde::{Deserialize, Serialize};

mod layout;
mod model_state;
pub use layout::*;
pub use model_state::*;

pub const EXECUTION_PROTOCOL_VERSION: u32 = 2;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RustMessage {
    Init(InitMsg),
    Register(RegisterMsg),
    Prepare {
        rpc_id: u64,
        request_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        owner_rank: Option<u32>,
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
#[serde(deny_unknown_fields)]
pub struct InitMsg {
    pub protocol_version: u32,
    pub worker_epoch: u64,
    pub descriptor_version: u32,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_rank: Option<u32>,
    pub prompt_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecuteMsg {
    pub rpc_id: u64,
    pub step_id: u64,
    pub worker_epoch: u64,
    pub scheduled: Vec<ScheduledRequestMsg>,
    pub finished_request_ids: Vec<u64>,
    pub preempted_request_ids: Vec<u64>,
    pub num_batched_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_plan: Option<Vec<RankSchedule>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankSchedule {
    pub rank: u32,
    pub request_ids: Vec<u64>,
    pub finished_request_ids: Vec<u64>,
    pub preempted_request_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankCompletion {
    pub rank: u32,
    pub step_id: u64,
    pub worker_epoch: u64,
    pub schedule: RankSchedule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Forward,
    HeadOnly,
    ExportOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Sampling {
    None,
    Target,
    Verify { drafts: u32 },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledRequestMsg {
    pub request_id: u64,
    pub lease_id: u64,
    pub purpose: oh_my_vllm_scheduler::execution::WorkPurpose,
    pub action: Action,
    pub sampling: Sampling,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_token_ids: Option<Vec<u32>>,
    pub forward_start: u32,
    pub forward_end: u32,
    pub accepted_end: u32,
    pub endpoint: u32,
    pub model_state: ModelStateWire,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AbortMsg {
    pub request_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpc_id: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "Ready carries the startup descriptor"
)]
pub enum WorkerMessage {
    Ready {
        protocol_version: u32,
        worker_epoch: u64,
        descriptor: ModelDescriptor,
        logical_num_blocks: u32,
        mamba_blocks: u32,
        cache_layout: CacheLayout,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_memory: Option<serde_json::Value>,
    },
    ExecuteResult(ExecuteResultMsg),
    Prepared {
        rpc_id: u64,
        prompt_token_ids: Vec<u32>,
    },
    AbortResult {
        rpc_id: u64,
        request_id: u64,
        ranks: Vec<u32>,
    },
    Error(ErrorMsg),
    StatsResult {
        rpc_id: u64,
        stats: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecuteResultMsg {
    pub rpc_id: u64,
    pub step_id: u64,
    pub worker_epoch: u64,
    pub outputs: Vec<RequestResultMsg>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rank_completions: Vec<RankCompletion>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestResultMsg {
    pub request_id: u64,
    pub lease_id: u64,
    pub purpose: oh_my_vllm_scheduler::execution::WorkPurpose,
    pub action: Action,
    pub error: Option<String>,
    pub token_ids: Vec<u32>,
    pub num_accepted_draft_tokens: u32,
    pub new_draft_token_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: Option<String>,
    pub reasoning_tokens: usize,
    pub computed_end: u32,
    pub accepted_end: u32,
    pub validated_exports: Vec<EndpointRef>,
    pub committed_page_valid_ends: Vec<PageValidEnd>,
    pub mtp_accepted_endpoint: Option<EndpointRef>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_requires_version_layout_and_descriptor() {
        let legacy = serde_json::json!({"type": "ready", "logical_num_blocks": 1608,
            "mamba_blocks": 128});
        assert!(decode(&encode(&legacy).unwrap()).is_err());
    }
}
