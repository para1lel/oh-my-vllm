use oh_my_vllm_model_contract::ModelDescriptor;
use serde::{Deserialize, Serialize};

use super::collectives::CollectivePlan;

pub const GROUP_PROTOCOL_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankHeader {
    pub version: u32,
    pub epoch: u64,
    pub rank: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankReady {
    pub header: RankHeader,
    pub device_uuid: String,
    pub descriptor: ModelDescriptor,
    pub bootstrap_nonce: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankResult {
    pub header: RankHeader,
    pub step_id: u64,
    pub collective_plan_sha256: String,
    pub completed_collectives: u32,
    pub transaction_ids: Vec<u64>,
    pub device_work_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestWork {
    pub request_id: u64,
    pub transaction_id: u64,
    pub owner_rank: u32,
    pub rows: u32,
    pub prefill: bool,
}

#[derive(Debug, Clone)]
pub struct StepPlan {
    pub step_id: u64,
    pub collective_plan: CollectivePlan,
    pub requests: Vec<RequestWork>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Eager,
    Graph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankSchedule {
    pub rank: u32,
    pub rows: u32,
    pub requests: Vec<RequestWork>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepDispatch {
    pub version: u32,
    pub epoch: u64,
    pub step_id: u64,
    pub mode: ExecutionMode,
    pub collective_plan_sha256: String,
    pub collective_count: u32,
    pub collective_plan: CollectivePlan,
    pub ranks: Vec<RankSchedule>,
}
