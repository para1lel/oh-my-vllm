use oh_my_vllm_kv_cache::group::RequestId;
use serde::{Deserialize, Serialize};

use crate::execution::{CacheStepResult, ExecutionKey, WorkSpec};
use crate::policy::qwen::QwenAssignment;

#[derive(Debug, Clone)]
pub struct ScheduledRequest<S = QwenAssignment> {
    pub request_id: RequestId,
    pub execution: ExecutionKey,
    pub work: WorkSpec,
    pub token_ids: Vec<u32>,
    pub prefill_token_ids: Option<Vec<u32>>,
    pub num_computed_tokens: usize,
    pub state: S,
    pub accepted_end: usize,
}

#[derive(Debug)]
pub struct SchedulerOutput<S = QwenAssignment> {
    pub scheduled: Vec<ScheduledRequest<S>>,
    pub finished_request_ids: Vec<RequestId>,
    pub preempted_request_ids: Vec<RequestId>,
    pub num_batched_tokens: usize,
    pub cache_hit_tokens: usize,
}

impl<S> Default for SchedulerOutput<S> {
    fn default() -> Self {
        Self {
            scheduled: Vec::new(),
            finished_request_ids: Vec::new(),
            preempted_request_ids: Vec::new(),
            num_batched_tokens: 0,
            cache_hit_tokens: 0,
        }
    }
}

#[derive(Debug)]
pub struct WorkerOutput {
    pub outputs: Vec<RequestOutput>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestOutput {
    pub request_id: RequestId,
    pub execution: ExecutionKey,
    pub work: WorkSpec,
    pub cache_result: CacheStepResult,
    pub token_ids: Vec<u32>,
    pub num_accepted_draft_tokens: usize,
    pub new_draft_token_ids: Vec<u32>,
}
