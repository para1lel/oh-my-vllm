use oh_my_vllm_kv_cache::group::RequestId;

use crate::policy::qwen::QwenState;

#[derive(Debug, Clone)]
pub struct ScheduledRequest<S = QwenState> {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub prefill_token_ids: Option<Vec<u32>>,
    pub num_computed_tokens: usize,
    pub state: S,
}

#[derive(Debug)]
pub struct SchedulerOutput<S = QwenState> {
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

#[derive(Debug)]
pub struct RequestOutput {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub num_accepted_draft_tokens: usize,
    pub new_draft_token_ids: Vec<u32>,
}
