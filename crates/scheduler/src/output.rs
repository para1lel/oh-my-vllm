use oh_my_vllm_kv_cache::group::RequestId;

#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub prefill_token_ids: Option<Vec<u32>>,
    pub num_computed_tokens: usize,
    pub fa_block_table: Vec<u32>,
    pub mamba_block_table: Vec<u32>,
    pub fa_block_copies: Vec<(u32, u32, u32)>,
    pub prefix_state_slot: Option<u32>,
    pub cache_checkpoint: Option<(usize, u32)>,
}

#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub scheduled: Vec<ScheduledRequest>,
    pub finished_request_ids: Vec<RequestId>,
    pub preempted_request_ids: Vec<RequestId>,
    pub num_batched_tokens: usize,
    pub cache_hit_tokens: usize,
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
