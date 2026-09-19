//! The data the scheduler hands off to the model worker each step.

use oh_my_vllm_kv_cache::group::RequestId;

// ── per-request slice ─────────────────────────────────────────────────────────

/// The worker's view of one scheduled request.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub request_id: RequestId,
    /// All token ids the model must process this step: prompt tokens that have
    /// not yet been computed, plus unverified MTP draft tokens if any.
    pub token_ids: Vec<u32>,
    /// How many tokens at the head of `token_ids` already have KV in the cache
    /// (i.e. can be skipped by the attention computation).
    pub num_computed_tokens: usize,
    /// FA block table for this request (positional block ids).
    pub fa_block_table: Vec<u32>,
    /// Mamba block table for this request (positional block ids, nulls included).
    pub mamba_block_table: Vec<u32>,
}

// ── step output ───────────────────────────────────────────────────────────────

/// Everything the model worker needs for one inference step.
#[derive(Debug, Default)]
pub struct SchedulerOutput {
    /// Requests to execute this step, in scheduling order.
    pub scheduled: Vec<ScheduledRequest>,
    /// Requests whose last output token was generated; the worker should not
    /// run them and the scheduler will remove them after this step.
    pub finished_request_ids: Vec<RequestId>,
    /// Requests that were preempted (evicted) this step and will be re-queued.
    pub preempted_request_ids: Vec<RequestId>,
    /// Total tokens the GPU will process across all scheduled requests.
    pub num_batched_tokens: usize,
}

// ── worker feedback ───────────────────────────────────────────────────────────

/// Token results the Python worker returns after one step.
#[derive(Debug)]
pub struct WorkerOutput {
    /// One entry per scheduled request, in the same order as
    /// `SchedulerOutput::scheduled`.
    pub outputs: Vec<RequestOutput>,
}

#[derive(Debug)]
pub struct RequestOutput {
    pub request_id: RequestId,
    /// Verified next token for this request (greedy / sampled).
    pub next_token_id: u32,
    /// How many of the speculative draft tokens were accepted.
    /// 0 means speculation was not used or all drafts were rejected.
    pub num_accepted_draft_tokens: usize,
    /// New MTP draft tokens for the next step; empty when MTP is off.
    pub new_draft_token_ids: Vec<u32>,
}
