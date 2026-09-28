//! Per-request state tracked by the scheduler.

use oh_my_vllm_kv_cache::{
    group::{CacheRequest, RequestId},
    hash::BlockHash,
};

// ── status ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// In the waiting queue; no KV blocks allocated.
    Waiting,
    /// Being executed; KV blocks live in the pool.
    Running,
    /// Was running, evicted back to waiting; blocks freed.
    Preempted,
}

// ── request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    /// Length of the original prompt portion of `token_ids`.
    pub prompt_len: usize,
    /// Prompt tokens plus all accepted output tokens so far.
    ///
    /// Does NOT include unverified MTP draft tokens; those live in
    /// `draft_token_ids` and are concatenated by `num_tokens_with_spec()`.
    pub token_ids: Vec<u32>,
    /// Unverified MTP draft tokens appended by the speculative decoder.
    ///
    /// Cleared when the worker returns; replaced by `set_drafts()` before the
    /// next decode step if MTP is enabled.
    pub draft_token_ids: Vec<u32>,
    /// Tokens whose KV is resident on the GPU entering the current step.
    pub num_computed_tokens: usize,
    /// Tokens sent to the GPU in the most recent step that have not yet
    /// been committed (relevant for Mamba align-mode deferral).
    pub num_in_flight_tokens: usize,
    /// Maximum number of output tokens to generate.
    pub max_tokens: usize,
    /// Chain hashes for full blocks of `token_ids`: initialized at admission
    /// and extended when verified output fills another block.
    pub block_hashes: Vec<BlockHash>,
    pub status: RequestStatus,
}

impl Request {
    pub fn new(
        id: RequestId,
        prompt_token_ids: Vec<u32>,
        max_tokens: usize,
        block_hashes: Vec<BlockHash>,
    ) -> Self {
        let prompt_len = prompt_token_ids.len();
        Self {
            id,
            prompt_len,
            token_ids: prompt_token_ids,
            draft_token_ids: Vec::new(),
            num_computed_tokens: 0,
            num_in_flight_tokens: 0,
            max_tokens,
            block_hashes,
            status: RequestStatus::Waiting,
        }
    }

    pub fn num_generated_tokens(&self) -> usize {
        self.token_ids.len().saturating_sub(self.prompt_len)
    }

    /// True when the request has produced `max_tokens` output tokens.
    pub fn is_finished(&self) -> bool {
        self.num_generated_tokens() >= self.max_tokens
    }

    /// Append a verified output token and drop any pending draft tokens.
    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.draft_token_ids.clear();
    }

    /// Set new MTP draft tokens for the next forward pass.
    pub fn set_drafts(&mut self, draft_ids: Vec<u32>) {
        self.draft_token_ids = draft_ids;
    }

    /// Total tokens the model must process, including unverified MTP drafts.
    pub fn num_tokens_with_spec(&self) -> usize {
        self.token_ids.len() + self.draft_token_ids.len()
    }
}

// ── CacheRequest impl ─────────────────────────────────────────────────────────

impl CacheRequest for Request {
    fn request_id(&self) -> RequestId {
        self.id
    }
    /// Canonical token count (prompt + accepted output, no drafts).
    /// Used by `cache_blocks` to exclude unverified tokens from the prefix cache.
    fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }
    fn num_prompt_tokens(&self) -> usize {
        self.prompt_len
    }
    fn num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }
    fn num_in_flight_tokens(&self) -> usize {
        self.num_in_flight_tokens
    }
    fn block_hashes(&self) -> &[BlockHash] {
        &self.block_hashes
    }
    fn is_waiting_or_preempted(&self) -> bool {
        matches!(
            self.status,
            RequestStatus::Waiting | RequestStatus::Preempted
        )
    }
}
