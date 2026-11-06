use oh_my_vllm_kv_cache::{
    group::{CacheRequest, RequestId},
    hash::BlockHash,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    Running,
    Preempted,
    Finalizing,
    Finalized,
    Retiring,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    pub ticket: Option<crate::session::RequestTicket>,
    pub cache_domain: oh_my_vllm_kv_cache::radix::CacheDomain,
    pub in_flight: Option<crate::execution::ExecutionKey>,
    pub(crate) submitted: bool,
    pub work: Option<crate::execution::WorkSpec>,
    pub accepted_end: Option<usize>,
    pub prompt_len: usize,
    pub token_ids: Vec<u32>,
    pub draft_token_ids: Vec<u32>,
    pub num_computed_tokens: usize,
    pub prefill_start: usize,
    pub num_in_flight_tokens: usize,
    pub max_tokens: usize,
    pub enable_mtp: bool,
    pub block_hashes: Vec<BlockHash>,
    pub cache_salt: Option<BlockHash>,
    pub cache_owner_rank: Option<u32>,
    pub record_partial_prefix: bool,
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
            ticket: None,
            cache_domain: Default::default(),
            in_flight: None,
            submitted: false,
            work: None,
            accepted_end: None,
            prompt_len,
            token_ids: prompt_token_ids,
            draft_token_ids: Vec::new(),
            num_computed_tokens: 0,
            prefill_start: 0,
            num_in_flight_tokens: 0,
            max_tokens,
            enable_mtp: false,
            block_hashes,
            cache_salt: None,
            cache_owner_rank: None,
            record_partial_prefix: true,
            status: RequestStatus::Waiting,
        }
    }

    pub fn content_salt(&self) -> Option<BlockHash> {
        let domain = &self.cache_domain;
        let fields = [
            &domain.model,
            &domain.template,
            &domain.layout,
            &domain.namespace,
        ];
        if fields.iter().all(|field| field.is_empty()) {
            return self.cache_salt;
        }
        let mut encoded = Vec::new();
        for field in fields {
            encoded.extend_from_slice(&(field.len() as u64).to_le_bytes());
            encoded.extend_from_slice(field.as_bytes());
        }
        let tokens = encoded.into_iter().map(u32::from).collect::<Vec<_>>();
        Some(oh_my_vllm_kv_cache::hash::hash_block_tokens(
            self.cache_salt,
            &tokens,
            &[],
        ))
    }

    pub fn num_generated_tokens(&self) -> usize {
        self.token_ids.len().saturating_sub(self.prompt_len)
    }

    pub fn is_finished(&self) -> bool {
        self.num_generated_tokens() >= self.max_tokens
    }

    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.draft_token_ids.clear();
    }

    pub fn set_drafts(&mut self, draft_ids: Vec<u32>) {
        self.draft_token_ids = draft_ids;
    }

    pub fn num_tokens_with_spec(&self) -> usize {
        self.token_ids.len() + self.draft_token_ids.len()
    }
}

impl CacheRequest for Request {
    fn request_id(&self) -> RequestId {
        self.id
    }

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

    fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    fn cache_salt(&self) -> Option<BlockHash> {
        self.content_salt()
    }

    fn record_partial_prefix(&self) -> bool {
        self.record_partial_prefix
    }

    fn is_waiting_or_preempted(&self) -> bool {
        matches!(
            self.status,
            RequestStatus::Waiting | RequestStatus::Preempted
        )
    }
}
