mod allocation;
mod completion;
mod lookup;

use std::collections::{BTreeMap, HashMap};

pub use oh_my_vllm_kv_cache::radix::{
    CacheDomain, CheckpointExport, CheckpointId, CheckpointRead, EndpointSource, ExportRole,
    KvSpan, KvSpanMode, NodeId, PageAccess, PageBinding, PageCopy, PageId, PrefixEvidence, SlotKey,
};
use oh_my_vllm_kv_cache::{
    group::RequestId,
    radix::{SlotPool, TokenRadix},
};

use crate::session::{AdmissionGeneration, SessionId};

#[derive(Debug, Clone)]
pub struct QwenAssignment {
    pub evidence: PrefixEvidence,
    pub resume: CheckpointRead,
    pub page_table: Vec<PageBinding>,
    pub kv_spans: Vec<KvSpan>,
    pub page_copies: Vec<PageCopy>,
    pub exports: Vec<CheckpointExport>,
    pub endpoint_source: EndpointSource,
    pub private_checkpoints: Vec<CheckpointId>,
    pub mtp_endpoint: Option<(usize, CheckpointId)>,
}

impl Default for QwenAssignment {
    fn default() -> Self {
        Self {
            evidence: Default::default(),
            resume: CheckpointRead::Zero,
            page_table: vec![],
            kv_spans: vec![],
            page_copies: vec![],
            exports: vec![],
            endpoint_source: EndpointSource::Forward,
            private_checkpoints: vec![],
            mtp_endpoint: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct QwenHit {
    pub evidence: PrefixEvidence,
    pub resume: CheckpointRead,
    pub pages: Vec<PageBinding>,
    pub mtp_endpoint: Option<(usize, CheckpointId)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenResources {
    pub usable_checkpoints: usize,
    pub available_checkpoints: usize,
    pub cached_checkpoints: usize,
    pub usable_pages: usize,
    pub available_pages: usize,
    pub cached_pages: usize,
    pub private_checkpoints: usize,
    pub available_private_checkpoints: usize,
    pub scratch_checkpoints: usize,
}

#[derive(Clone, Default)]
struct Active {
    assignment: QwenAssignment,
    pages: BTreeMap<usize, PageBinding>,
    checkpoints: Vec<CheckpointId>,
    copy_sources: Vec<PageId>,
    private: Vec<CheckpointId>,
    endpoints: BTreeMap<usize, CheckpointId>,
}

struct SuccessfulTail {
    domain: CacheDomain,
    node: NodeId,
    generation: AdmissionGeneration,
}

pub struct QwenResidentCache {
    tree: TokenRadix,
    checkpoints: SlotPool,
    pages: SlotPool,
    private: SlotPool,
    active: HashMap<RequestId, Active>,
    tails: HashMap<SessionId, SuccessfulTail>,
    mtp_contexts: HashMap<CheckpointId, usize>,
    usable: u32,
    max_seqs: usize,
    max_context: usize,
    scratch: usize,
    batch_tokens: usize,
}

impl QwenResidentCache {
    pub fn new(
        usable_checkpoints: u32,
        max_num_seqs: usize,
        max_context: usize,
        max_num_batched_tokens: usize,
    ) -> Self {
        assert!(usable_checkpoints > 0 && max_num_seqs > 0 && max_context > 0);
        let pages = usable_checkpoints
            .checked_mul(64)
            .expect("page capacity overflow");
        let private = u32::try_from(
            max_num_seqs
                .checked_mul(6)
                .expect("private capacity overflow"),
        )
        .unwrap();
        Self {
            tree: TokenRadix::default(),
            checkpoints: SlotPool::new(1, 1, usable_checkpoints),
            pages: SlotPool::new(1, 1, pages),
            private: SlotPool::new(1, usable_checkpoints + 1, private),
            active: HashMap::new(),
            tails: HashMap::new(),
            mtp_contexts: HashMap::new(),
            usable: usable_checkpoints,
            max_seqs: max_num_seqs,
            max_context,
            scratch: max_num_batched_tokens.div_ceil(8192) + max_num_seqs,
            batch_tokens: max_num_batched_tokens,
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod budget_tests;
