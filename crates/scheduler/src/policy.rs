use std::fmt::Debug;

use oh_my_vllm_kv_cache::{
    group::RequestId,
    hash::BlockHash,
    state::{StateError, StateTransaction},
};

use crate::{Request, SchedulerConfig};

pub mod qwen;

#[derive(Clone, Default)]
pub struct PrefixHit<H> {
    pub length: usize,
    pub blocks: H,
}

pub trait CachePolicy: Sized {
    type Hit: Default + Clone;
    type State: Debug + Clone;
    type Snapshot: Clone;
    type Transfer;
    type TransferPlan: Debug + Clone;
    type Resources: Debug;

    fn empty_like(&self) -> Self;

    fn resource_counts(&self) -> Self::Resources;

    fn reset_prefix_cache(&mut self) -> bool;

    fn accepts(&self, request: &Request, config: &SchedulerConfig, host_enabled: bool) -> bool;

    fn initialize_hashes(&self, request: &mut Request);

    fn extend_hashes(&self, request: &mut Request);

    fn prefill_quantum(&self, budget: usize) -> usize;

    fn align_prefill(&self, request: &Request, start: usize, count: usize, budget: usize) -> usize;

    fn find_prefix(&self, request: &Request) -> PrefixHit<Self::Hit>;

    fn fallback_prefix(
        &self,
        request: &Request,
        previous_length: usize,
    ) -> Option<PrefixHit<Self::Hit>>;

    fn prefix_hash(&self, request: &Request, end: usize) -> BlockHash;

    fn new_step_starts(&mut self);

    fn prefix_eviction_needed(
        &self,
        request: &Request,
        hit: &PrefixHit<Self::Hit>,
        count: usize,
        lookahead: usize,
    ) -> bool;

    fn allocate(
        &mut self,
        request: &Request,
        hit: PrefixHit<Self::Hit>,
        count: usize,
        lookahead: usize,
        has_running: bool,
    ) -> bool;

    fn assignment(&self, request_id: RequestId) -> Self::State;

    fn commit_cache(&mut self, request: &Request);

    fn free(&mut self, request_id: RequestId);

    fn store_request(&mut self, request_id: RequestId) -> Self::Transfer;

    fn store_prefix(&mut self, hit: PrefixHit<Self::Hit>) -> Self::Transfer;

    fn load_snapshot(
        &mut self,
        request: &Request,
        source: &Self::Snapshot,
        prefix: bool,
        count: usize,
        lookahead: usize,
    ) -> Option<Self::Transfer>;

    // Plans contain wire values, never the ownership needed to keep a transfer alive.
    fn transfer_plan(transfer: Option<&Self::Transfer>) -> Self::TransferPlan;

    // Call only after validated completion; request ownership is released separately.
    fn finish_transfer(&mut self, transfer: Self::Transfer) -> Self::Snapshot;
}

pub trait TransactionalCachePolicy: CachePolicy {
    // Validate the full identity set before moving any owner out of the policy.
    fn detach_transactions(
        &mut self,
        request_ids: &[RequestId],
    ) -> Result<Vec<StateTransaction>, StateError>;

    // Restore ownership only, after validating the full set; do not publish accepted state.
    fn restore_transactions(
        &mut self,
        transactions: Vec<StateTransaction>,
    ) -> Result<(), StateError>;
}
