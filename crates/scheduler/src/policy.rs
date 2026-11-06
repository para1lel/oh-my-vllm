use std::fmt::Debug;

use oh_my_vllm_kv_cache::{
    group::RequestId,
    hash::BlockHash,
    state::{StateError, StateTransaction},
};

use crate::{Request, SchedulerConfig};

pub mod qwen;
pub mod resident;

#[derive(Clone, Default)]
pub struct PrefixHit<H> {
    pub length: usize,
    pub blocks: H,
}

pub trait CachePolicy: Sized {
    type Hit: Default + Clone;
    type State: Debug + Clone;
    type Resources: Debug;

    fn empty_like(&self) -> Self;

    fn resource_counts(&self) -> Self::Resources;

    fn reset_prefix_cache(&mut self) -> bool;

    fn reset_worker_epoch(&mut self, _: u64) {
        assert!(
            self.reset_prefix_cache(),
            "worker reset requires an empty policy"
        );
    }

    fn accepts(&self, request: &Request, config: &SchedulerConfig) -> bool;

    fn initialize_hashes(&self, request: &mut Request);

    fn extend_hashes(&self, request: &mut Request);

    fn batch_token_limit(&self, config: &SchedulerConfig) -> usize {
        config.max_num_batched_tokens
    }

    fn prefill_block_size(&self) -> usize {
        1
    }

    fn align_prefill(&self, request: &Request, start: usize, count: usize, budget: usize) -> usize;

    fn prefill_ranges(
        &self,
        request: &Request,
        start: usize,
        limit: usize,
        budget: usize,
    ) -> Vec<(usize, usize)>;

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

    fn commit_cache(&mut self, request: &Request, step: &crate::execution::CacheStepResult);

    fn completion_plan(&self, request: &Request) -> crate::execution::CompletionPlan {
        crate::execution::CompletionPlan::Ready {
            endpoint: request.num_computed_tokens,
        }
    }

    fn validate_cache_step(
        &self,
        _: &Request,
        _: &crate::execution::CacheStepResult,
    ) -> Result<(), &'static str> {
        Ok(())
    }

    fn finish_request(&mut self, _: &Request, _: crate::execution::FinishOutcome) {}

    fn free(&mut self, request_id: RequestId);
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
