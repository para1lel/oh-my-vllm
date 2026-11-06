use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use oh_my_vllm_kv_cache::{HybridCoordinator, group::RequestId, hash::BlockHash};
use oh_my_vllm_scheduler::{
    Request, SchedulerConfig,
    policy::{CachePolicy, PrefixHit},
};

use super::ownership::RequestOwners;

type Pool = HybridCoordinator;
type PoolHit = <Pool as CachePolicy>::Hit;

#[derive(Clone, Default)]
pub struct Owned<T> {
    rank: usize,
    value: T,
}

pub struct Dep4Cache {
    pools: [Pool; 4],
    residents: [BTreeSet<RequestId>; 4],
    resident_limit: usize,
    pub(crate) owners: Arc<Mutex<RequestOwners>>,
}

impl Dep4Cache {
    pub(crate) fn new(
        template: Pool,
        owners: Arc<Mutex<RequestOwners>>,
        max_num_seqs: usize,
    ) -> Self {
        assert!(max_num_seqs > 0, "DEP4 needs a positive resident capacity");
        let pools = std::array::from_fn(|_| template.empty_like());
        let cache = Self {
            pools,
            residents: Default::default(),
            resident_limit: max_num_seqs.div_ceil(4),
            owners,
        };
        cache.refresh();
        cache
    }

    fn can_reside(&self, rank: usize, id: RequestId) -> bool {
        self.residents[rank].contains(&id) || self.residents[rank].len() < self.resident_limit
    }

    fn rank(&self, id: RequestId) -> usize {
        self.owners.lock().unwrap().rank(id)
    }

    fn request_rank(&self, request: &Request) -> usize {
        request
            .cache_owner_rank
            .expect("DEP4 request cache has no Rust rank") as usize
    }

    fn pool(&self, request: &Request) -> &Pool {
        &self.pools[self.request_rank(request)]
    }

    fn refresh(&self) {
        let mut owners = self.owners.lock().unwrap();
        for (rank, pool) in self.pools.iter().enumerate() {
            let (_, fa, _, _, state, _) = pool.resource_counts();
            owners.set_available(rank, fa, state);
        }
    }

    fn hit(hit: PrefixHit<Owned<PoolHit>>) -> PrefixHit<PoolHit> {
        PrefixHit {
            length: hit.length,
            blocks: hit.blocks.value,
        }
    }

    fn owned(rank: usize, hit: PrefixHit<PoolHit>) -> PrefixHit<Owned<PoolHit>> {
        PrefixHit {
            length: hit.length,
            blocks: Owned {
                rank,
                value: hit.blocks,
            },
        }
    }
}

impl CachePolicy for Dep4Cache {
    type Hit = Owned<PoolHit>;
    type State = <Pool as CachePolicy>::State;
    type Resources = <Pool as CachePolicy>::Resources;

    fn empty_like(&self) -> Self {
        Self::new(
            self.pools[0].empty_like(),
            Arc::new(Mutex::new(RequestOwners::default())),
            self.resident_limit * 4,
        )
    }

    fn resource_counts(&self) -> Self::Resources {
        self.pools
            .iter()
            .map(Pool::resource_counts)
            .fold((0, 0, 0, 0, 0, 0), |a, b| {
                (
                    a.0 + b.0,
                    a.1 + b.1,
                    a.2 + b.2,
                    a.3 + b.3,
                    a.4 + b.4,
                    a.5 + b.5,
                )
            })
    }

    fn reset_prefix_cache(&mut self) -> bool {
        self.pools
            .iter_mut()
            .map(Pool::reset_prefix_cache)
            .fold(true, |a, b| a & b)
    }

    fn accepts(&self, request: &Request, config: &SchedulerConfig) -> bool {
        self.pool(request).accepts(request, config)
    }

    fn initialize_hashes(&self, request: &mut Request) {
        self.pool(request).initialize_hashes(request);
    }

    fn extend_hashes(&self, request: &mut Request) {
        self.pool(request).extend_hashes(request);
    }

    fn prefill_block_size(&self) -> usize {
        self.pools[0].block_size()
    }

    fn align_prefill(&self, request: &Request, start: usize, count: usize, budget: usize) -> usize {
        self.pool(request)
            .align_prefill(request, start, count, budget)
    }

    fn prefill_ranges(
        &self,
        request: &Request,
        start: usize,
        limit: usize,
        budget: usize,
    ) -> Vec<(usize, usize)> {
        self.pool(request)
            .prefill_ranges(request, start, limit, budget)
    }

    fn find_prefix(&self, request: &Request) -> PrefixHit<Self::Hit> {
        let rank = self.request_rank(request);
        Self::owned(rank, self.pools[rank].find_prefix(request))
    }

    fn fallback_prefix(&self, request: &Request, previous: usize) -> Option<PrefixHit<Self::Hit>> {
        let rank = self.request_rank(request);
        self.pools[rank]
            .fallback_prefix(request, previous)
            .map(|hit| Self::owned(rank, hit))
    }

    fn prefix_hash(&self, request: &Request, end: usize) -> BlockHash {
        self.pool(request).prefix_hash(request, end)
    }

    fn new_step_starts(&mut self) {
        for pool in &mut self.pools {
            pool.new_step_starts();
        }
        self.refresh();
    }

    fn prefix_eviction_needed(
        &self,
        request: &Request,
        hit: &PrefixHit<Self::Hit>,
        count: usize,
        lookahead: usize,
    ) -> bool {
        let rank = self.request_rank(request);
        if hit.length > 0 {
            assert_eq!(rank, hit.blocks.rank);
        }
        <Pool as CachePolicy>::prefix_eviction_needed(
            &self.pools[rank],
            request,
            &Self::hit(hit.clone()),
            count,
            lookahead,
        )
    }

    fn allocate(
        &mut self,
        request: &Request,
        hit: PrefixHit<Self::Hit>,
        count: usize,
        lookahead: usize,
        has_running: bool,
    ) -> bool {
        let rank = self.request_rank(request);
        if hit.length > 0 {
            assert_eq!(rank, hit.blocks.rank);
        }
        if !self.can_reside(rank, request.id) {
            return false;
        }
        let result =
            self.pools[rank].allocate(request, Self::hit(hit), count, lookahead, has_running);
        if result {
            self.residents[rank].insert(request.id);
        }
        self.refresh();
        result
    }

    fn assignment(&self, id: RequestId) -> Self::State {
        self.pools[self.rank(id)].assignment(id)
    }

    fn commit_cache(
        &mut self,
        request: &Request,
        step: &oh_my_vllm_scheduler::execution::CacheStepResult,
    ) {
        let rank = self.request_rank(request);
        <Pool as CachePolicy>::commit_cache(&mut self.pools[rank], request, step);
        self.refresh();
    }

    fn validate_cache_step(
        &self,
        request: &Request,
        step: &oh_my_vllm_scheduler::execution::CacheStepResult,
    ) -> Result<(), &'static str> {
        self.pool(request).validate_cache_step(request, step)
    }

    fn free(&mut self, id: RequestId) {
        let rank = self.rank(id);
        self.pools[rank].free(id);
        self.residents[rank].remove(&id);
        self.refresh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oh_my_vllm_scheduler::Scheduler;

    #[test]
    fn full_owner_does_not_consume_another_owners_capacity() {
        let owners = Arc::new(Mutex::new(RequestOwners::default()));
        let mut cache = Dep4Cache::new(
            Pool::new(4, 128, true, 0).with_mamba_capacity(8),
            owners.clone(),
            4,
        );
        for id in 1..=4 {
            let mut request = Request::new(id, vec![id as u32; 384], 1, vec![]);
            owners.lock().unwrap().bind_cache(&mut request);
            assert_eq!(owners.lock().unwrap().rank(id), (id - 1) as usize);
            cache.initialize_hashes(&mut request);
            let hit = cache.find_prefix(&request);
            assert!(cache.allocate(&request, hit, 384, 0, false));
            assert_eq!(cache.pools[(id - 1) as usize].pool().num_free_blocks(), 0);
        }
        let mut request = Request::new(5, vec![9; 384], 1, vec![]);
        owners.lock().unwrap().bind_cache(&mut request);
        cache.initialize_hashes(&mut request);
        let hit = cache.find_prefix(&request);
        assert!(!cache.allocate(&request, hit, 384, 0, false));
        cache.free(1);
        let hit = cache.find_prefix(&request);
        assert!(cache.allocate(&request, hit, 384, 0, false));
        assert_eq!(cache.pools[1].pool().num_free_blocks(), 0);
    }

    #[test]
    fn resident_quota_keeps_prepared_requests_queued() {
        let owners = Arc::new(Mutex::new(RequestOwners::default()));
        let mut cache = Dep4Cache::new(
            Pool::new(256, 128, true, 0).with_mamba_capacity(128),
            owners.clone(),
            64,
        );
        let config = SchedulerConfig {
            max_num_seqs: 64,
            max_num_batched_tokens: 4096,
            enable_mtp: false,
            mtp_draft_len: 0,
        };
        let mut requests = Vec::new();
        for id in 1..=68 {
            let mut request = Request::new(id, vec![id as u32; 256], 4, vec![]);
            owners.lock().unwrap().bind_cache(&mut request);
            cache.initialize_hashes(&mut request);
            assert!(cache.accepts(&request, &config));
            requests.push(request);
        }
        for request in &requests[..64] {
            let hit = cache.find_prefix(request);
            assert!(cache.allocate(request, hit, 256, 0, false));
        }
        assert!(cache.residents.iter().all(|rows| rows.len() == 16));
        let queued = &requests[64];
        let before = cache.resource_counts();
        let hit = cache.find_prefix(queued);
        assert!(!cache.allocate(queued, hit, 256, 0, true));
        assert_eq!(cache.resource_counts(), before);
        cache.free(1);
        let hit = cache.find_prefix(queued);
        assert!(cache.allocate(queued, hit, 256, 0, true));
        assert_eq!(cache.rank(1), 0);
    }

    #[test]
    fn scheduler_uses_four_pools_before_queueing_and_preserves_restart_owner() {
        let owners = Arc::new(Mutex::new(RequestOwners::default()));
        let cache = Dep4Cache::new(
            Pool::new(4, 128, true, 0).with_mamba_capacity(8),
            owners.clone(),
            8,
        );
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: 4096,
                enable_mtp: false,
                mtp_draft_len: 0,
            },
            cache,
        );
        for id in 1..=5 {
            let mut request = Request::new(id, vec![id as u32; 384], 1, vec![]);
            owners.lock().unwrap().bind_cache(&mut request);
            assert!(scheduler.add_request(request));
        }
        let step = scheduler.schedule();
        scheduler.mark_submitted(&step).unwrap();
        assert_eq!(step.scheduled.len(), 4);
        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 4);
        let rank = owners.lock().unwrap().rank(1);
        scheduler.abort(1);
        assert_eq!(owners.lock().unwrap().rank(1), rank);
        let next = scheduler.schedule();
        assert!(next.scheduled.is_empty());
        assert_eq!(scheduler.num_waiting(), 1);
    }
}
