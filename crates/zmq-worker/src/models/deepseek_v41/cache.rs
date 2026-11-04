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
    pub(crate) owners: Arc<Mutex<RequestOwners>>,
}

impl Dep4Cache {
    pub(crate) fn new(template: Pool, owners: Arc<Mutex<RequestOwners>>) -> Self {
        let pools = std::array::from_fn(|_| template.empty_like());
        let cache = Self { pools, owners };
        cache.refresh();
        cache
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
    type Snapshot = Owned<<Pool as CachePolicy>::Snapshot>;
    type Transfer = Owned<<Pool as CachePolicy>::Transfer>;
    type TransferPlan = <Pool as CachePolicy>::TransferPlan;
    type Resources = <Pool as CachePolicy>::Resources;

    fn empty_like(&self) -> Self {
        Self::new(
            self.pools[0].empty_like(),
            Arc::new(Mutex::new(RequestOwners::default())),
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

    fn accepts(&self, request: &Request, config: &SchedulerConfig, host: bool) -> bool {
        self.pool(request).accepts(request, config, host)
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
        let result =
            self.pools[rank].allocate(request, Self::hit(hit), count, lookahead, has_running);
        self.refresh();
        result
    }

    fn assignment(&self, id: RequestId) -> Self::State {
        self.pools[self.rank(id)].assignment(id)
    }

    fn commit_cache(&mut self, request: &Request) {
        let rank = self.request_rank(request);
        self.pools[rank].commit_cache(request);
        self.refresh();
    }

    fn free(&mut self, id: RequestId) {
        let rank = self.rank(id);
        self.pools[rank].free(id);
        self.refresh();
    }

    fn store_request(&mut self, id: RequestId) -> Self::Transfer {
        let rank = self.rank(id);
        Owned {
            rank,
            value: self.pools[rank].store_request(id),
        }
    }

    fn store_prefix(&mut self, hit: PrefixHit<Self::Hit>) -> Self::Transfer {
        let rank = hit.blocks.rank;
        Owned {
            rank,
            value: self.pools[rank].store_prefix(Self::hit(hit)),
        }
    }

    fn load_snapshot(
        &mut self,
        request: &Request,
        source: &Self::Snapshot,
        prefix: bool,
        count: usize,
        lookahead: usize,
    ) -> Option<Self::Transfer> {
        let rank = self.request_rank(request);
        if rank != source.rank {
            return None;
        }
        let result = self.pools[rank]
            .load_snapshot(request, &source.value, prefix, count, lookahead)
            .map(|value| Owned { rank, value });
        self.refresh();
        result
    }

    fn transfer_plan(transfer: Option<&Self::Transfer>) -> Self::TransferPlan {
        let mut plan = Pool::transfer_plan(transfer.map(|transfer| &transfer.value));
        plan.owner_rank = transfer.map(|transfer| transfer.rank as u32);
        plan
    }

    fn finish_transfer(&mut self, transfer: Self::Transfer) -> Self::Snapshot {
        let result = Owned {
            rank: transfer.rank,
            value: self.pools[transfer.rank].finish_transfer(transfer.value),
        };
        self.refresh();
        result
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
    fn prefix_candidate_keeps_cache_rank_after_request_owner_release() {
        let owners = Arc::new(Mutex::new(RequestOwners::default()));
        let mut cache = Dep4Cache::new(
            Pool::new(8, 128, true, 0).with_mamba_capacity(8),
            owners.clone(),
        );
        let mut request = Request::new(1, vec![7; 384], 1, vec![]);
        owners.lock().unwrap().bind_cache(&mut request);
        cache.initialize_hashes(&mut request);
        let hit = cache.find_prefix(&request);
        assert!(cache.allocate(&request, hit, 384, 0, false));
        request.num_computed_tokens = 384;
        request.append_token(9);
        cache.commit_cache(&request);
        cache.free(1);
        let plan = owners.lock().unwrap().schedule(1, &[], &[1], &[]).unwrap();
        let replies = plan
            .iter()
            .map(|schedule| crate::protocol::RankCompletion {
                rank: schedule.rank,
                step_id: 1,
                schedule: schedule.clone(),
            })
            .collect::<Vec<_>>();
        owners.lock().unwrap().complete(1, &replies).unwrap();
        let hit = cache.find_prefix(&request);
        assert!(hit.length > 0);
        let transfer = cache.store_prefix(hit);
        assert_eq!(
            Dep4Cache::transfer_plan(Some(&transfer)).owner_rank,
            Some(0)
        );
        cache.finish_transfer(transfer);
    }

    #[test]
    fn scheduler_uses_four_pools_before_queueing_and_preserves_restart_owner() {
        let owners = Arc::new(Mutex::new(RequestOwners::default()));
        let cache = Dep4Cache::new(
            Pool::new(4, 128, true, 0).with_mamba_capacity(8),
            owners.clone(),
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
        assert_eq!(step.scheduled.len(), 4);
        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 4);
        let rank = owners.lock().unwrap().rank(1);
        scheduler.abort(1);
        assert_eq!(owners.lock().unwrap().rank(1), rank);
        let next = scheduler.schedule();
        assert!(next.scheduled.iter().any(|row| row.request_id == 5));
    }
}
