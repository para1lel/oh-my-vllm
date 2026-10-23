use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use oh_my_vllm_kv_cache::{
    hash::BlockHash,
    state::{AllocationRequest, StateArena, StateError, StateTransaction, StorageLease},
};
use oh_my_vllm_model_contract::{ModelDescriptor, RequestPlacement};
use oh_my_vllm_scheduler::{
    Request, SchedulerConfig,
    policy::{CachePolicy, PrefixHit, TransactionalCachePolicy},
};

use oh_my_vllm_zmq_worker::group::model::ModelState;
use serde::Serialize;

const GROUPS: [&str; 3] = ["latent", "window", "history"];

#[derive(Debug, Clone, Serialize)]
pub struct Layout {
    pub emit: bool,
}

pub struct ArenaPolicy {
    arena: StateArena,
    descriptor: ModelDescriptor,
    epoch: u64,
    detached: BTreeMap<u64, u64>,
    emits: BTreeMap<u64, bool>,
    pub commits: Arc<AtomicUsize>,
    step: u64,
    accepted: BTreeMap<u64, StorageLease>,
    pending: BTreeMap<u64, StateTransaction>,
}

impl ArenaPolicy {
    pub fn new(descriptor: ModelDescriptor, epoch: u64) -> Self {
        Self {
            arena: StateArena::new(
                epoch,
                &descriptor.cache_schema,
                &descriptor.parallel_plan,
                96,
            )
            .unwrap(),
            descriptor,
            epoch,
            step: 0,
            accepted: BTreeMap::new(),
            pending: BTreeMap::new(),
            detached: BTreeMap::new(),
            emits: BTreeMap::new(),
            commits: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl CachePolicy for ArenaPolicy {
    type Hit = ();
    type State = ModelState<Layout>;
    type Snapshot = ();
    type Transfer = StorageLease;
    type TransferPlan = ();
    type Resources = usize;

    fn empty_like(&self) -> Self {
        Self::new(self.descriptor.clone(), self.epoch + 1)
    }

    fn resource_counts(&self) -> Self::Resources {
        GROUPS
            .iter()
            .flat_map(|group| (0..4).map(move |rank| (*group, rank)))
            .map(|(group, rank)| self.arena.free_blocks(group, rank).unwrap())
            .sum()
    }

    fn reset_prefix_cache(&mut self) -> bool {
        true
    }

    fn accepts(&self, request: &Request, _: &SchedulerConfig, host: bool) -> bool {
        !host && request.prompt_len + request.max_tokens <= 97
    }

    fn initialize_hashes(&self, _: &mut Request) {}

    fn extend_hashes(&self, _: &mut Request) {}

    fn prefill_quantum(&self, _: usize) -> usize {
        1
    }

    fn align_prefill(&self, _: &Request, _: usize, count: usize, _: usize) -> usize {
        count
    }

    fn find_prefix(&self, _: &Request) -> PrefixHit<Self::Hit> {
        PrefixHit::default()
    }

    fn fallback_prefix(&self, _: &Request, _: usize) -> Option<PrefixHit<Self::Hit>> {
        None
    }

    fn prefix_hash(&self, _: &Request, _: usize) -> BlockHash {
        unreachable!("this test policy has no prefix cache")
    }

    fn new_step_starts(&mut self) {
        self.step += 1;
    }

    fn prefix_eviction_needed(
        &self,
        _: &Request,
        _: &PrefixHit<Self::Hit>,
        _: usize,
        _: usize,
    ) -> bool {
        false
    }

    fn allocate(
        &mut self,
        request: &Request,
        _: PrefixHit<Self::Hit>,
        count: usize,
        _: usize,
        _: bool,
    ) -> bool {
        assert!(!self.pending.contains_key(&request.id));
        let ranks: Vec<_> =
            if self.descriptor.parallel_plan.request_placement == RequestPlacement::Replicated {
                (0..4).collect()
            } else {
                vec![(request.id % 4) as u32]
            };
        let allocations = ranks
            .iter()
            .flat_map(|rank| {
                GROUPS.map(|group| AllocationRequest {
                    group: group.into(),
                    rank: *rank,
                    blocks: 1,
                })
            })
            .collect::<Vec<_>>();
        self.emits.insert(
            request.id,
            request.num_computed_tokens + count >= request.token_ids.len().max(request.prompt_len),
        );
        let reads = self
            .accepted
            .get(&request.id)
            .into_iter()
            .collect::<Vec<_>>();
        match self
            .arena
            .reserve(self.step, request.id, &allocations, &reads)
        {
            Ok(transaction) => {
                self.pending.insert(request.id, transaction);
                true
            }
            Err(_) => false,
        }
    }

    fn assignment(&self, request_id: u64) -> Self::State {
        ModelState {
            transaction: self.pending[&request_id].key(),
            reads: self.pending[&request_id].read_blocks().to_vec(),
            writes: self.pending[&request_id].blocks().to_vec(),
            layout: Layout {
                emit: self.emits[&request_id],
            },
        }
    }

    fn commit_cache(&mut self, request: &Request) {
        let transaction = self.pending.remove(&request.id).unwrap();
        let selected = transaction.select_storage(transaction.blocks()).unwrap();
        self.accepted.insert(request.id, selected);
        self.commits.fetch_add(1, Ordering::SeqCst);
    }

    fn free(&mut self, request_id: u64) {
        self.pending.remove(&request_id);
        self.accepted.remove(&request_id);
    }

    fn store_request(&mut self, _: u64) -> Self::Transfer {
        unreachable!("host cache is disabled")
    }

    fn store_prefix(&mut self, _: PrefixHit<Self::Hit>) -> Self::Transfer {
        unreachable!("host cache is disabled")
    }

    fn load_snapshot(
        &mut self,
        _: &Request,
        _: &Self::Snapshot,
        _: bool,
        _: usize,
        _: usize,
    ) -> Option<Self::Transfer> {
        unreachable!("host cache is disabled")
    }

    fn transfer_plan(_: Option<&Self::Transfer>) -> Self::TransferPlan {
        unreachable!("host cache is disabled")
    }

    fn finish_transfer(&mut self, _: Self::Transfer) -> Self::Snapshot {
        unreachable!("host cache is disabled")
    }
}

impl TransactionalCachePolicy for ArenaPolicy {
    fn detach_transactions(&mut self, ids: &[u64]) -> Result<Vec<StateTransaction>, StateError> {
        let unique = ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != ids.len()
            || unique != self.pending.keys().copied().collect()
            || !self.detached.is_empty()
        {
            return Err(StateError::Invalid("invalid detach set"));
        }
        let transactions = ids
            .iter()
            .map(|id| self.pending.remove(id).unwrap())
            .collect::<Vec<_>>();
        self.detached = transactions
            .iter()
            .map(|transaction| {
                let key = transaction.key();
                (key.request_id, key.transaction_id)
            })
            .collect();
        Ok(transactions)
    }

    fn restore_transactions(
        &mut self,
        transactions: Vec<StateTransaction>,
    ) -> Result<(), StateError> {
        let keys = transactions
            .iter()
            .map(|transaction| {
                let key = transaction.key();
                (key.request_id, key.transaction_id)
            })
            .collect::<BTreeMap<_, _>>();
        if keys.len() != transactions.len() || keys != self.detached || !self.pending.is_empty() {
            return Err(StateError::Invalid("invalid restore set"));
        }
        self.pending = transactions
            .into_iter()
            .map(|transaction| (transaction.key().request_id, transaction))
            .collect();
        self.detached.clear();
        Ok(())
    }
}
