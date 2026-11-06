use std::collections::BTreeMap;

use oh_my_vllm_kv_cache::{
    hash::BlockHash,
    state::{AllocationRequest, StateArena, StateBlock, StateTransaction, StorageLease},
};
use oh_my_vllm_model_contract::{
    CacheGroup, CacheKind, CacheSchema, CacheSource, CacheStorage, ParallelPlan, ParallelStrategy,
    RequestPlacement, ScalarFormat,
};
use oh_my_vllm_scheduler::{
    Request, RequestOutput, Scheduler, SchedulerConfig, WorkerOutput,
    policy::{CachePolicy, PrefixHit},
};

const GROUPS: [&str; 3] = ["latent", "window", "history"];

struct ArenaPolicy {
    arena: StateArena,
    step: u64,
    accepted: BTreeMap<u64, StorageLease>,
    pending: BTreeMap<u64, StateTransaction>,
}

impl ArenaPolicy {
    fn new() -> Self {
        let schema = CacheSchema {
            schema_version: 1,
            groups: GROUPS
                .iter()
                .map(|name| CacheGroup {
                    name: (*name).into(),
                    kind: CacheKind::SlidingWindow,
                    block_tokens: 1,
                    capacity_blocks: 4,
                    storage: CacheStorage::Primitive {
                        format: ScalarFormat::F32,
                    },
                    layers: vec![0],
                    source: CacheSource::Owned {},
                })
                .collect(),
        };
        let parallel = ParallelPlan {
            strategy: ParallelStrategy::Tp1,
            world_size: 1,
            ranks: vec![0],
            request_placement: RequestPlacement::Replicated,
        };
        Self {
            arena: StateArena::new(1, &schema, &parallel, 12).unwrap(),
            step: 0,
            accepted: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }
}

impl CachePolicy for ArenaPolicy {
    type Hit = ();
    type State = Vec<StateBlock>;
    type Resources = [usize; 3];

    fn empty_like(&self) -> Self {
        Self::new()
    }

    fn resource_counts(&self) -> Self::Resources {
        GROUPS.map(|group| self.arena.free_blocks(group, 0).unwrap())
    }

    fn reset_prefix_cache(&mut self) -> bool {
        true
    }

    fn accepts(&self, request: &Request, _: &SchedulerConfig) -> bool {
        request.prompt_len <= 3 && request.max_tokens <= 3
    }

    fn initialize_hashes(&self, _: &mut Request) {}

    fn extend_hashes(&self, _: &mut Request) {}

    fn align_prefill(&self, _: &Request, _: usize, count: usize, _: usize) -> usize {
        count
    }

    fn prefill_ranges(&self, _: &Request, _: usize, limit: usize, _: usize) -> Vec<(usize, usize)> {
        if limit == 0 {
            vec![]
        } else {
            vec![(1, limit)]
        }
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
        _: usize,
        _: usize,
        _: bool,
    ) -> bool {
        assert!(!self.pending.contains_key(&request.id));
        let allocations = GROUPS.map(|group| AllocationRequest {
            group: group.into(),
            rank: 0,
            blocks: 1,
        });
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
        self.pending[&request_id].blocks().to_vec()
    }

    fn commit_cache(
        &mut self,
        request: &Request,
        _: &oh_my_vllm_scheduler::execution::CacheStepResult,
    ) {
        let transaction = self.pending.remove(&request.id).unwrap();
        let selected = transaction.select_storage(transaction.blocks()).unwrap();
        self.accepted.insert(request.id, selected);
    }

    fn free(&mut self, request_id: u64) {
        self.pending.remove(&request_id);
        self.accepted.remove(&request_id);
    }
}

#[test]
fn scheduler_uses_model_selected_state_and_validates_before_commit() {
    let mut scheduler = Scheduler::new(SchedulerConfig::default(), ArenaPolicy::new());
    assert!(!scheduler.add_request(Request::new(9, vec![1; 4], 2, vec![])));
    assert!(scheduler.add_request(Request::new(1, vec![1, 2], 3, vec![])));
    let first = scheduler.schedule();
    assert_eq!(first.scheduled.len(), 1);
    scheduler.mark_submitted(&first).unwrap();
    let assignment = &first.scheduled[0].state;
    assert_eq!(assignment.len(), 3);
    let mut names = assignment
        .iter()
        .map(|block| block.group.as_str())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["history", "latent", "window"]);
    assert_eq!(scheduler.cache_resource_counts(), [2; 3]);
    assert!(scheduler.update(WorkerOutput { outputs: vec![] }).is_err());
    assert_eq!(scheduler.cache_resource_counts(), [2; 3]);

    fn output(row: &oh_my_vllm_scheduler::ScheduledRequest<Vec<StateBlock>>) -> WorkerOutput {
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                execution: row.execution,
                work: row.work,
                cache_result: oh_my_vllm_scheduler::execution::CacheStepResult {
                    execution: row.execution,
                    computed_end: row.work.range().1,
                    accepted_end: row.accepted_end + 1,
                    exported_checkpoints: vec![],
                    page_valid_ends: vec![],
                    mtp_accepted_end: None,
                },
                token_ids: vec![3],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![],
            }],
        }
    }

    scheduler.update(output(&first.scheduled[0])).unwrap();
    assert_eq!(scheduler.cache_resource_counts(), [2; 3]);
    let second = scheduler.schedule();
    scheduler.mark_submitted(&second).unwrap();
    assert_ne!(&second.scheduled[0].state, assignment);
    assert_eq!(scheduler.cache_resource_counts(), [1; 3]);
    scheduler.update(output(&second.scheduled[0])).unwrap();
    assert_eq!(scheduler.cache_resource_counts(), [2; 3]);
    let replacement = scheduler.empty_like();
    assert_eq!(replacement.cache_resource_counts(), [3; 3]);
    assert_eq!(scheduler.cache_resource_counts(), [2; 3]);
    scheduler.abort(1);
    assert_eq!(scheduler.cache_resource_counts(), [3; 3]);
    assert_eq!(scheduler.num_running() + scheduler.num_waiting(), 0);
}
