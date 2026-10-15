use crate::{
    coordinator::HybridCoordinator,
    group::{CacheRequest, RequestId},
    hash::BlockHash,
};

#[derive(Clone)]
struct Req {
    id: RequestId,
    num_tokens: usize,
    num_prompt_tokens: usize,
    num_computed_tokens: usize,
    num_in_flight_tokens: usize,
    block_hashes: Vec<BlockHash>,
    waiting: bool,
}

impl Req {
    fn new(id: u64, num_tokens: usize) -> Self {
        Self {
            id,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_computed_tokens: 0,
            num_in_flight_tokens: 0,
            block_hashes: Vec::new(),
            waiting: true,
        }
    }

    fn with_computed(mut self, computed: usize) -> Self {
        self.num_computed_tokens = computed;
        self
    }

    fn with_hashes(mut self, hashes: Vec<BlockHash>) -> Self {
        self.block_hashes = hashes;
        self
    }
}

impl CacheRequest for Req {
    fn request_id(&self) -> RequestId {
        self.id
    }

    fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    fn num_prompt_tokens(&self) -> usize {
        self.num_prompt_tokens
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
        self.waiting
    }
}

const BS: usize = 4;

fn coord_no_cache(num_blocks: u32) -> HybridCoordinator {
    HybridCoordinator::new(num_blocks, BS, false, 0)
}

#[test]
fn independent_state_exhaustion_does_not_allocate_fa_pages() {
    let mut c = coord_no_cache(12).with_mamba_capacity(2);
    c.new_step_starts();
    let first = Req::new(1, BS);
    assert!(
        c.allocate_slots(&first, (vec![], vec![]), 0, BS, 0, false)
            .is_some()
    );
    let free = c.pool().num_free_blocks();
    let second = Req::new(2, BS);
    assert!(
        c.allocate_slots(&second, (vec![], vec![]), 0, BS, 0, true)
            .is_none()
    );
    assert_eq!(c.pool().num_free_blocks(), free);
    c.free(1);
    c.new_step_starts();
    assert!(
        c.allocate_slots(&second, (vec![], vec![]), 0, BS, 0, false)
            .is_some()
    );
}

#[test]
fn speculative_prefix_hit_allocates_private_state_and_draft_slots() {
    for separate in [false, true] {
        let mut kv = HybridCoordinator::new(64, 784, true, 0);
        if separate {
            kv = kv.with_mamba_capacity(16);
        }
        kv.set_speculative_blocks(4);
        let tokens: Vec<u32> = (0..784).collect();
        let hashes = kv.compute_block_hashes(&tokens);
        let req = Req::new(1, 784).with_hashes(hashes.clone());
        kv.allocate_slots(&req, (vec![], vec![]), 0, 784, 0, false)
            .unwrap();
        let checkpoint = kv.mamba_blocks(1)[0];
        kv.commit_cache(&req.clone().with_computed(784));
        kv.free(1);
        kv.new_step_starts();
        let req2 = Req::new(2, 785).with_hashes(hashes);
        let (fa, mb, hit) = kv.get_computed_blocks(&req2);
        assert_eq!(hit, 784);
        assert_eq!(mb, vec![checkpoint]);
        kv.allocate_slots(&req2, (fa, mb), hit, 1, 0, false)
            .unwrap();
        let table = kv.mamba_blocks(2);
        assert_eq!(table.len(), 6);
        assert_eq!(table[0], checkpoint);
        let unique: std::collections::HashSet<_> = table.iter().copied().collect();
        assert_eq!(unique.len(), 6);
        kv.free(2);
        assert_eq!(kv.pool().num_free_blocks(), 63);
    }
}
