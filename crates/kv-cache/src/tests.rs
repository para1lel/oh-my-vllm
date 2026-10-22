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
    token_ids: Vec<u32>,
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
            token_ids: (0..num_tokens as u32).collect(),
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

    fn token_ids(&self) -> &[u32] {
        &self.token_ids
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

#[test]
fn partial_prefixes_keep_private_continuations_and_transactional_ownership() {
    for (length, speculative, separate) in [(317, 0, false), (1234, 4, true)] {
        let mut kv = HybridCoordinator::new(64, 784, true, 0);
        if separate {
            kv = kv.with_mamba_capacity(16);
        }
        kv.set_speculative_blocks(speculative);
        let mut source = Req::new(1, length);
        source.block_hashes = kv.compute_block_hashes(&source.token_ids);
        let endpoint = length - 1;
        for end in [784.min(endpoint), endpoint] {
            if end == source.num_computed_tokens {
                continue;
            }
            kv.new_step_starts();
            kv.allocate_slots(
                &source,
                (vec![], vec![]),
                0,
                end - source.num_computed_tokens,
                0,
                false,
            )
            .unwrap();
            source.num_computed_tokens = end;
            if end == endpoint {
                assert_eq!(kv.get_computed_blocks(&source).2, endpoint / 784 * 784);
                assert_eq!(kv.cache_checkpoint(1).unwrap().0, endpoint);
            }
            kv.commit_cache(&source);
        }
        kv.free(1);
        kv.new_step_starts();
        let mut first = source.clone();
        first.id = 2;
        first.num_computed_tokens = 0;
        let (fa, mb, hit) = kv.get_computed_blocks(&first);
        assert_eq!(hit, endpoint);
        let tail = *fa.last().unwrap();
        let state = *mb.last().unwrap();
        kv.allocate_slots(&first, (fa, mb), hit, 1, 0, false)
            .unwrap();
        assert_eq!(kv.prefix_state_slot(2), Some(state));
        assert_eq!(
            kv.block_copies(2),
            vec![(
                tail,
                *kv.full_attn_blocks(2).last().unwrap(),
                (hit % 784) as u32
            )],
        );
        assert_ne!(kv.full_attn_blocks(2)[hit / 784], tail);
        assert_ne!(kv.mamba_blocks(2)[hit / 784], state);
        assert!(kv.pool().block(tail).ref_cnt > 0);

        let mut second = first.clone();
        second.id = 3;
        let (fa, mb, hit) = kv.get_computed_blocks(&second);
        assert_eq!(hit, endpoint);
        kv.allocate_slots(&second, (fa, mb), hit, 1, 0, true)
            .unwrap();
        assert_ne!(kv.full_attn_blocks(2), kv.full_attn_blocks(3));
        assert_ne!(kv.mamba_blocks(2)[hit / 784], kv.mamba_blocks(3)[hit / 784]);
        first.num_computed_tokens = length;
        kv.commit_cache(&first);
        kv.free(2);
        assert_eq!(kv.get_computed_blocks(&second).2, endpoint);
        kv.free(3);
        assert_eq!(kv.pool().num_free_blocks(), 63);
        assert!(kv.reset_prefix_cache());
        assert_eq!(kv.get_computed_blocks(&second).2, 0);
    }

    let mut kv = HybridCoordinator::new(8, 784, true, 0).with_mamba_capacity(3);
    let source = Req::new(1, 317);
    kv.allocate_slots(&source, (vec![], vec![]), 0, 316, 0, false)
        .unwrap();
    kv.commit_cache(&source.clone().with_computed(316));
    kv.free(1);
    kv.allocate_slots(&Req::new(3, 1), (vec![], vec![]), 0, 1, 0, false)
        .unwrap();
    let request = Req::new(2, 317);
    let (fa, mb, hit) = kv.get_computed_blocks(&request);
    let before = kv.resource_counts();
    assert_eq!(hit, 316);
    assert!(
        kv.allocate_slots(&request, (fa, mb), hit, 1, 0, false)
            .is_none()
    );
    assert_eq!(kv.resource_counts(), before);
    assert_eq!(kv.get_computed_blocks(&request).2, 316);
}

#[test]
fn restored_partial_checkpoint_keeps_read_lease_until_commit_or_cancel() {
    use crate::snapshot::BlockTables;

    for (speculative, separate) in [(0, false), (4, true)] {
        for commit in [false, true] {
            let mut kv = HybridCoordinator::new(32, 784, true, 0);
            if separate {
                kv = kv.with_mamba_capacity(16);
            }
            kv.set_speculative_blocks(speculative);
            let baseline = kv.resource_counts();
            let mut request = Req::new(1, 2000).with_computed(1566);
            request.block_hashes = kv.compute_block_hashes(&request.token_ids);
            let tables = kv
                .reserve_tables(&BlockTables {
                    fa: vec![1, 2],
                    mamba: vec![0, 7],
                    mamba_last_state: None,
                    mamba_read_source: None,
                })
                .unwrap();
            let mut read = tables.mamba[1];
            kv.pin_tables(&tables);
            kv.install_tables(1, tables.clone(), true, 1566);
            kv.allocate_slots(&request, (vec![], vec![]), 0, 2, speculative, true)
                .unwrap();
            assert_eq!(kv.prefix_state_slot(1), Some(read));
            assert_ne!(kv.mamba_blocks(1)[1], read);
            kv.release_tables(&tables);
            for _ in 0..2 {
                let stored = kv.request_tables(1);
                assert_eq!(stored.mamba_read_source, Some(read));
                let read_index = stored
                    .mamba_sources()
                    .iter()
                    .position(|&slot| slot == read)
                    .unwrap();
                kv.free(1);
                let loaded = kv.reserve_tables(&stored).unwrap();
                read = loaded.mamba_read_source.unwrap();
                assert_eq!(loaded.mamba_sources()[read_index], read);
                assert_ne!(loaded.mamba[1], read);
                kv.pin_tables(&loaded);
                kv.install_tables(1, loaded.clone(), false, 1566);
                kv.allocate_slots(&request, (vec![], vec![]), 0, 2, speculative, true)
                    .unwrap();
                kv.release_tables(&loaded);
                assert_eq!(kv.prefix_state_slot(1), Some(read));
            }
            let leased = kv.resource_counts();
            if !separate {
                assert_eq!(kv.pool().block(read).ref_cnt, 1);
            }
            if commit {
                request.num_computed_tokens = 1568;
                kv.commit_cache(&request);
                assert_eq!(kv.resource_counts().4, leased.4 + 1);
                assert_eq!(kv.prefix_state_slot(1), None);
            }
            kv.free(1);
            assert_eq!(kv.resource_counts().1, baseline.1);
            assert_eq!(kv.resource_counts().4, baseline.4);
        }
    }

    let mut kv = HybridCoordinator::new(4, 784, true, 0).with_mamba_capacity(2);
    let tables = kv
        .reserve_tables(&BlockTables {
            fa: vec![1, 2],
            mamba: vec![0, 7],
            mamba_last_state: None,
            mamba_read_source: None,
        })
        .unwrap();
    kv.pin_tables(&tables);
    kv.install_tables(1, tables.clone(), true, 1566);
    let reserved = kv.resource_counts();
    assert!(
        kv.allocate_slots(
            &Req::new(1, 2000).with_computed(1566),
            (vec![], vec![]),
            0,
            2,
            0,
            true,
        )
        .is_none()
    );
    assert_eq!(kv.resource_counts(), reserved);
    kv.free(1);
    kv.release_tables(&tables);
    assert_eq!(kv.resource_counts().1, 3);
    assert_eq!(kv.resource_counts().4, 1);
}
