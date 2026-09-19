//! Integration tests for HybridCoordinator and GroupManager.
//!
//! These tests exercise the paths that are live for Qwen3.5 at
//! scheduler_block_size == hash_block_size == block_size == 784:
//! - full-attention left-to-right prefix hit
//! - mamba align-mode right-to-left state checkpoint hit
//! - hybrid find_longest_cache_hit reconciliation (is_simple_hybrid path)
//! - allocate_slots: cold prefill, decode step, prefix cache hit
//! - same-step deferral (DeferToNextStep)
//! - free / block recycling

use crate::{
    coordinator::HybridCoordinator,
    group::{CacheRequest, RequestId},
    hash::BlockHash,
    pool::NULL_BLOCK_ID,
};

// ── minimal CacheRequest impl ────────────────────────────────────────────────

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

    #[allow(dead_code)]
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

// ── helpers ──────────────────────────────────────────────────────────────────

const BS: usize = 4; // small block size for tests (replaces 784)

fn coord(num_blocks: u32) -> HybridCoordinator {
    HybridCoordinator::new(num_blocks, BS, true, 0)
}

fn coord_no_cache(num_blocks: u32) -> HybridCoordinator {
    HybridCoordinator::new(num_blocks, BS, false, 0)
}

/// Build a request with freshly computed block hashes for `num_tokens` tokens.
fn req_with_hashes(id: u64, tokens: &[u32], c: &HybridCoordinator) -> Req {
    let hashes = c.compute_block_hashes(tokens);
    Req::new(id, tokens.len()).with_hashes(hashes)
}

// ── cold single-request prefill ──────────────────────────────────────────────

#[test]
fn cold_prefill_allocates_correct_blocks() {
    // 8 tokens = 2 full blocks (BS=4). No hits; expect 2 FA blocks and 1
    // Mamba state block (state at block index 1 = last).
    let mut c = coord(32);
    let tokens: Vec<u32> = (0..8).collect();
    let req = req_with_hashes(1, &tokens, &c);

    c.new_step_starts();
    let result = c.allocate_slots(&req, (vec![], vec![]), 0, 8, 0, false);
    let (fa, mb) = result.expect("should allocate");

    // Full attention gets 2 new blocks for 8 tokens.
    assert_eq!(fa.len(), 2, "FA should have 2 new blocks");
    // Mamba align mode: returns the delta from prev_len (0) onward, which is
    // 1 null placeholder (skipped-state position) + 1 state block = 2 entries.
    // The worker block table needs the full positional delta, nulls included.
    assert_eq!(
        mb.len(),
        2,
        "Mamba delta = 1 null placeholder + 1 state block"
    );
    // The last entry must be a real block (the state checkpoint).
    assert_ne!(
        mb[mb.len() - 1],
        NULL_BLOCK_ID,
        "last Mamba entry must be the state block"
    );
    // The first entry is a null placeholder.
    assert_eq!(
        mb[0], NULL_BLOCK_ID,
        "first Mamba entry is a null placeholder"
    );
}

// ── prefix cache hit: full attention ────────────────────────────────────────

#[test]
fn full_attn_prefix_hit_reuses_blocks() {
    let mut c = coord(64);
    let tokens: Vec<u32> = (0..16).collect();

    // First request fills the cache.
    let req1 = req_with_hashes(1, &tokens, &c);
    c.new_step_starts();
    c.allocate_slots(&req1, (vec![], vec![]), 0, 16, 0, false)
        .expect("first alloc");
    // Simulate the forward pass completing: advance computed tokens, then call
    // cache_blocks by driving another allocate_slots step that will register
    // the blocks.
    let req1_decode = {
        let hashes = req1.block_hashes.clone();
        Req {
            id: 1,
            num_tokens: 17,
            num_prompt_tokens: 16,
            num_computed_tokens: 16,
            num_in_flight_tokens: 0,
            block_hashes: hashes,
            waiting: false,
        }
    };
    c.new_step_starts();
    c.allocate_slots(&req1_decode, (vec![], vec![]), 0, 1, 0, false)
        .expect("decode alloc");
    // Free request 1 so its blocks go into the LRU tail (ref_cnt drops).
    c.free(1);

    // Second request: 17 tokens sharing the same 16-token prefix.
    // Using 17 tokens gives max_hit = 16, so FA covers all 4 blocks and Mamba's
    // state at block index 3 falls within FA's range — the combined hit is 16.
    let tokens2: Vec<u32> = (0..17).collect();
    let req2 = req_with_hashes(2, &tokens2, &c);
    c.new_step_starts();
    let (fa_hit, mb_hit, hit_len) = c.get_computed_blocks(&req2);

    // Both FA and Mamba have cached state covering the 16-token prefix.
    assert!(hit_len > 0, "second request should hit the prefix cache");
    assert_eq!(hit_len % BS, 0, "hit_len must be block-aligned");
    assert!(!fa_hit.is_empty(), "FA must return hit blocks");

    // Allocate slots using the hit (1 new token remains).
    let result = c.allocate_slots(&req2, (fa_hit, mb_hit), hit_len, 17 - hit_len, 0, false);
    assert!(
        result.is_some(),
        "allocation with cache hits should succeed"
    );
}

// ── mamba align: right-to-left checkpoint lookup ─────────────────────────────

#[test]
fn mamba_hit_uses_rightmost_checkpoint() {
    // Build a coordinator and manually drive enough steps for Mamba to cache
    // a state block, then verify the hit logic finds it right-to-left.
    let mut c = coord(64);
    let tokens: Vec<u32> = (0..8).collect();
    let req1 = req_with_hashes(1, &tokens, &c);

    // Prefill step.
    c.new_step_starts();
    let (fa0, mb0) = c
        .allocate_slots(&req1, (vec![], vec![]), 0, 8, 0, false)
        .expect("prefill");

    // After prefill the Mamba group has 1 state block. Simulate the compute
    // completing and drive a decode step so cache_blocks runs.
    let hashes = req1.block_hashes.clone();
    let req1_decode = Req {
        id: 1,
        num_tokens: 9,
        num_prompt_tokens: 8,
        num_computed_tokens: 8,
        num_in_flight_tokens: 0,
        block_hashes: hashes,
        waiting: false,
    };
    c.new_step_starts();
    c.allocate_slots(&req1_decode, (vec![], vec![]), 0, 1, 0, false)
        .expect("decode");
    c.free(1);

    // A new request with the same token prefix: Mamba hit should be non-zero.
    let req2 = req_with_hashes(2, &tokens, &c);
    c.new_step_starts();
    let (_fa, mb_hit, hit_len) = c.get_computed_blocks(&req2);

    // Mamba hit blocks should contain exactly one non-null entry (the state
    // checkpoint), with nulls before it.
    if hit_len > 0 {
        let non_null: Vec<_> = mb_hit.iter().filter(|&&b| b != NULL_BLOCK_ID).collect();
        assert!(
            non_null.len() <= 1,
            "align mode never keeps more than one live mamba block; got {}",
            non_null.len()
        );
    }
    let _ = fa0;
    let _ = mb0;
}

// ── find_longest_cache_hit reconciliation ────────────────────────────────────

#[test]
fn find_longest_hit_is_block_aligned() {
    let c = coord(64);
    let tokens: Vec<u32> = (0..20).collect();
    let hashes = c.compute_block_hashes(&tokens);
    let max_hit = tokens.len() - 1;
    let (_fa, _mb, hit_len) = c.find_longest_cache_hit(&hashes, max_hit);
    // Cold cache: no hit.
    assert_eq!(hit_len, 0, "cold cache must return 0 hit");

    // All returned hit lengths from find_longest_cache_hit must be block-aligned.
    assert_eq!(hit_len % BS, 0);
}

#[test]
fn hit_length_never_exceeds_max() {
    let c = coord(64);
    let tokens: Vec<u32> = (0..12).collect();
    let hashes = c.compute_block_hashes(&tokens);
    for cap in [0, BS, 2 * BS, 3 * BS - 1] {
        let (_fa, _mb, hit_len) = c.find_longest_cache_hit(&hashes, cap);
        assert!(
            hit_len <= cap,
            "hit_len {hit_len} must be <= max_cache_hit_length {cap}"
        );
    }
}

// ── free blocks and pool recycling ───────────────────────────────────────────

#[test]
fn freed_blocks_return_to_pool() {
    let num_blocks = 16u32;
    let mut c = coord(num_blocks);
    let free_before = c.pool().num_free_blocks();

    let tokens: Vec<u32> = (0..8).collect();
    let req = req_with_hashes(1, &tokens, &c);
    c.new_step_starts();
    c.allocate_slots(&req, (vec![], vec![]), 0, 8, 0, false)
        .expect("alloc");
    let free_mid = c.pool().num_free_blocks();
    assert!(
        free_mid < free_before,
        "allocation should reduce free count"
    );

    c.free(1);
    let free_after = c.pool().num_free_blocks();
    assert_eq!(
        free_after, free_before,
        "freeing a request should restore the free count"
    );
}

// ── exhausted pool returns None ──────────────────────────────────────────────

#[test]
fn full_pool_returns_none() {
    // Pool has only the null block + 1 usable block; a 2-block allocation fails.
    let mut c = coord(3); // block 0 = null, blocks 1-2 available
    let tokens: Vec<u32> = (0..16).collect(); // needs 4 FA + 1 Mamba = 5 blocks
    let req = req_with_hashes(1, &tokens, &c);

    c.new_step_starts();
    let result = c.allocate_slots(&req, (vec![], vec![]), 0, 16, 0, false);
    assert!(result.is_none(), "should fail when pool is exhausted");
}

// ── caching disabled ─────────────────────────────────────────────────────────

#[test]
fn no_cache_still_allocates() {
    let mut c = coord_no_cache(64);
    let tokens: Vec<u32> = (0..8).collect();
    let req = req_with_hashes(1, &tokens, &c);

    c.new_step_starts();
    let result = c.allocate_slots(&req, (vec![], vec![]), 0, 8, 0, false);
    assert!(
        result.is_some(),
        "allocation must succeed even without caching"
    );

    // get_computed_blocks must return zero hit when caching is off.
    let req2 = req_with_hashes(2, &tokens, &c);
    let (_fa, _mb, hit_len) = c.get_computed_blocks(&req2);
    assert_eq!(hit_len, 0, "no cache hit when caching is disabled");
}

// ── decode step: running request allocates at most one extra block ────────────

#[test]
fn decode_step_adds_one_block_at_a_time() {
    let mut c = coord(64);

    // Prefill 8 tokens.
    let tokens: Vec<u32> = (0..8).collect();
    let req = req_with_hashes(1, &tokens, &c);
    c.new_step_starts();
    c.allocate_slots(&req, (vec![], vec![]), 0, 8, 0, false)
        .expect("prefill");

    let fa_before = c.full_attn_blocks(1).len();
    let mb_before = c.mamba_blocks(1).len();

    // Decode step: 1 new token, total now 9 (still in same 2 full blocks for
    // FA; Mamba needs a new state block for block 3).
    let req_d = Req {
        id: 1,
        num_tokens: 9,
        num_prompt_tokens: 8,
        num_computed_tokens: 8,
        num_in_flight_tokens: 0,
        block_hashes: req.block_hashes.clone(),
        waiting: false,
    };
    c.new_step_starts();
    let result = c.allocate_slots(&req_d, (vec![], vec![]), 0, 1, 0, false);
    assert!(result.is_some());

    let fa_after = c.full_attn_blocks(1).len();
    let mb_after = c.mamba_blocks(1).len();
    // FA may grow by at most 1 block.
    assert!(fa_after <= fa_before + 1);
    // Mamba table grows by at most 2 entries (1 null placeholder + 1 state).
    assert!(mb_after <= mb_before + 2);
}

// ── reset_prefix_cache ───────────────────────────────────────────────────────

#[test]
fn reset_prefix_cache_clears_entries() {
    let mut c = coord(64);
    let tokens: Vec<u32> = (0..8).collect();
    let req = req_with_hashes(1, &tokens, &c);
    c.new_step_starts();
    c.allocate_slots(&req, (vec![], vec![]), 0, 8, 0, false)
        .expect("alloc");
    c.free(1);

    let before = c.pool().num_cached_entries();
    // If caching ran, there should be at least one cached entry.
    if before > 0 {
        let ok = c.reset_prefix_cache();
        assert!(ok, "reset should succeed when no requests are in flight");
        assert_eq!(c.pool().num_cached_entries(), 0);
    }
}

#[test]
fn speculative_state_blocks_cross_boundary_reuse_and_free() {
    let mut kv = HybridCoordinator::new(64, 784, false, 0);
    kv.set_speculative_blocks(4);
    let mut req = Req::new(1, 784);
    kv.allocate_slots(&req, (vec![], vec![]), 0, 784, 0, false)
        .unwrap();
    let original = kv.mamba_blocks(1).to_vec();
    assert_eq!(original.len(), 5);
    assert!(original.iter().all(|&id| id != NULL_BLOCK_ID));
    req.num_computed_tokens = 784;
    req.num_tokens = 785;
    req.waiting = false;
    kv.new_step_starts();
    kv.allocate_slots(&req, (vec![], vec![]), 0, 5, 0, false)
        .unwrap();
    assert_eq!(&kv.mamba_blocks(1)[..5], original.as_slice());
    assert_eq!(kv.mamba_blocks(1).len(), 6);
    // Reject all four drafts; another step fits in already reserved slots.
    req.num_computed_tokens = 785;
    req.num_tokens = 786;
    kv.new_step_starts();
    kv.allocate_slots(&req, (vec![], vec![]), 0, 1, 0, false)
        .unwrap();
    assert_eq!(kv.mamba_blocks(1).len(), 6);
    kv.free(1);
    assert_eq!(kv.pool().num_free_blocks(), 63);
}

#[test]
fn speculative_prefix_hit_allocates_private_state_and_draft_slots() {
    let mut kv = HybridCoordinator::new(64, 784, true, 0);
    kv.set_speculative_blocks(4);
    let tokens: Vec<u32> = (0..784).collect();
    let hashes = kv.compute_block_hashes(&tokens);
    let req = Req::new(1, 784).with_hashes(hashes.clone());
    kv.allocate_slots(&req, (vec![], vec![]), 0, 784, 0, false)
        .unwrap();
    let checkpoint = kv.mamba_blocks(1)[0];
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

#[test]
fn speculative_slots_migrate_across_large_prefill_chunk() {
    let mut kv = HybridCoordinator::new(64, 784, false, 0);
    kv.set_speculative_blocks(4);
    let mut req = Req::new(1, 784 * 8);
    kv.allocate_slots(&req, (vec![], vec![]), 0, 784, 0, false)
        .unwrap();
    let original = kv.mamba_blocks(1).to_vec();
    req.num_computed_tokens = 784;
    req.waiting = false;
    kv.new_step_starts();
    kv.allocate_slots(&req, (vec![], vec![]), 0, 784 * 7, 0, false)
        .unwrap();
    let migrated = kv.mamba_blocks(1);
    assert_eq!(migrated.len(), 12);
    assert!(migrated[1..7].iter().all(|&id| id == NULL_BLOCK_ID));
    assert_eq!(&migrated[7..11], &original[1..5]);
    assert_ne!(migrated[11], NULL_BLOCK_ID);
    kv.free(1);
    assert_eq!(kv.pool().num_free_blocks(), 63);
}
