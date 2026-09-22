//! Per-group KV cache managers — the port of vLLM's `SingleTypeKVCacheManager`
//! subclasses, specialized to the two groups Qwen3.8 actually has.
//!
//! vLLM dispatches one manager per KV cache group through a class hierarchy
//! (`FullAttentionManager`, `MambaManager`, `SlidingWindowManager`, …). Qwen3.8
//! has exactly two groups — 16 full-attention layers and 48 GatedDeltaNet layers
//! in `mamba_cache_mode == "align"` — so this is an enum, not a trait object:
//! [`GroupKind`] selects between the two behaviours and the shared bookkeeping
//! lives in one struct.
//!
//! # What this port leaves out, and why
//!
//! The decisive fact is that for Qwen3.8 the scheduler block size, the hash block
//! size and both groups' block sizes are all 784 tokens. `mamba_cache_mode` is
//! `align`, so `Platform.check_and_update_config` copies `cache_config.block_size`
//! into `cache_config.mamba_block_size`; `resolve_kv_cache_block_sizes` then takes
//! `lcm` of the group block sizes for the scheduler and `gcd` for the hash unit,
//! and both reduce to 784. `HybridKVCacheCoordinator` derives
//! `enable_partial_hash_hits` from `block_size > hash_block_size`, so it is
//! `False` here. That kills, provably, every one of:
//!
//! - fine-grained hash probing inside a block (both finders' partial branches),
//! - partial-hit copy-on-write (`_partial_hit_reqs`, `_pending_cow_copies`,
//!   `_apply_cow`, `move_block_hashes`): a hit length is always a multiple of the
//!   block size, so `_has_partial_local_hit` is never true,
//! - `_cache_partial_tail_block`, whose first statement is
//!   `if self.block_size == hash_block_size: return None`,
//! - the tail of `FullAttentionManager.cache_blocks`, guarded by the same test.
//!
//! MTP verification reserves one recurrent state slot per draft token, even
//! though the draft model has no GDN layers (see ADR-003). Ordinary decoding
//! reserves no speculative slots. `retention_interval` is `None`, so
//! `reachable_block_mask` returns `None` and caching is dense; and with
//! no KV connector there are no external computed tokens. Sliding-window,
//! cross-attention, DCP/PCP and encoder paths do not exist for this model at all.
//!
//! Debug assertions below pin the assumptions that make the omissions valid, so a
//! future configuration change fails loudly in tests instead of silently serving
//! stale KV.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::hash::{BlockHash, BlockHashWithGroupId};
use crate::pool::{BlockPool, NULL_BLOCK_ID};

/// Scheduler-assigned request handle.
///
/// vLLM keys every manager map by the client-facing request-id string. Ours is a
/// dense integer minted by the scheduler, which keeps these maps hashing a
/// register-sized key instead of a heap string; translation to and from the
/// external id happens once, at the API boundary.
pub type RequestId = u64;

/// The request state the KV cache manager reads.
///
/// A trait rather than a struct so the scheduler's `Request` stays the single
/// owner of this state — the cache manager borrows a view of it and the crates
/// need no dependency cycle.
pub trait CacheRequest {
    fn request_id(&self) -> RequestId;

    /// Prompt tokens plus tokens generated so far (vLLM's `Request.num_tokens`).
    fn num_tokens(&self) -> usize;

    fn num_prompt_tokens(&self) -> usize;

    /// Tokens whose KV is already computed and committed.
    fn num_computed_tokens(&self) -> usize;

    /// Tokens counted in `num_computed_tokens` whose forward pass has not
    /// finished yet. Their blocks must not be freed as "skipped".
    fn num_in_flight_tokens(&self) -> usize;

    /// Chain hashes of the request's full blocks, at the hash block size.
    fn block_hashes(&self) -> &[BlockHash];

    /// True for `WAITING` and `PREEMPTED`, the two states the free-block
    /// watermark applies to.
    fn is_waiting_or_preempted(&self) -> bool;
}

/// Which of Qwen3.8's two KV cache layouts a group holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroupKind {
    /// Full attention: every token's KV is kept until the request finishes, and a
    /// cache hit is the longest matching prefix.
    FullAttention,
    /// GatedDeltaNet in `align` mode: only the recurrent state after the last
    /// computed token matters, so at most one live block per request and hits
    /// match the *last* cached checkpoint rather than a prefix.
    MambaAlign,
}

/// How many blocks a group needs, or a request to retry next step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlocksNeeded {
    Blocks(usize),
    /// The hit this request wants was cached earlier in *this* scheduling step,
    /// so its contents are not on the GPU yet. vLLM signals this by returning
    /// `num_gpu_blocks + 1` — a count guaranteed to fail the capacity check;
    /// naming the case avoids the sentinel.
    DeferToNextStep,
}

pub struct GroupManager {
    kind: GroupKind,
    group_id: u32,
    block_size: usize,
    enable_caching: bool,
    num_speculative_blocks: usize,
    /// Block ids backing each request, in token order. Interior `NULL_BLOCK_ID`
    /// entries are positions whose KV is intentionally absent (Mamba skips every
    /// block but the state checkpoint).
    req_to_blocks: FxHashMap<RequestId, Vec<u32>>,
    /// Leading blocks already registered in the prefix cache.
    num_cached_block: FxHashMap<RequestId, usize>,
    /// `MambaAlign`: requests that have been through `allocate_new_blocks` at
    /// least once, distinguishing a decode step from a first prefill.
    allocated_block_reqs: FxHashSet<RequestId>,
    /// `MambaAlign`: index of the block holding the state from two steps ago,
    /// freeable once the current step's copy has run.
    last_state_block_idx: FxHashMap<RequestId, usize>,
    /// `MambaAlign`: hashes cached during the current step. A request may not hit
    /// these, because the block's contents are written by a forward pass that has
    /// not run yet.
    cached_blocks_this_step: FxHashSet<BlockHashWithGroupId>,
}

#[inline]
fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

impl GroupManager {
    pub fn new(kind: GroupKind, group_id: u32, block_size: usize, enable_caching: bool) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            kind,
            group_id,
            block_size,
            enable_caching,
            num_speculative_blocks: 0,
            req_to_blocks: FxHashMap::default(),
            num_cached_block: FxHashMap::default(),
            allocated_block_reqs: FxHashSet::default(),
            last_state_block_idx: FxHashMap::default(),
            cached_blocks_this_step: FxHashSet::default(),
        }
    }

    pub fn set_speculative_blocks(&mut self, count: usize) {
        self.num_speculative_blocks = count;
    }

    #[inline]
    pub fn kind(&self) -> GroupKind {
        self.kind
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Blocks currently assigned to `request_id`, including null placeholders.
    pub fn blocks(&self, request_id: RequestId) -> &[u32] {
        self.req_to_blocks
            .get(&request_id)
            .map_or(&[][..], Vec::as_slice)
    }

    /// Tokens at the head of the sequence whose KV this group no longer reads.
    ///
    /// Full attention never skips; Mamba keeps only the state after the last
    /// computed token, so everything before it is dead.
    #[inline]
    fn num_skipped_tokens(&self, num_computed_tokens: usize) -> usize {
        match self.kind {
            GroupKind::FullAttention => 0,
            GroupKind::MambaAlign => num_computed_tokens.saturating_sub(1),
        }
    }

    /// Longest cache hit this group can serve for `block_hashes`, capped at
    /// `max_length` tokens.
    ///
    /// Returns the hit blocks (one entry per block, `NULL_BLOCK_ID` where the
    /// group holds nothing) and the hit length in tokens, always a multiple of
    /// the block size.
    pub fn find_longest_cache_hit(
        &self,
        block_hashes: &[BlockHash],
        max_length: usize,
        pool: &BlockPool,
    ) -> (Vec<u32>, usize) {
        if !self.enable_caching {
            return (Vec::new(), 0);
        }
        let max_num_blocks = (max_length / self.block_size).min(block_hashes.len());
        match self.kind {
            // Left to right, stop at the first miss: a full-attention hit is a
            // prefix, and chain hashing means a miss at block i makes every
            // later block unreachable anyway.
            GroupKind::FullAttention => {
                let mut hit = Vec::new();
                for &h in &block_hashes[..max_num_blocks] {
                    match pool.get_cached_block(BlockHashWithGroupId::new(h, self.group_id)) {
                        Some(id) => hit.push(id),
                        None => break,
                    }
                }
                let hit_length = hit.len() * self.block_size;
                (hit, hit_length)
            }
            // Right to left, stop at the first hit: the recurrent state at token
            // k summarizes everything before it, so the *deepest* cached
            // checkpoint is the best one and the blocks it skipped stay null.
            GroupKind::MambaAlign => {
                for i in (0..max_num_blocks).rev() {
                    let key = BlockHashWithGroupId::new(block_hashes[i], self.group_id);
                    if let Some(id) = pool.get_cached_block(key) {
                        let mut hit = vec![NULL_BLOCK_ID; i];
                        hit.push(id);
                        return (hit, (i + 1) * self.block_size);
                    }
                }
                (Vec::new(), 0)
            }
        }
    }

    /// Blocks this group must take from the pool to hold `num_tokens`.
    ///
    /// Counts cache-hit blocks that are sitting in the free queue: touching them
    /// pulls them out of it, so they are not available to anyone else.
    pub fn get_num_blocks_to_allocate(
        &self,
        request_id: RequestId,
        num_tokens: usize,
        new_computed_blocks: &[u32],
        total_computed_tokens: usize,
        num_tokens_main_model: usize,
        pool: &BlockPool,
    ) -> BlocksNeeded {
        let num_req_blocks = self.blocks(request_id).len();

        if self.kind == GroupKind::MambaAlign {
            // A hit on a block cached earlier this step would read KV that the
            // GPU has not written yet.
            if let Some(&last) = new_computed_blocks.last()
                && last != NULL_BLOCK_ID
                && let Some(h) = pool.block(last).block_hash()
                && self.cached_blocks_this_step.contains(&h)
            {
                return BlocksNeeded::DeferToNextStep;
            }
            // Lookahead tokens are excluded on purpose: scheduling
            // `k * block_size + num_lookahead` tokens would break the alignment
            // the mode is named for, and MTP's draft has no GDN layers to feed.
            let num_required =
                cdiv(num_tokens_main_model, self.block_size) + self.num_speculative_blocks;
            let num_new = num_required as isize
                - new_computed_blocks.len() as isize
                - num_req_blocks as isize;
            // Only ever one block: the new state checkpoint. Everything before
            // it is null.
            let num_new = if num_new > 0 {
                if self.allocated_block_reqs.contains(&request_id) {
                    1
                } else {
                    1 + self.num_speculative_blocks
                }
            } else {
                0
            };
            return BlocksNeeded::Blocks(num_new + num_evictable(new_computed_blocks, pool));
        }

        let num_required = cdiv(num_tokens, self.block_size);
        if self.num_cached_block.contains_key(&request_id) {
            // Running request: prefix-cache lookup already happened on its first
            // pass, so there can be no new hit blocks. Under speculative decoding
            // `num_required` can be *below* what is held, when draft tokens were
            // allocated for and then rejected.
            debug_assert!(new_computed_blocks.is_empty());
            return BlocksNeeded::Blocks(num_required.saturating_sub(num_req_blocks));
        }

        debug_assert_eq!(
            self.num_skipped_tokens(total_computed_tokens),
            0,
            "full attention never skips"
        );
        let num_local_computed_blocks = new_computed_blocks.len() + num_req_blocks;
        let num_new = num_required.saturating_sub(num_local_computed_blocks);
        BlocksNeeded::Blocks(num_new + num_evictable(new_computed_blocks, pool))
    }

    /// Take a reference on prefix-cache hit blocks and record them for the
    /// request. Only ever called on a request's first allocation.
    pub fn add_local_computed_blocks(
        &mut self,
        request_id: RequestId,
        new_computed_blocks: &[u32],
        num_local_computed_tokens: usize,
        pool: &mut BlockPool,
    ) {
        debug_assert!(self.blocks(request_id).is_empty());
        debug_assert_eq!(
            num_local_computed_tokens % self.block_size,
            0,
            "hits are block-aligned here, so no partial-hit copy-on-write exists"
        );
        let num_skipped_blocks =
            self.num_skipped_tokens(num_local_computed_tokens) / self.block_size;
        // Skipped hit blocks are re-created as nulls below rather than
        // referenced: this group will never read them.
        let kept = &new_computed_blocks[num_skipped_blocks.min(new_computed_blocks.len())..];

        if self.enable_caching {
            pool.touch(kept);
        } else {
            debug_assert!(kept.iter().all(|&b| b == NULL_BLOCK_ID));
        }

        let blocks = self.req_to_blocks.entry(request_id).or_default();
        blocks.extend(std::iter::repeat_n(NULL_BLOCK_ID, num_skipped_blocks));
        blocks.extend_from_slice(kept);
        // Every block placed here already carries a hash, so `cache_blocks` must
        // start past them.
        self.num_cached_block.insert(request_id, blocks.len());
    }

    /// Pull the blocks needed to cover `num_tokens`, returning the ids appended
    /// to the request's block table this step (null placeholders included, since
    /// the worker's table must stay positionally aligned).
    pub fn allocate_new_blocks(
        &mut self,
        request_id: RequestId,
        num_tokens: usize,
        num_tokens_main_model: usize,
        pool: &mut BlockPool,
    ) -> Vec<u32> {
        match self.kind {
            GroupKind::FullAttention => {
                let num_required = cdiv(num_tokens, self.block_size);
                let have = self.blocks(request_id).len();
                if num_required <= have {
                    return Vec::new();
                }
                let new = pool
                    .get_new_blocks(num_required - have)
                    .expect("capacity was checked before allocating");
                self.req_to_blocks
                    .entry(request_id)
                    .or_default()
                    .extend_from_slice(&new);
                new
            }
            GroupKind::MambaAlign => {
                let num_required =
                    cdiv(num_tokens_main_model, self.block_size) + self.num_speculative_blocks;
                let prev_len = self.blocks(request_id).len();
                if num_required <= prev_len {
                    // Over-allocated in an earlier step; nothing to do.
                    self.allocated_block_reqs.insert(request_id);
                    return Vec::new();
                }
                if prev_len > 0 {
                    // Either the state block from the previous step, or the block
                    // a prefix hit landed on. Both become freeable once this
                    // step's state has been copied out of them.
                    let offset = if self.allocated_block_reqs.contains(&request_id) {
                        self.num_speculative_blocks
                    } else {
                        0
                    };
                    self.last_state_block_idx
                        .insert(request_id, prev_len - 1 - offset);
                }

                // Every block but the last holds no state.
                let num_skipped_blocks = num_required - self.num_speculative_blocks - 1;
                let blocks = self.req_to_blocks.entry(request_id).or_default();
                if prev_len < num_skipped_blocks {
                    blocks.extend(std::iter::repeat_n(
                        NULL_BLOCK_ID,
                        num_skipped_blocks - prev_len,
                    ));
                }
                if self.allocated_block_reqs.contains(&request_id) {
                    for idx in prev_len - self.num_speculative_blocks..prev_len {
                        if idx >= num_skipped_blocks {
                            break;
                        }
                        blocks.push(blocks[idx]);
                        blocks[idx] = NULL_BLOCK_ID;
                    }
                }
                let num_new = num_required - blocks.len();
                debug_assert!(num_new <= self.num_speculative_blocks + 1);
                let mut appended = blocks[prev_len..].to_vec();

                let new = pool
                    .get_new_blocks(num_new)
                    .expect("capacity was checked before allocating");
                self.req_to_blocks
                    .entry(request_id)
                    .or_default()
                    .extend_from_slice(&new);
                self.allocated_block_reqs.insert(request_id);
                appended.extend_from_slice(&new);
                appended
            }
        }
    }

    /// Register the request's newly-filled full blocks in the prefix cache.
    ///
    /// `num_tokens` is already rounded down to the cacheable boundary by the
    /// coordinator.
    pub fn cache_blocks(
        &mut self,
        request: &impl CacheRequest,
        num_tokens: usize,
        pool: &mut BlockPool,
    ) {
        if !self.enable_caching {
            return;
        }
        let rid = request.request_id();
        let num_cached = self.num_cached_block.get(&rid).copied().unwrap_or(0);
        let num_full = num_tokens / self.block_size;
        if num_cached >= num_full {
            return;
        }
        let hashes = request.block_hashes();
        debug_assert!(num_full <= hashes.len());
        debug_assert!(num_full <= self.blocks(rid).len());

        let to_cache: Vec<(usize, u32)> = self.blocks(rid)[num_cached..num_full]
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, id)| id != NULL_BLOCK_ID)
            .map(|(i, id)| (num_cached + i, id))
            .collect();

        for &(idx, id) in &to_cache {
            let key = BlockHashWithGroupId::new(hashes[idx], self.group_id);
            pool.cache_block(id, key, ((idx + 1) * self.block_size) as u32);
            if self.kind == GroupKind::MambaAlign {
                self.cached_blocks_this_step.insert(key);
            }
        }
        self.num_cached_block.insert(rid, num_full);
    }

    /// Free blocks the group will no longer read, replacing them with nulls.
    ///
    /// `processed_computed_tokens` counts only tokens whose forward pass has
    /// completed — freeing on the strength of in-flight tokens would hand a block
    /// away while a kernel is still reading it.
    pub fn remove_skipped_blocks(
        &mut self,
        request_id: RequestId,
        processed_computed_tokens: usize,
        pool: &mut BlockPool,
    ) {
        if self.kind == GroupKind::FullAttention {
            // Nothing is ever outside the window.
            return;
        }
        let num_skipped_tokens = self.num_skipped_tokens(processed_computed_tokens);
        if num_skipped_tokens > 0 {
            let len = self.blocks(request_id).len();
            let num_skipped_blocks = (num_skipped_tokens / self.block_size).min(len);
            self.remove_blocks_in_range(request_id, 0, num_skipped_blocks, pool);
        }

        // The block holding the state from two steps back. The previous step's
        // block is still needed: it is the source of this step's state copy.
        let Some(&idx) = self.last_state_block_idx.get(&request_id) else {
            return;
        };
        if idx + 1 >= cdiv(processed_computed_tokens, self.block_size) {
            return;
        }
        let Some(blocks) = self.req_to_blocks.get_mut(&request_id) else {
            return;
        };
        if idx < blocks.len() && blocks[idx] != NULL_BLOCK_ID {
            let victim = blocks[idx];
            blocks[idx] = NULL_BLOCK_ID;
            pool.free_blocks(&[victim]);
        }
    }

    /// Null out `[first_block, last_block)`, freeing what was there.
    ///
    /// Walks backward and stops at the first null, which both skips work already
    /// done on an earlier call and produces a tail-first list — exactly the order
    /// [`BlockPool::free_blocks`] requires.
    fn remove_blocks_in_range(
        &mut self,
        request_id: RequestId,
        first_block: usize,
        last_block: usize,
        pool: &mut BlockPool,
    ) {
        let Some(blocks) = self.req_to_blocks.get_mut(&request_id) else {
            return;
        };
        if first_block >= last_block {
            return;
        }
        let last_block = last_block.min(blocks.len());
        let mut freed = Vec::new();
        for i in (first_block..last_block).rev() {
            if blocks[i] == NULL_BLOCK_ID {
                break;
            }
            freed.push(blocks[i]);
            blocks[i] = NULL_BLOCK_ID;
        }
        if !freed.is_empty() {
            pool.free_blocks(&freed);
        }
    }

    /// Release everything held for a finished or preempted request.
    pub fn free(&mut self, request_id: RequestId, pool: &mut BlockPool) {
        self.num_cached_block.remove(&request_id);
        self.allocated_block_reqs.remove(&request_id);
        self.last_state_block_idx.remove(&request_id);
        if let Some(mut blocks) = self.req_to_blocks.remove(&request_id) {
            // Tail-first, so deep blocks are evicted before the shallow prefixes
            // that other requests are more likely to share.
            blocks.reverse();
            pool.free_blocks(&blocks);
        }
    }

    /// Clear per-step state at the start of a scheduling pass.
    pub fn new_step_starts(&mut self) {
        self.cached_blocks_this_step.clear();
    }
}

/// Hit blocks that are in the free queue: touching them removes them from it, so
/// the capacity check has to count them as spent.
fn num_evictable(blocks: &[u32], pool: &BlockPool) -> usize {
    blocks
        .iter()
        .filter(|&&id| id != NULL_BLOCK_ID && pool.block(id).ref_cnt == 0)
        .count()
}
