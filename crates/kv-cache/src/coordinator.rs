//! Hybrid KV cache coordinator for Qwen3.8's two-group layout.
//!
//! Qwen3.8 has two KV cache groups:
//! - Group 0: 16 full-attention layers, one block per request per 784 tokens.
//! - Group 1: 48 GatedDeltaNet layers in `mamba_cache_mode = "align"`, at most
//!   one live block per request (the state checkpoint after the last token).
//!
//! vLLM's `HybridKVCacheCoordinator` runs a fixed-point loop over all groups to
//! reconcile cache hits; `is_simple_hybrid` (two groups, full attention first)
//! short-circuits to a single iteration. That simplification applies here too.
//!
//! # `allocate_slots` sequence
//!
//! The two-phase ordering follows vLLM issue #33775:
//! 1. `remove_skipped_blocks` — free Mamba blocks the scheduler no longer reads
//!    (the one two steps back).
//! 2. `get_num_blocks_to_allocate` — count what each group needs, including
//!    cache-hit blocks that are in the free queue (touching them de-queues them).
//! 3. If the total exceeds available blocks minus watermark, return `None`.
//! 4. `add_local_computed_blocks` for **all** groups before any `allocate_new_blocks`
//!    (touching the free queue in one pass first prevents double-counting).
//! 5. `allocate_new_blocks` for each group.
//! 6. `cache_blocks` — register newly-full blocks in the prefix cache.
//!
//! # Block-size invariant
//!
//! `scheduler_block_size == hash_block_size == block_size == 784` for Qwen3.8.
//! This is asserted at construction time and relied upon throughout:
//! `_align_cacheable(n) = round_down(n, 784)` with no partial-hit branches.

use crate::group::{BlocksNeeded, CacheRequest, GroupKind, GroupManager, RequestId};
use crate::hash::{BlockHash, hash_request_tokens};
use crate::pool::BlockPool;
pub use crate::pool::NULL_BLOCK_ID;

/// Watermark: keep this many blocks free for decode-step continuations.
///
/// vLLM's scheduler computes this as `(num_gpu_blocks * watermark_ratio).round()`
/// with `watermark_ratio = 0.01` by default. Callers that want to match vLLM
/// exactly should compute it from the pool size and pass it in; the default of 0
/// is safe and matches a freshly constructed pool.
pub const DEFAULT_WATERMARK_BLOCKS: usize = 0;

/// The two-group KV cache coordinator for Qwen3.8.
///
/// Owns one [`GroupManager`] per KV cache group and the shared [`BlockPool`].
pub struct HybridCoordinator {
    /// Group 0 — full attention.
    full_attn: GroupManager,
    /// Group 1 — GatedDeltaNet in align mode.
    mamba: GroupManager,
    pool: BlockPool,
    mamba_pool: Option<BlockPool>,
    /// Token-aligned boundary for caching: always a multiple of `block_size`.
    block_size: usize,
    /// Blocks reserved for already-running requests.
    watermark_blocks: usize,
}

impl HybridCoordinator {
    /// Create a coordinator for a pool of `num_blocks` blocks.
    ///
    /// `block_size` must be the same for both groups (Qwen3.8: 784).
    pub fn new(
        num_blocks: u32,
        block_size: usize,
        enable_caching: bool,
        watermark_blocks: usize,
    ) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            full_attn: GroupManager::new(GroupKind::FullAttention, 0, block_size, enable_caching),
            mamba: GroupManager::new(GroupKind::MambaAlign, 1, block_size, enable_caching),
            pool: BlockPool::new(num_blocks, enable_caching),
            mamba_pool: None,
            block_size,
            watermark_blocks,
        }
    }

    /// Allocate GDN checkpoints independently from token-proportional FA pages.
    pub fn with_mamba_capacity(mut self, blocks: u32) -> Self {
        assert!(blocks > 1, "Mamba pool needs a null and a writable slot");
        assert!(
            self.pool.num_free_blocks() + 1 == self.pool.num_blocks() as usize
                && self.pool.num_cached_entries() == 0
                && self.mamba_pool.is_none(),
            "configure Mamba capacity only on a fresh coordinator"
        );
        self.mamba_pool = Some(BlockPool::new(blocks, self.pool.enable_caching()));
        self
    }

    // ── public accessors ────────────────────────────────────────────────────

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    pub fn set_speculative_blocks(&mut self, count: usize) {
        self.mamba.set_speculative_blocks(count);
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn full_attn_blocks(&self, request_id: RequestId) -> &[u32] {
        self.full_attn.blocks(request_id)
    }

    pub fn mamba_blocks(&self, request_id: RequestId) -> &[u32] {
        self.mamba.blocks(request_id)
    }

    // ── per-step reset ──────────────────────────────────────────────────────

    /// Must be called at the start of each scheduling step, before any
    /// `allocate_slots` or `find_longest_cache_hit` calls.
    pub fn new_step_starts(&mut self) {
        self.full_attn.new_step_starts();
        self.mamba.new_step_starts();
    }

    /// Fresh logical blocks requiring worker-side zeroing before their first use.
    pub fn take_newly_allocated(&mut self) -> Vec<u32> {
        let mut blocks = self.pool.take_newly_allocated();
        if let Some(pool) = &mut self.mamba_pool {
            blocks.extend(pool.take_newly_allocated());
        }
        blocks
    }

    // ── prefix cache lookup ─────────────────────────────────────────────────

    /// Find the longest cache hit for `block_hashes`, capped at
    /// `max_cache_hit_length` tokens.
    ///
    /// Returns `(full_attn_hit_blocks, mamba_hit_blocks, hit_length)`.
    ///
    /// The hit length is always a multiple of `block_size`. Because
    /// `is_simple_hybrid` is true (full attention first, one other group),
    /// the loop converges in a single iteration: run full attention, then
    /// clamp Mamba to that length; whatever Mamba returns is the final answer,
    /// and full attention is trimmed to it.
    pub fn find_longest_cache_hit(
        &self,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        // Full attention: left-to-right prefix scan.
        let (fa_blocks, fa_len) =
            self.full_attn
                .find_longest_cache_hit(block_hashes, max_cache_hit_length, &self.pool);

        // Mamba: right-to-left checkpoint scan, bounded by the full-attention hit.
        let (mb_blocks, mb_len) = self.mamba.find_longest_cache_hit(
            block_hashes,
            fa_len,
            self.mamba_pool.as_ref().unwrap_or(&self.pool),
        );

        // Reconcile: is_simple_hybrid means one iteration is sufficient.
        // Full attention is downward-closed, so we trim it to the Mamba hit.
        let hit_length = mb_len.min(fa_len);
        let fa_num_blocks = hit_length.div_ceil(self.block_size);
        let mut fa_out = fa_blocks;
        fa_out.truncate(fa_num_blocks);

        (fa_out, mb_blocks, hit_length)
    }

    // ── block allocation ────────────────────────────────────────────────────

    /// Attempt to allocate KV cache slots for `request`.
    ///
    /// Returns `Some((new_full_attn_blocks, new_mamba_blocks))` with the block
    /// ids appended to each group's table this step, or `None` if the pool
    /// does not have enough free blocks. `None` is the preemption signal: the
    /// scheduler must free a lower-priority request and retry.
    ///
    /// Arguments:
    /// - `new_computed_blocks`: hit blocks from `find_longest_cache_hit` —
    ///   `(full_attn_hits, mamba_hits)`.
    /// - `num_new_computed_tokens`: the reconciled hit length in tokens.
    /// - `num_new_tokens`: tokens the scheduler wants to compute this step
    ///   (includes unverified draft tokens under MTP).
    /// - `num_lookahead_tokens`: lookahead from `num_tokens_with_spec`.
    /// - `total_computed_tokens`: `request.num_computed_tokens + num_new_computed_tokens`.
    /// - `has_scheduled_reqs`: whether other requests are already running this
    ///   step; controls whether the watermark applies.
    pub fn allocate_slots(
        &mut self,
        request: &impl CacheRequest,
        new_computed_blocks: (Vec<u32>, Vec<u32>),
        num_new_computed_tokens: usize,
        num_new_tokens: usize,
        num_lookahead_tokens: usize,
        has_scheduled_reqs: bool,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        let rid = request.request_id();
        let num_computed = request.num_computed_tokens();
        let total_computed = num_computed + num_new_computed_tokens;
        let num_tokens_main_model = total_computed + num_new_tokens;
        // Slots needed to cover all tokens including lookahead, clamped by
        // `max_model_len`. Lookahead is excluded for Mamba (done inside the
        // group manager), so pass the uncapped value here.
        let num_tokens_need_slot = num_tokens_main_model + num_lookahead_tokens;

        let watermark = if has_scheduled_reqs && request.is_waiting_or_preempted() {
            self.watermark_blocks
        } else {
            0
        };

        // Phase 0: free blocks that Mamba's align mode no longer reads.
        let processed = total_computed.saturating_sub(request.num_in_flight_tokens());
        self.full_attn
            .remove_skipped_blocks(rid, processed, &mut self.pool);
        self.mamba.remove_skipped_blocks(
            rid,
            processed,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );

        // Phase 1: count what each group needs.
        let fa_needed = self.full_attn.get_num_blocks_to_allocate(
            rid,
            num_tokens_need_slot,
            &new_computed_blocks.0,
            total_computed,
            num_tokens_main_model,
            &self.pool,
        );
        let mb_needed = self.mamba.get_num_blocks_to_allocate(
            rid,
            num_tokens_need_slot,
            &new_computed_blocks.1,
            total_computed,
            num_tokens_main_model,
            self.mamba_pool.as_ref().unwrap_or(&self.pool),
        );

        // A `DeferToNextStep` sentinel from Mamba means the hit block was
        // cached this step and its content is not on the GPU yet — skip this
        // request entirely.
        let fa_count = match fa_needed {
            BlocksNeeded::DeferToNextStep => return None,
            BlocksNeeded::Blocks(n) => n,
        };
        let mb_count = match mb_needed {
            BlocksNeeded::DeferToNextStep => return None,
            BlocksNeeded::Blocks(n) => n,
        };

        let required = fa_count + mb_count + watermark;
        let insufficient = if let Some(pool) = &self.mamba_pool {
            fa_count + watermark > self.pool.num_free_blocks() || mb_count > pool.num_free_blocks()
        } else {
            required > self.pool.num_free_blocks()
        };
        if insufficient {
            return None;
        }

        // Phase 2a: register prefix-cache hit blocks for all groups (touch-only,
        // no new allocations). Must run on all groups before any new-block
        // allocation so the free-queue state is consistent (vLLM issue #33775).
        //
        // Only called for first-time allocations: vLLM's coordinator guards with
        // `new_computed_blocks is not empty_kv_cache_blocks`. Running requests
        // already have their blocks registered — calling add_local_computed_blocks
        // again would violate its "blocks must be empty" invariant.
        let (fa_hits, mb_hits) = new_computed_blocks;
        let has_new_computed = !fa_hits.is_empty() || !mb_hits.is_empty();
        let is_new_request = self.full_attn.blocks(rid).is_empty();
        if has_new_computed || is_new_request {
            let local_computed_tokens = total_computed;
            // Only call add_local_computed_blocks for first-time allocations.
            if is_new_request {
                self.full_attn.add_local_computed_blocks(
                    rid,
                    &fa_hits,
                    local_computed_tokens,
                    &mut self.pool,
                );
                self.mamba.add_local_computed_blocks(
                    rid,
                    &mb_hits,
                    local_computed_tokens,
                    self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
                );
            }
        }

        // Phase 2b: allocate new blocks for each group.
        let new_fa = self.full_attn.allocate_new_blocks(
            rid,
            num_tokens_need_slot,
            num_tokens_main_model,
            &mut self.pool,
        );
        let new_mb = self.mamba.allocate_new_blocks(
            rid,
            num_tokens_need_slot,
            num_tokens_main_model,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );

        // Phase 3: register newly-full blocks in the prefix cache. Cap at
        // `request.num_tokens` so unverified draft tokens are excluded.
        let num_tokens_to_cache = (total_computed + num_new_tokens).min(request.num_tokens());
        let cacheable = align_down(num_tokens_to_cache, self.block_size);
        self.full_attn
            .cache_blocks(request, cacheable, &mut self.pool);
        self.mamba.cache_blocks(
            request,
            cacheable,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );

        Some((new_fa, new_mb))
    }

    // ── request teardown ────────────────────────────────────────────────────

    /// Free all blocks held for `request_id`.
    pub fn free(&mut self, request_id: RequestId) {
        self.full_attn.free(request_id, &mut self.pool);
        self.mamba.free(
            request_id,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );
    }

    /// Reset the prefix cache, releasing all cached block entries.
    ///
    /// Returns `false` if there are any in-flight requests (matching vLLM's
    /// guard: resetting while requests hold blocks would corrupt their tables).
    pub fn reset_prefix_cache(&mut self) -> bool {
        let fa = self.pool.reset_prefix_cache();
        let mamba = self
            .mamba_pool
            .as_mut()
            .is_none_or(BlockPool::reset_prefix_cache);
        fa && mamba
    }

    // ── hash helpers ────────────────────────────────────────────────────────

    /// Compute the chain hashes for `token_ids` at `self.block_size`.
    ///
    /// This is a thin forwarding wrapper so callers need not import `hash`.
    pub fn compute_block_hashes(&self, token_ids: &[u32]) -> Vec<BlockHash> {
        hash_request_tokens(None, token_ids, self.block_size, &[])
    }
}

#[inline]
fn align_down(n: usize, alignment: usize) -> usize {
    n / alignment * alignment
}

// ── get_computed_blocks helper ───────────────────────────────────────────────

impl HybridCoordinator {
    /// High-level entry point called by the scheduler before `allocate_slots`.
    ///
    /// Returns `(fa_hit_blocks, mamba_hit_blocks, num_new_computed_tokens)`.
    /// `num_new_computed_tokens` is the reconciled prefix-cache hit length in
    /// tokens; it is always `>= 0` and `< request.num_tokens()`.
    ///
    /// When prefix caching is disabled the pool returns no hits, so both hit
    /// vectors are empty and `num_new_computed_tokens` is 0.
    pub fn get_computed_blocks(&self, request: &impl CacheRequest) -> (Vec<u32>, Vec<u32>, usize) {
        if !self.pool.enable_caching() {
            return (Vec::new(), Vec::new(), 0);
        }
        // Cap at num_tokens - 1: even a full prefix hit must recompute the
        // last token to obtain logits.
        let max_hit = request.num_tokens().saturating_sub(1);
        let (fa, mb, hit_len) = self.find_longest_cache_hit(request.block_hashes(), max_hit);
        (fa, mb, hit_len)
    }
}
