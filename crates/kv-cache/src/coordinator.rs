use crate::group::{BlocksNeeded, CacheRequest, GroupKind, GroupManager, RequestId};
use crate::hash::{BlockHash, BlockHashWithGroupId, hash_block_tokens, hash_request_tokens};
use crate::pool::BlockPool;
pub use crate::pool::NULL_BLOCK_ID;
use rustc_hash::FxHashMap;

pub const DEFAULT_WATERMARK_BLOCKS: usize = 0;

pub struct HybridCoordinator {
    full_attn: GroupManager,
    mamba: GroupManager,
    pool: BlockPool,
    mamba_pool: Option<BlockPool>,
    block_size: usize,
    watermark_blocks: usize,
    copies: FxHashMap<RequestId, Vec<(u32, u32, u32)>>,
    restores: FxHashMap<RequestId, u32>,
    checkpoints: FxHashMap<RequestId, (usize, u32)>,
}

impl HybridCoordinator {
    pub fn empty_like(&self) -> Self {
        let mut replacement = Self::new(
            self.pool.num_blocks(),
            self.block_size,
            self.pool.enable_caching(),
            self.watermark_blocks,
        );
        if let Some(pool) = &self.mamba_pool {
            replacement = replacement.with_mamba_capacity(pool.num_blocks());
        }
        replacement.set_speculative_blocks(self.mamba.speculative_blocks());
        replacement
    }

    pub fn resource_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mamba = self.mamba_pool.as_ref().unwrap_or(&self.pool);
        (
            self.pool.num_blocks() as usize,
            self.pool.num_free_blocks(),
            self.pool.num_cached_entries(),
            mamba.num_blocks() as usize,
            mamba.num_free_blocks(),
            mamba.num_cached_entries(),
        )
    }

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
            copies: FxHashMap::default(),
            restores: FxHashMap::default(),
            checkpoints: FxHashMap::default(),
        }
    }

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

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    pub fn mamba_pool_capacity(&self) -> Option<u32> {
        self.mamba_pool.as_ref().map(BlockPool::num_blocks)
    }

    pub fn set_speculative_blocks(&mut self, count: usize) {
        self.mamba.set_speculative_blocks(count);
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn partial_checkpoint(&self, request: &impl CacheRequest) -> Option<usize> {
        let end = request.num_prompt_tokens().saturating_sub(1);
        let fa_required = (end + 1 + self.mamba.speculative_blocks()).div_ceil(self.block_size) + 1;
        let state_required = 2 + self.mamba.speculative_blocks();
        let fits = if let Some(pool) = &self.mamba_pool {
            fa_required < self.pool.num_blocks() as usize
                && state_required < pool.num_blocks() as usize
        } else {
            fa_required + state_required < self.pool.num_blocks() as usize
        };
        (self.pool.enable_caching()
            && request.record_partial_prefix()
            && fits
            && !request.num_prompt_tokens().is_multiple_of(self.block_size)
            && !end.is_multiple_of(self.block_size))
        .then_some(end)
    }

    pub fn block_copies(&self, request_id: RequestId) -> Vec<(u32, u32, u32)> {
        self.copies.get(&request_id).cloned().unwrap_or_default()
    }

    pub fn prefix_state_slot(&self, request_id: RequestId) -> Option<u32> {
        self.restores.get(&request_id).copied()
    }

    pub fn cache_checkpoint(&self, request_id: RequestId) -> Option<(usize, u32)> {
        self.checkpoints.get(&request_id).copied()
    }

    pub fn full_attn_blocks(&self, request_id: RequestId) -> &[u32] {
        self.full_attn.blocks(request_id)
    }

    pub fn mamba_blocks(&self, request_id: RequestId) -> &[u32] {
        self.mamba.blocks(request_id)
    }

    pub fn new_step_starts(&mut self) {
        self.full_attn.new_step_starts();
        self.mamba.new_step_starts();
    }

    pub fn find_longest_cache_hit(
        &self,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let (fa_blocks, fa_len) =
            self.full_attn
                .find_longest_cache_hit(block_hashes, max_cache_hit_length, &self.pool);

        let (mb_blocks, mb_len) = self.mamba.find_longest_cache_hit(
            block_hashes,
            fa_len,
            self.mamba_pool.as_ref().unwrap_or(&self.pool),
        );

        let hit_length = mb_len.min(fa_len);
        let fa_num_blocks = hit_length.div_ceil(self.block_size);
        let mut fa_out = fa_blocks;
        fa_out.truncate(fa_num_blocks);

        (fa_out, mb_blocks, hit_length)
    }

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
        let num_tokens_need_slot = num_tokens_main_model + num_lookahead_tokens;

        let watermark = if has_scheduled_reqs && request.is_waiting_or_preempted() {
            self.watermark_blocks
        } else {
            0
        };

        let processed = total_computed.saturating_sub(request.num_in_flight_tokens());
        self.full_attn
            .remove_skipped_blocks(rid, processed, &mut self.pool);
        self.mamba.remove_skipped_blocks(
            rid,
            processed,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );

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

        let fa_partial =
            self.full_attn
                .partial_source(rid, &new_computed_blocks.0, total_computed, &self.pool);
        let mb_partial = self.mamba.partial_source(
            rid,
            &new_computed_blocks.1,
            total_computed,
            self.mamba_pool.as_ref().unwrap_or(&self.pool),
        );
        let fa_count = match fa_needed {
            BlocksNeeded::DeferToNextStep => return None,
            BlocksNeeded::Blocks(n) => n + usize::from(fa_partial.is_some()),
        };
        let mb_count = match mb_needed {
            BlocksNeeded::DeferToNextStep => return None,
            BlocksNeeded::Blocks(n) => n + usize::from(mb_partial.is_some()),
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

        let (fa_hits, mb_hits) = new_computed_blocks;
        let has_new_computed = !fa_hits.is_empty() || !mb_hits.is_empty();
        let is_new_request = self.full_attn.blocks(rid).is_empty();
        if has_new_computed || is_new_request {
            let local_computed_tokens = total_computed;
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

        if fa_partial.is_some() {
            let (source, destination) =
                self.full_attn
                    .replace_partial(rid, total_computed, &mut self.pool);
            self.copies.insert(
                rid,
                vec![(
                    source,
                    destination,
                    (total_computed % self.block_size) as u32,
                )],
            );
        }
        if mb_partial.is_some() {
            let (source, _) = self.mamba.replace_partial(
                rid,
                total_computed,
                self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
            );
            self.restores.insert(rid, source);
        }

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

        let num_tokens_to_cache = (total_computed + num_new_tokens).min(request.num_tokens());
        let cacheable = align_down(num_tokens_to_cache, self.block_size);
        self.full_attn
            .cache_blocks(request, cacheable, &mut self.pool);
        self.mamba.cache_blocks(
            request,
            cacheable,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );

        if self.partial_checkpoint(request) == Some(num_tokens_main_model) {
            let hash = partial_hash(request, num_tokens_main_model, self.block_size);
            self.full_attn
                .stage_partial(rid, num_tokens_main_model, hash, &mut self.pool);
            let slot = self.mamba.stage_partial(
                rid,
                num_tokens_main_model,
                hash,
                self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
            );
            self.checkpoints.insert(rid, (num_tokens_main_model, slot));
        }
        Some((new_fa, new_mb))
    }

    pub fn commit_cache(&mut self, request: &impl CacheRequest) {
        let rid = request.request_id();
        self.copies.remove(&rid);
        self.restores.remove(&rid);
        self.checkpoints.remove(&rid);
        let computed = request.num_computed_tokens().min(request.num_tokens());
        self.full_attn.commit_cache(rid, computed, &mut self.pool);
        self.mamba.commit_cache(
            rid,
            computed,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );
    }

    pub fn free(&mut self, request_id: RequestId) {
        self.copies.remove(&request_id);
        self.restores.remove(&request_id);
        self.checkpoints.remove(&request_id);
        self.full_attn.free(request_id, &mut self.pool);
        self.mamba.free(
            request_id,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        let fa = self.pool.reset_prefix_cache();
        let mamba = self
            .mamba_pool
            .as_mut()
            .is_none_or(BlockPool::reset_prefix_cache);
        fa && mamba
    }

    pub fn compute_block_hashes(&self, token_ids: &[u32]) -> Vec<BlockHash> {
        hash_request_tokens(None, token_ids, self.block_size, &[])
    }

    pub fn extend_block_hashes(&self, token_ids: &[u32], hashes: &mut Vec<BlockHash>) {
        let first_unhashed = hashes
            .len()
            .checked_mul(self.block_size)
            .expect("hashed token count overflow");
        assert!(
            first_unhashed <= token_ids.len(),
            "token history cannot shrink below its hashed prefix"
        );
        for tokens in token_ids[first_unhashed..].chunks_exact(self.block_size) {
            hashes.push(hash_block_tokens(hashes.last().copied(), tokens, &[]));
        }
    }
}

#[inline]
fn align_down(n: usize, alignment: usize) -> usize {
    n / alignment * alignment
}

impl HybridCoordinator {
    pub fn get_computed_blocks(&self, request: &impl CacheRequest) -> (Vec<u32>, Vec<u32>, usize) {
        if !self.pool.enable_caching() {
            return (Vec::new(), Vec::new(), 0);
        }
        let max_hit = request.num_tokens().saturating_sub(1);
        let (mut fa, mut mb, mut hit_len) =
            self.find_longest_cache_hit(request.block_hashes(), max_hit);
        let (full_fa, full_length) =
            self.full_attn
                .find_longest_cache_hit(request.block_hashes(), max_hit, &self.pool);
        let pool = self.mamba_pool.as_ref().unwrap_or(&self.pool);
        for (state, end) in pool.partial_blocks(1, self.block_size) {
            let index = end / self.block_size;
            if end <= hit_len || end > max_hit || index * self.block_size > full_length {
                continue;
            }
            let hash = partial_hash(request, end, self.block_size);
            if pool.block(state).block_hash() != Some(BlockHashWithGroupId::new(hash, 1)) {
                continue;
            }
            if let Some(tail) = self
                .pool
                .get_cached_block(BlockHashWithGroupId::new(hash, 0))
            {
                fa = full_fa[..index].to_vec();
                fa.push(tail);
                mb = vec![NULL_BLOCK_ID; index];
                mb.push(state);
                hit_len = end;
            }
        }
        (fa, mb, hit_len)
    }
}

fn partial_hash(request: &impl CacheRequest, end: usize, block_size: usize) -> BlockHash {
    let index = end / block_size;
    let parent = if index == 0 {
        request.cache_salt()
    } else {
        Some(request.block_hashes()[index - 1])
    };
    hash_block_tokens(parent, &request.token_ids()[index * block_size..end], &[])
}
