use rustc_hash::{FxHashMap, FxHashSet};

use crate::hash::{BlockHash, BlockHashWithGroupId};
use crate::pool::{BlockPool, NULL_BLOCK_ID};

pub type RequestId = u64;

pub trait CacheRequest {
    fn request_id(&self) -> RequestId;

    fn num_tokens(&self) -> usize;

    fn num_prompt_tokens(&self) -> usize;

    fn num_computed_tokens(&self) -> usize;

    fn num_in_flight_tokens(&self) -> usize;

    fn block_hashes(&self) -> &[BlockHash];

    fn token_ids(&self) -> &[u32];

    fn cache_salt(&self) -> Option<BlockHash> {
        None
    }

    fn record_partial_prefix(&self) -> bool {
        true
    }

    fn is_waiting_or_preempted(&self) -> bool;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroupKind {
    FullAttention,
    MambaAlign,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlocksNeeded {
    Blocks(usize),
    DeferToNextStep,
}

pub struct GroupManager {
    kind: GroupKind,
    group_id: u32,
    block_size: usize,
    enable_caching: bool,
    num_speculative_blocks: usize,
    req_to_blocks: FxHashMap<RequestId, Vec<u32>>,
    num_cached_block: FxHashMap<RequestId, usize>,
    allocated_block_reqs: FxHashSet<RequestId>,
    last_state_block_idx: FxHashMap<RequestId, usize>,
    cached_blocks_this_step: FxHashSet<BlockHashWithGroupId>,
    pending_cache: FxHashMap<RequestId, Vec<(usize, u32)>>,
    held_sources: FxHashMap<RequestId, Vec<u32>>,
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
            pending_cache: FxHashMap::default(),
            held_sources: FxHashMap::default(),
        }
    }

    pub fn set_speculative_blocks(&mut self, count: usize) {
        self.num_speculative_blocks = count;
    }

    pub fn speculative_blocks(&self) -> usize {
        self.num_speculative_blocks
    }

    #[inline]
    pub fn kind(&self) -> GroupKind {
        self.kind
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn blocks(&self, request_id: RequestId) -> &[u32] {
        self.req_to_blocks
            .get(&request_id)
            .map_or(&[][..], Vec::as_slice)
    }

    #[inline]
    fn num_skipped_tokens(&self, num_computed_tokens: usize) -> usize {
        match self.kind {
            GroupKind::FullAttention => 0,
            GroupKind::MambaAlign => num_computed_tokens.saturating_sub(1),
        }
    }

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
            if let Some(&last) = new_computed_blocks.last()
                && last != NULL_BLOCK_ID
                && let Some(h) = pool.block(last).block_hash()
                && self.cached_blocks_this_step.contains(&h)
            {
                return BlocksNeeded::DeferToNextStep;
            }
            let num_required =
                cdiv(num_tokens_main_model, self.block_size) + self.num_speculative_blocks;
            let num_new = num_required as isize
                - new_computed_blocks.len() as isize
                - num_req_blocks as isize;
            let num_new = if num_new > 0 {
                if self.allocated_block_reqs.contains(&request_id) {
                    1
                } else {
                    (num_new as usize).min(1 + self.num_speculative_blocks)
                }
            } else {
                0
            };
            return BlocksNeeded::Blocks(num_new + num_evictable(new_computed_blocks, pool));
        }

        let num_required = cdiv(num_tokens, self.block_size);
        if self.num_cached_block.contains_key(&request_id) {
            assert!(new_computed_blocks.is_empty());
            return BlocksNeeded::Blocks(num_required.saturating_sub(num_req_blocks));
        }

        assert_eq!(
            self.num_skipped_tokens(total_computed_tokens),
            0,
            "full attention never skips"
        );
        let num_local_computed_blocks = new_computed_blocks.len() + num_req_blocks;
        let num_new = num_required.saturating_sub(num_local_computed_blocks);
        BlocksNeeded::Blocks(num_new + num_evictable(new_computed_blocks, pool))
    }

    pub fn add_local_computed_blocks(
        &mut self,
        request_id: RequestId,
        new_computed_blocks: &[u32],
        num_local_computed_tokens: usize,
        pool: &mut BlockPool,
    ) {
        assert!(self.blocks(request_id).is_empty());
        let num_skipped_blocks =
            self.num_skipped_tokens(num_local_computed_tokens) / self.block_size;
        let kept = &new_computed_blocks[num_skipped_blocks.min(new_computed_blocks.len())..];

        if self.enable_caching {
            pool.touch(kept);
        } else {
            assert!(kept.iter().all(|&b| b == NULL_BLOCK_ID));
        }

        let blocks = self.req_to_blocks.entry(request_id).or_default();
        blocks.extend(std::iter::repeat_n(NULL_BLOCK_ID, num_skipped_blocks));
        blocks.extend_from_slice(kept);
        self.num_cached_block
            .insert(request_id, num_local_computed_tokens / self.block_size);
    }

    pub fn partial_source(
        &self,
        request_id: RequestId,
        hits: &[u32],
        computed: usize,
        pool: &BlockPool,
    ) -> Option<u32> {
        if computed.is_multiple_of(self.block_size) {
            return None;
        }
        let blocks = if self.blocks(request_id).is_empty() {
            hits
        } else {
            self.blocks(request_id)
        };
        let id = *blocks.get(computed / self.block_size)?;
        let block = pool.block(id);
        (id != NULL_BLOCK_ID
            && block.cache_is_committed()
            && block.num_hashed_tokens() as usize == computed)
            .then_some(id)
    }

    pub fn replace_partial(
        &mut self,
        request_id: RequestId,
        computed: usize,
        pool: &mut BlockPool,
    ) -> (u32, u32) {
        let destination = pool.get_new_blocks(1).expect("capacity was checked")[0];
        let blocks = self
            .req_to_blocks
            .get_mut(&request_id)
            .expect("request exists");
        let source = std::mem::replace(&mut blocks[computed / self.block_size], destination);
        self.held_sources
            .entry(request_id)
            .or_default()
            .push(source);
        (source, destination)
    }

    pub fn stage_partial(
        &mut self,
        request_id: RequestId,
        end: usize,
        hash: BlockHash,
        pool: &mut BlockPool,
    ) -> u32 {
        let id = self.blocks(request_id)[end / self.block_size];
        assert_ne!(id, NULL_BLOCK_ID);
        pool.stage_cache_block(
            id,
            BlockHashWithGroupId::new(hash, self.group_id),
            end as u32,
        );
        self.pending_cache
            .entry(request_id)
            .or_default()
            .push((end, id));
        id
    }

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
                    self.allocated_block_reqs.insert(request_id);
                    return Vec::new();
                }
                if prev_len > 0 {
                    let offset = if self.allocated_block_reqs.contains(&request_id) {
                        self.num_speculative_blocks
                    } else {
                        0
                    };
                    self.last_state_block_idx
                        .insert(request_id, prev_len - 1 - offset);
                }

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
                assert!(num_new <= self.num_speculative_blocks + 1);
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
        assert!(num_full <= hashes.len());
        assert!(num_full <= self.blocks(rid).len());

        let to_cache: Vec<(usize, u32)> = self.blocks(rid)[num_cached..num_full]
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, id)| id != NULL_BLOCK_ID)
            .map(|(i, id)| (num_cached + i, id))
            .collect();

        for &(idx, id) in &to_cache {
            let key = BlockHashWithGroupId::new(hashes[idx], self.group_id);
            pool.stage_cache_block(id, key, ((idx + 1) * self.block_size) as u32);
            if self.kind == GroupKind::MambaAlign {
                self.cached_blocks_this_step.insert(key);
            }
        }
        self.pending_cache.entry(rid).or_default().extend(
            to_cache
                .into_iter()
                .map(|(index, id)| ((index + 1) * self.block_size, id)),
        );
        self.num_cached_block.insert(rid, num_full);
    }

    pub fn commit_cache(&mut self, request_id: RequestId, computed: usize, pool: &mut BlockPool) {
        if let Some(sources) = self.held_sources.remove(&request_id) {
            pool.free_blocks(&sources);
        }
        let Some(pending) = self.pending_cache.get_mut(&request_id) else {
            return;
        };
        pending.retain(|&(end, id)| {
            if end <= computed {
                pool.commit_cache_block(id);
                false
            } else {
                true
            }
        });
        if pending.is_empty() {
            self.pending_cache.remove(&request_id);
        }
    }

    pub fn remove_skipped_blocks(
        &mut self,
        request_id: RequestId,
        processed_computed_tokens: usize,
        pool: &mut BlockPool,
    ) {
        if self.kind == GroupKind::FullAttention {
            return;
        }
        let num_skipped_tokens = self.num_skipped_tokens(processed_computed_tokens);
        if num_skipped_tokens > 0 {
            let len = self.blocks(request_id).len();
            let num_skipped_blocks = (num_skipped_tokens / self.block_size).min(len);
            self.remove_blocks_in_range(request_id, 0, num_skipped_blocks, pool);
        }

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
                continue;
            }
            freed.push(blocks[i]);
            blocks[i] = NULL_BLOCK_ID;
        }
        if !freed.is_empty() {
            pool.free_blocks(&freed);
        }
    }

    pub fn free(&mut self, request_id: RequestId, pool: &mut BlockPool) {
        self.num_cached_block.remove(&request_id);
        self.allocated_block_reqs.remove(&request_id);
        self.last_state_block_idx.remove(&request_id);
        self.pending_cache.remove(&request_id);
        if let Some(sources) = self.held_sources.remove(&request_id) {
            pool.free_blocks(&sources);
        }
        if let Some(mut blocks) = self.req_to_blocks.remove(&request_id) {
            blocks.reverse();
            pool.free_blocks(&blocks);
        }
    }

    pub fn new_step_starts(&mut self) {
        self.cached_blocks_this_step.clear();
    }
}

fn num_evictable(blocks: &[u32], pool: &BlockPool) -> usize {
    blocks
        .iter()
        .filter(|&&id| id != NULL_BLOCK_ID && pool.block(id).ref_cnt == 0)
        .count()
}
