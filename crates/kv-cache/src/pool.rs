use rustc_hash::FxHashMap;

use crate::block::{FreeKVCacheBlockQueue, KVCacheBlock};
use crate::hash::BlockHashWithGroupId;

pub const NULL_BLOCK_ID: u32 = 0;

#[derive(Debug)]
enum CachedBlocks {
    One(u32),
    Many(Vec<u32>),
}

impl CachedBlocks {
    #[inline]
    fn first_committed(&self, queue: &FreeKVCacheBlockQueue) -> Option<u32> {
        match self {
            Self::One(id) => queue.block(*id).cache_is_committed().then_some(*id),
            Self::Many(ids) => ids
                .iter()
                .copied()
                .find(|&id| queue.block(id).cache_is_committed()),
        }
    }

    fn insert(&mut self, id: u32) {
        match self {
            Self::One(existing) => {
                if *existing != id {
                    *self = Self::Many(vec![*existing, id]);
                }
            }
            Self::Many(ids) => {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
    }

    fn remove(&mut self, id: u32) -> bool {
        match self {
            Self::One(existing) => *existing == id,
            Self::Many(ids) => {
                ids.retain(|&x| x != id);
                match ids.len() {
                    0 => true,
                    1 => {
                        *self = Self::One(ids[0]);
                        false
                    }
                    _ => false,
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct BlockPool {
    free_queue: FreeKVCacheBlockQueue,
    cached: FxHashMap<BlockHashWithGroupId, CachedBlocks>,
    enable_caching: bool,
}

impl BlockPool {
    pub fn new(num_blocks: u32, enable_caching: bool) -> Self {
        assert!(
            num_blocks > 0,
            "the pool needs at least one block for the null block"
        );
        let mut free_queue = FreeKVCacheBlockQueue::new(num_blocks);
        let null = free_queue.popleft().expect("num_blocks > 0");
        assert_eq!(null, NULL_BLOCK_ID);
        free_queue.block_mut(null).is_null = true;

        Self {
            free_queue,
            cached: FxHashMap::default(),
            enable_caching,
        }
    }

    #[inline]
    pub fn num_blocks(&self) -> u32 {
        self.free_queue.num_blocks()
    }

    #[inline]
    pub fn num_free_blocks(&self) -> usize {
        self.free_queue.len()
    }

    #[inline]
    pub fn num_cached_entries(&self) -> usize {
        self.cached.len()
    }

    #[inline]
    pub fn enable_caching(&self) -> bool {
        self.enable_caching
    }

    #[inline]
    pub fn block(&self, block_id: u32) -> &KVCacheBlock {
        self.free_queue.block(block_id)
    }

    #[inline]
    pub fn get_cached_block(&self, hash: BlockHashWithGroupId) -> Option<u32> {
        self.cached
            .get(&hash)
            .and_then(|entry| entry.first_committed(&self.free_queue))
    }

    pub fn partial_blocks(
        &self,
        group: u32,
        block_size: usize,
    ) -> impl Iterator<Item = (u32, usize)> + '_ {
        (1..self.num_blocks()).filter_map(move |id| {
            let block = self.block(id);
            let end = block.num_hashed_tokens() as usize;
            (block.cache_is_committed()
                && !end.is_multiple_of(block_size)
                && block
                    .block_hash()
                    .is_some_and(|hash| hash.group_id == group))
            .then_some((id, end))
        })
    }

    pub fn get_new_blocks(&mut self, n: usize) -> Option<Vec<u32>> {
        let ids = self.free_queue.popleft_n(n)?;
        for &id in &ids {
            if self.enable_caching {
                self.maybe_evict_cached_block(id);
            }
            let block = self.free_queue.block_mut(id);
            assert_eq!(block.ref_cnt, 0, "allocated a block that was still in use");
            block.ref_cnt = 1;
        }
        Some(ids)
    }

    fn maybe_evict_cached_block(&mut self, block_id: u32) {
        let Some(hash) = self.free_queue.block(block_id).block_hash() else {
            return;
        };
        if let Some(entry) = self.cached.get_mut(&hash)
            && entry.remove(block_id)
        {
            self.cached.remove(&hash);
        }
        self.free_queue.block_mut(block_id).reset_block_hash();
    }

    pub fn touch(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            if self.free_queue.block(id).is_null {
                continue;
            }
            if self.free_queue.block(id).ref_cnt == 0 {
                self.free_queue.remove(id);
            }
            self.free_queue.block_mut(id).ref_cnt += 1;
        }
    }

    pub fn free_blocks(&mut self, ordered_blocks: &[u32]) {
        let mut evict_first = Vec::new();
        let mut evict_last = Vec::new();

        for &id in ordered_blocks {
            let block = self.free_queue.block_mut(id);
            if block.is_null {
                continue;
            }
            assert!(block.ref_cnt > 0, "freeing block {id} with ref_cnt 0");
            block.ref_cnt -= 1;
            if block.ref_cnt == 0 {
                if !block.cache_is_committed() {
                    self.maybe_evict_cached_block(id);
                    evict_first.push(id);
                } else if block.block_hash().is_none() || !self.enable_caching {
                    evict_first.push(id);
                } else {
                    evict_last.push(id);
                }
            }
        }

        self.free_queue.prepend_n(&evict_first);
        self.free_queue.append_n(&evict_last);
    }

    pub fn cache_block(&mut self, block_id: u32, hash: BlockHashWithGroupId, num_tokens: u32) {
        if !self.enable_caching {
            return;
        }
        assert!(
            !self.free_queue.block(block_id).is_null,
            "the null block must never be cached"
        );
        self.maybe_evict_cached_block(block_id);
        self.free_queue
            .block_mut(block_id)
            .set_block_hash(hash, num_tokens);
        self.cached
            .entry(hash)
            .and_modify(|e| e.insert(block_id))
            .or_insert(CachedBlocks::One(block_id));
    }

    pub fn stage_cache_block(
        &mut self,
        block_id: u32,
        hash: BlockHashWithGroupId,
        num_tokens: u32,
    ) {
        self.cache_block(block_id, hash, num_tokens);
        if self.enable_caching {
            self.free_queue.block_mut(block_id).stage_cache();
        }
    }

    pub fn commit_cache_block(&mut self, block_id: u32) {
        self.free_queue.block_mut(block_id).commit_cache();
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        let num_used = self.num_blocks() as usize - self.free_queue.len();
        if num_used > 1 {
            tracing::warn!(
                num_used,
                "refusing to reset the prefix cache while blocks are in use"
            );
            return false;
        }
        for id in 0..self.num_blocks() {
            self.free_queue.block_mut(id).reset_block_hash();
        }
        self.cached.clear();
        true
    }
}
