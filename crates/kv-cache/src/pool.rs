//! Block allocation and the prefix-cache index.
//!
//! Port of vLLM's `BlockPool` + `BlockHashToBlockMap` (`vllm/v1/core/block_pool.py`).
//! The pool owns the free queue and the hash → block index, and is the only place
//! that mutates `ref_cnt`.

use rustc_hash::FxHashMap;

use crate::block::{FreeKVCacheBlockQueue, KVCacheBlock};
use crate::hash::BlockHashWithGroupId;

/// Block id reserved as the null block. Handed out as padding for slots that are
/// not yet computed; never freed, and its `ref_cnt` is deliberately not tracked.
pub const NULL_BLOCK_ID: u32 = 0;

/// Blocks registered under one hash.
///
/// Several physical blocks can share a hash: a block stays in the map while still
/// referenced, so a second request computing the same prefix can register its own
/// block under the same key. vLLM models this as a `KVCacheBlock | dict[int, ...]`
/// union to cut GC pressure from inner dicts; we keep the same shape for the
/// different reason that `One` needs no heap allocation, which is the common case.
#[derive(Debug)]
enum CachedBlocks {
    One(u32),
    Many(Vec<u32>),
}

impl CachedBlocks {
    /// The block to serve on a cache hit. Any of them is correct — they hold
    /// identical KV content — so pick the first deterministically.
    #[inline]
    fn first(&self) -> u32 {
        match self {
            Self::One(id) => *id,
            Self::Many(ids) => ids[0],
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

    /// Remove `id`; returns true when the entry is now empty and should be dropped.
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

/// Owns every KV cache block and the prefix-cache index over them.
#[derive(Debug)]
pub struct BlockPool {
    free_queue: FreeKVCacheBlockQueue,
    cached: FxHashMap<BlockHashWithGroupId, CachedBlocks>,
    enable_caching: bool,
}

impl BlockPool {
    /// Create a pool over `num_blocks` blocks.
    ///
    /// Block 0 becomes the null block: it is popped off the free queue at init and
    /// never handed back, matching vLLM.
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

    /// Total blocks including the null block.
    #[inline]
    pub fn num_blocks(&self) -> u32 {
        self.free_queue.num_blocks()
    }

    /// Blocks currently available for allocation.
    #[inline]
    pub fn num_free_blocks(&self) -> usize {
        self.free_queue.len()
    }

    /// Number of distinct prefix-cache entries.
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

    /// Look up a cached block by hash without taking a reference to it.
    ///
    /// The returned block may still be sitting in the free queue; the caller must
    /// [`Self::touch`] it to claim it.
    #[inline]
    pub fn get_cached_block(&self, hash: BlockHashWithGroupId) -> Option<u32> {
        self.cached.get(&hash).map(CachedBlocks::first)
    }

    /// Allocate `n` fresh blocks with `ref_cnt = 1`.
    ///
    /// Returns `None` if fewer than `n` blocks are free, leaving the pool
    /// untouched so a failed allocation cannot partially drain the queue. Each
    /// block returned has had any stale prefix-cache entry evicted.
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

    /// Drop `block_id`'s prefix-cache entry, if any, so the block can be reused.
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

    /// Take a reference on already-cached blocks (a prefix-cache hit).
    ///
    /// A hit block may be free (`ref_cnt == 0`); it must leave the free queue
    /// before its count goes up, or it could be evicted while in use.
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

    /// Release one reference per block, returning newly-free blocks to the queue.
    ///
    /// `ordered_blocks` must be **tail-first** (last block of the request first).
    /// vLLM pushes that responsibility onto callers, and the ordering is what
    /// implements rule 2 of the eviction contract: among blocks freed at the same
    /// time, the deepest in the chain is evicted first, so shallower — and
    /// therefore more widely shared — prefixes survive longer.
    ///
    /// Blocks with no cache value go to the front (LIFO: reused immediately, which
    /// keeps them hot); cached blocks go to the back (FIFO/LRU: their prefix
    /// entries stay usable as long as possible).
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
                if block.block_hash().is_none() || !self.enable_caching {
                    evict_first.push(id);
                } else {
                    evict_last.push(id);
                }
            }
        }

        self.free_queue.prepend_n(&evict_first);
        self.free_queue.append_n(&evict_last);
    }

    /// Register `block_id` as caching `num_tokens` tokens under `hash`.
    ///
    /// `num_tokens` equals the block size for a full block, and is smaller for the
    /// trailing partial block of a request (vLLM's `cache_partial_block`).
    pub fn cache_block(&mut self, block_id: u32, hash: BlockHashWithGroupId, num_tokens: u32) {
        if !self.enable_caching {
            return;
        }
        assert!(
            !self.free_queue.block(block_id).is_null,
            "the null block must never be cached"
        );
        // Re-caching under a different hash would orphan the old entry.
        self.maybe_evict_cached_block(block_id);
        self.free_queue
            .block_mut(block_id)
            .set_block_hash(hash, num_tokens);
        self.cached
            .entry(hash)
            .and_modify(|e| e.insert(block_id))
            .or_insert(CachedBlocks::One(block_id));
    }

    /// Drop the whole prefix cache.
    ///
    /// Refuses while any block other than the null block is referenced, because
    /// in-flight requests hold block ids whose contents would silently change.
    /// Returns whether the cache was cleared.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::BlockHash;

    fn hash(v: u64) -> BlockHashWithGroupId {
        BlockHashWithGroupId::new(BlockHash(v), 0)
    }

    #[test]
    fn null_block_is_reserved_at_init() {
        let pool = BlockPool::new(4, true);
        assert!(pool.block(NULL_BLOCK_ID).is_null);
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn cached_reference_is_not_reallocated() {
        let mut pool = BlockPool::new(4, true);
        let ids = pool.get_new_blocks(3).unwrap();
        pool.free_blocks(&ids);
        pool.touch(&[ids[0]]);
        let reused = pool.get_new_blocks(2).unwrap();
        assert!(!reused.contains(&ids[0]));
        assert!(pool.get_new_blocks(1).is_none());
    }

    #[test]
    fn allocation_skips_the_null_block() {
        let mut pool = BlockPool::new(4, true);
        let ids = pool.get_new_blocks(3).unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
        assert!(ids.iter().all(|&id| pool.block(id).ref_cnt == 1));
        assert_eq!(pool.num_free_blocks(), 0);
    }

    #[test]
    fn allocation_failure_is_atomic() {
        let mut pool = BlockPool::new(4, true);
        assert!(pool.get_new_blocks(4).is_none());
        assert_eq!(pool.num_free_blocks(), 3, "queue must be untouched");
        assert!(pool.get_new_blocks(3).is_some());
    }

    #[test]
    fn free_returns_blocks_and_clears_refs() {
        let mut pool = BlockPool::new(5, true);
        let ids = pool.get_new_blocks(3).unwrap();
        pool.free_blocks(&ids);
        assert_eq!(pool.num_free_blocks(), 4);
        assert!(ids.iter().all(|&id| pool.block(id).ref_cnt == 0));
    }

    #[test]
    fn shared_block_survives_until_the_last_reference_drops() {
        let mut pool = BlockPool::new(5, true);
        let ids = pool.get_new_blocks(1).unwrap();
        let id = ids[0];
        pool.cache_block(id, hash(1), 16);
        pool.touch(&[id]);
        assert_eq!(pool.block(id).ref_cnt, 2);

        pool.free_blocks(&[id]);
        assert_eq!(pool.block(id).ref_cnt, 1);
        assert_eq!(pool.num_free_blocks(), 3, "still held by one request");

        pool.free_blocks(&[id]);
        assert_eq!(pool.num_free_blocks(), 4);
    }

    #[test]
    fn cached_blocks_evict_after_uncached_ones() {
        let mut pool = BlockPool::new(6, true);
        let ids = pool.get_new_blocks(5).unwrap(); // 1..=5
        // Cache the first two; leave the rest without cache value.
        pool.cache_block(ids[0], hash(10), 16);
        pool.cache_block(ids[1], hash(11), 16);

        // Free tail-first, as callers must.
        let mut ordered = ids.clone();
        ordered.reverse();
        pool.free_blocks(&ordered);

        // Uncached (5,4,3 in freed order) sit ahead of cached (1,2).
        let next = pool.get_new_blocks(1).unwrap();
        assert_eq!(next, vec![5], "an uncached block must be reused first");
        assert_eq!(pool.get_cached_block(hash(10)), Some(ids[0]));
        assert_eq!(pool.get_cached_block(hash(11)), Some(ids[1]));
    }

    #[test]
    fn touch_pulls_a_free_cached_block_out_of_the_queue() {
        let mut pool = BlockPool::new(4, true);
        let id = pool.get_new_blocks(1).unwrap()[0];
        pool.cache_block(id, hash(1), 16);
        pool.free_blocks(&[id]);
        assert_eq!(pool.num_free_blocks(), 3);

        // Cache hit: the block is free but still holds valid KV.
        assert_eq!(pool.get_cached_block(hash(1)), Some(id));
        pool.touch(&[id]);
        assert_eq!(pool.block(id).ref_cnt, 1);
        assert_eq!(pool.num_free_blocks(), 2, "must leave the free queue");
    }

    #[test]
    fn touch_ignores_the_null_block() {
        let mut pool = BlockPool::new(4, true);
        pool.touch(&[NULL_BLOCK_ID]);
        assert_eq!(pool.block(NULL_BLOCK_ID).ref_cnt, 0);
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn free_ignores_the_null_block() {
        let mut pool = BlockPool::new(4, true);
        pool.free_blocks(&[NULL_BLOCK_ID]);
        assert_eq!(pool.num_free_blocks(), 3, "null block must not be queued");
    }

    #[test]
    fn reallocation_evicts_the_stale_cache_entry() {
        let mut pool = BlockPool::new(3, true);
        let id = pool.get_new_blocks(1).unwrap()[0];
        pool.cache_block(id, hash(42), 16);
        pool.free_blocks(&[id]);
        assert_eq!(pool.get_cached_block(hash(42)), Some(id));

        // Drain the queue so the cached block has to be recycled.
        let reused = pool.get_new_blocks(2).unwrap();
        assert!(reused.contains(&id));
        assert_eq!(pool.get_cached_block(hash(42)), None);
        assert!(pool.block(id).block_hash().is_none());
        assert_eq!(pool.num_cached_entries(), 0);
    }

    #[test]
    fn duplicate_hashes_track_every_block() {
        let mut pool = BlockPool::new(5, true);
        let ids = pool.get_new_blocks(2).unwrap();
        pool.cache_block(ids[0], hash(7), 16);
        pool.cache_block(ids[1], hash(7), 16);
        assert_eq!(pool.num_cached_entries(), 1);
        assert_eq!(pool.get_cached_block(hash(7)), Some(ids[0]));

        // Evicting one leaves the hash usable through the other.
        pool.free_blocks(&[ids[0]]);
        pool.get_new_blocks(pool.num_free_blocks()).unwrap();
        assert_eq!(pool.get_cached_block(hash(7)), Some(ids[1]));
    }

    #[test]
    fn caching_disabled_skips_the_index_and_reuses_lifo() {
        let mut pool = BlockPool::new(6, false);
        let ids = pool.get_new_blocks(3).unwrap();
        pool.cache_block(ids[0], hash(1), 16);
        assert_eq!(pool.num_cached_entries(), 0, "no index when caching is off");
        assert!(pool.block(ids[0]).block_hash().is_none());

        let mut ordered = ids.clone();
        ordered.reverse();
        pool.free_blocks(&ordered);
        assert_eq!(pool.get_new_blocks(1).unwrap(), vec![3]);
    }

    #[test]
    fn reset_prefix_cache_refuses_while_blocks_are_in_use() {
        let mut pool = BlockPool::new(4, true);
        let ids = pool.get_new_blocks(2).unwrap();
        pool.cache_block(ids[0], hash(5), 16);
        assert!(!pool.reset_prefix_cache());
        assert_eq!(pool.get_cached_block(hash(5)), Some(ids[0]));

        pool.free_blocks(&ids);
        assert!(pool.reset_prefix_cache());
        assert_eq!(pool.get_cached_block(hash(5)), None);
        assert_eq!(pool.num_cached_entries(), 0);
    }

    #[test]
    fn group_id_separates_hybrid_layer_caches() {
        let mut pool = BlockPool::new(5, true);
        let ids = pool.get_new_blocks(2).unwrap();
        // Same token prefix, different KV cache group (GDN vs full attention).
        let g0 = BlockHashWithGroupId::new(BlockHash(99), 0);
        let g1 = BlockHashWithGroupId::new(BlockHash(99), 1);
        pool.cache_block(ids[0], g0, 16);
        pool.cache_block(ids[1], g1, 16);
        assert_eq!(pool.num_cached_entries(), 2);
        assert_eq!(pool.get_cached_block(g0), Some(ids[0]));
        assert_eq!(pool.get_cached_block(g1), Some(ids[1]));
    }
}
