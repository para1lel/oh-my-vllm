//! KV cache blocks and the free-block queue.
//!
//! Port of vLLM's `KVCacheBlock` + `FreeKVCacheBlockQueue`
//! (`vllm/v1/core/kv_cache_utils.py`). vLLM's docstring states the queue exists
//! because a `collections.deque` of Python objects was too slow — it hand-rolls
//! an intrusive doubly-linked list "to approach C++ deque performance". That is
//! the single clearest self-identified bottleneck in the scheduler, so it is the
//! first thing worth porting.
//!
//! # Design: index arena instead of raw pointers
//!
//! vLLM links blocks with object references; a literal Rust translation would use
//! `NonNull<KVCacheBlock>` plus `Box` ownership and a pile of `unsafe`. We use
//! `u32` arena indices instead, because block ids are *already* a dense range
//! `0..num_blocks` — the arena and the block table are the same thing. Links are
//! 4 bytes instead of 8, the whole arena is one contiguous allocation, and there
//! is no `unsafe`, so no aliasing UB to chase with miri.
//!
//! Two extra arena slots past the end hold the fake head/tail sentinels that vLLM
//! creates with `block_id=-1`. Keeping them at the end (rather than the front)
//! preserves `arena index == block_id` for every real block. Their purpose is the
//! same as in vLLM: a queued block always has both links set, so unlinking never
//! needs to special-case the ends.
//!
//! # Eviction order contract
//!
//! Copied verbatim from vLLM, because the prefix cache hit rate depends on it:
//!
//! 1. The least recently used block is at the front (LRU).
//! 2. If two blocks have the same last-accessed time (allocated by the same
//!    sequence), the one with more hash tokens (the tail of a block chain) is at
//!    the front.
//!
//! Point 2 is maintained by *callers* reversing the block order when freeing a
//! request — it is not enforced here. See [`crate::pool::BlockPool::free_blocks`].

use crate::hash::{BlockHash, BlockHashWithGroupId};

/// Sentinel for "no link". `u32::MAX` is safe to steal: a real arena index is at
/// most `num_blocks + 1`, and `num_blocks` is bounded by GPU memory.
pub(crate) const NULL: u32 = u32::MAX;

/// One KV cache block.
///
/// The block's id is its index in the arena, so it is not stored. `prev_free` /
/// `next_free` are only meaningful while the block sits in the free queue; they
/// are [`NULL`] otherwise, which is also the in-queue test (see
/// [`KVCacheBlock::is_queued`]).
#[derive(Debug, Clone)]
pub struct KVCacheBlock {
    /// Number of requests currently referencing this block. A block is evictable
    /// only at zero.
    pub ref_cnt: u32,
    /// Prefix-cache identity, set once the block is full (or via
    /// `cache_partial_block` for the trailing partial block of a request).
    block_hash: Option<BlockHashWithGroupId>,
    /// How many tokens the hash covers. Equals the block size for full blocks and
    /// is smaller for a cached partial block. Meaningless when `block_hash` is
    /// `None`.
    block_hash_num_tokens: u32,
    prev_free: u32,
    next_free: u32,
    /// The null block (id 0) is handed out as padding for un-computed slots. It is
    /// never freed and its `ref_cnt` is deliberately not maintained.
    pub is_null: bool,
}

impl KVCacheBlock {
    fn new() -> Self {
        Self {
            ref_cnt: 0,
            block_hash: None,
            block_hash_num_tokens: 0,
            prev_free: NULL,
            next_free: NULL,
            is_null: false,
        }
    }

    /// Prefix-cache hash, if this block has been cached.
    #[inline]
    pub fn block_hash(&self) -> Option<BlockHashWithGroupId> {
        self.block_hash
    }

    /// Number of tokens covered by [`Self::block_hash`]; 0 when uncached.
    #[inline]
    pub fn num_hashed_tokens(&self) -> u32 {
        if self.block_hash.is_some() {
            self.block_hash_num_tokens
        } else {
            0
        }
    }

    /// Mark this block as caching `num_tokens` tokens under `hash`.
    #[inline]
    pub fn set_block_hash(&mut self, hash: BlockHashWithGroupId, num_tokens: u32) {
        debug_assert!(num_tokens > 0, "a cached block must cover >=1 token");
        self.block_hash = Some(hash);
        self.block_hash_num_tokens = num_tokens;
    }

    /// Drop the prefix-cache identity, e.g. when the block is evicted and reused.
    #[inline]
    pub fn reset_block_hash(&mut self) {
        self.block_hash = None;
        self.block_hash_num_tokens = 0;
    }

    /// Whether the block is currently linked into the free queue.
    ///
    /// Relies on the sentinels: a linked block always has both neighbours, so a
    /// single link check is enough.
    #[inline]
    pub fn is_queued(&self) -> bool {
        self.prev_free != NULL
    }
}

/// Intrusive doubly-linked LRU queue over a contiguous arena of blocks.
///
/// Owns every block, including the two sentinels. All operations are O(1) except
/// the `_n` bulk variants, which are O(n) in the number of blocks moved, and
/// [`Self::iter_free`], which walks the list.
#[derive(Debug)]
pub struct FreeKVCacheBlockQueue {
    /// Length `num_blocks + 2`; the last two entries are the head/tail sentinels.
    blocks: Vec<KVCacheBlock>,
    num_blocks: u32,
    num_free: usize,
}

impl FreeKVCacheBlockQueue {
    /// Build an arena of `num_blocks` blocks, all free, linked in ascending id
    /// order so that the first allocations hand out low ids.
    pub fn new(num_blocks: u32) -> Self {
        let n = num_blocks as usize;
        let mut blocks = vec![KVCacheBlock::new(); n + 2];
        let head = num_blocks;
        let tail = num_blocks + 1;

        for i in 0..num_blocks {
            let idx = i as usize;
            blocks[idx].prev_free = if i == 0 {
                head
            } else {
                i - 1
            };
            blocks[idx].next_free = if i + 1 == num_blocks {
                tail
            } else {
                i + 1
            };
        }

        blocks[head as usize].prev_free = NULL;
        blocks[head as usize].next_free = if num_blocks == 0 {
            tail
        } else {
            0
        };
        blocks[tail as usize].prev_free = if num_blocks == 0 {
            head
        } else {
            num_blocks - 1
        };
        blocks[tail as usize].next_free = NULL;

        Self {
            blocks,
            num_blocks,
            num_free: n,
        }
    }

    #[inline]
    fn head(&self) -> u32 {
        self.num_blocks
    }

    #[inline]
    fn tail(&self) -> u32 {
        self.num_blocks + 1
    }

    /// Total number of blocks in the arena, sentinels excluded.
    #[inline]
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Number of blocks currently free.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_free
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_free == 0
    }

    /// Immutable access to a block by id.
    #[inline]
    pub fn block(&self, block_id: u32) -> &KVCacheBlock {
        debug_assert!(block_id < self.num_blocks, "block id out of range");
        &self.blocks[block_id as usize]
    }

    /// Mutable access to a block by id.
    ///
    /// Callers must not touch the link fields; use the queue methods for that.
    #[inline]
    pub fn block_mut(&mut self, block_id: u32) -> &mut KVCacheBlock {
        debug_assert!(block_id < self.num_blocks, "block id out of range");
        &mut self.blocks[block_id as usize]
    }

    /// Unlink `idx` from wherever it sits in the list.
    fn unlink(&mut self, idx: u32) {
        let (prev, next) = {
            let b = &self.blocks[idx as usize];
            (b.prev_free, b.next_free)
        };
        debug_assert!(
            prev != NULL && next != NULL,
            "block {idx} is not in the free queue"
        );
        self.blocks[prev as usize].next_free = next;
        self.blocks[next as usize].prev_free = prev;
        let b = &mut self.blocks[idx as usize];
        b.prev_free = NULL;
        b.next_free = NULL;
        self.num_free -= 1;
    }

    /// Link `idx` immediately after `anchor`.
    fn insert_after(&mut self, anchor: u32, idx: u32) {
        debug_assert!(idx < self.num_blocks, "cannot insert a sentinel");
        debug_assert!(
            !self.blocks[idx as usize].is_queued(),
            "block {idx} is already in the free queue (double free)"
        );
        let next = self.blocks[anchor as usize].next_free;
        debug_assert!(next != NULL, "anchor {anchor} is not linked");
        self.blocks[anchor as usize].next_free = idx;
        self.blocks[next as usize].prev_free = idx;
        let b = &mut self.blocks[idx as usize];
        b.prev_free = anchor;
        b.next_free = next;
        self.num_free += 1;
    }

    /// Link `idx` immediately before `anchor`.
    fn insert_before(&mut self, anchor: u32, idx: u32) {
        let prev = self.blocks[anchor as usize].prev_free;
        debug_assert!(prev != NULL, "anchor {anchor} is not linked");
        self.insert_after(prev, idx);
    }

    /// Pop the least recently used free block.
    pub fn popleft(&mut self) -> Option<u32> {
        let first = self.blocks[self.head() as usize].next_free;
        if first == self.tail() {
            return None;
        }
        self.unlink(first);
        Some(first)
    }

    /// Pop the `n` least recently used free blocks, front first.
    ///
    /// Returns `None` without mutating anything if fewer than `n` are free, so a
    /// failed allocation cannot leave the queue half-drained.
    pub fn popleft_n(&mut self, n: usize) -> Option<Vec<u32>> {
        if n > self.num_free {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            // Guarded by the length check above.
            out.push(self.popleft().expect("num_free accounting is wrong"));
        }
        Some(out)
    }

    /// Remove a specific block from the free queue, e.g. on a prefix-cache hit.
    ///
    /// Panics in debug builds if the block is not currently free.
    pub fn remove(&mut self, block_id: u32) {
        debug_assert!(block_id < self.num_blocks, "block id out of range");
        self.unlink(block_id);
    }

    /// Push one block to the back (most recently used position).
    pub fn append(&mut self, block_id: u32) {
        let tail = self.tail();
        self.insert_before(tail, block_id);
    }

    /// Push one block to the front (next to be evicted).
    pub fn prepend(&mut self, block_id: u32) {
        let head = self.head();
        self.insert_after(head, block_id);
    }

    /// Push blocks to the front, preserving their given order.
    ///
    /// Used for blocks with no prefix-cache value: they are reused first (LIFO),
    /// which keeps a request's freshly freed blocks hot for the next allocation.
    pub fn prepend_n(&mut self, block_ids: &[u32]) {
        let mut anchor = self.head();
        for &id in block_ids {
            self.insert_after(anchor, id);
            anchor = id;
        }
    }

    /// Push blocks to the back, preserving their given order.
    ///
    /// Used for cached blocks: they stay evictable-last (FIFO/LRU) so their prefix
    /// cache entries survive as long as possible.
    pub fn append_n(&mut self, block_ids: &[u32]) {
        let tail = self.tail();
        for &id in block_ids {
            self.insert_before(tail, id);
        }
    }

    /// Walk the free list front to back. O(n); intended for tests and metrics.
    pub fn iter_free(&self) -> FreeBlockIter<'_> {
        FreeBlockIter {
            queue: self,
            cursor: self.blocks[self.head() as usize].next_free,
        }
    }

    /// Collect the free list front to back. Test/debug helper mirroring vLLM's
    /// `get_all_free_blocks`.
    pub fn get_all_free_blocks(&self) -> Vec<u32> {
        self.iter_free().collect()
    }

    /// Hashes of every currently-free cached block, for cache-eviction accounting.
    pub fn free_cached_hashes(&self) -> impl Iterator<Item = BlockHash> + '_ {
        self.iter_free()
            .filter_map(move |id| self.block(id).block_hash().map(|h| h.block_hash))
    }

    /// Verify every structural invariant. Debug/test only — O(n).
    #[cfg(test)]
    fn check_invariants(&self) {
        let head = self.head();
        let tail = self.tail();
        assert_eq!(self.blocks[head as usize].prev_free, NULL);
        assert_eq!(self.blocks[tail as usize].next_free, NULL);

        // Forward walk: links are consistent and the length matches.
        let mut seen = 0usize;
        let mut prev = head;
        let mut cur = self.blocks[head as usize].next_free;
        while cur != tail {
            assert_ne!(cur, NULL, "list ran off the end");
            assert!(cur < self.num_blocks, "sentinel found inside the list");
            assert_eq!(
                self.blocks[cur as usize].prev_free, prev,
                "back link of {cur} is wrong"
            );
            seen += 1;
            assert!(seen <= self.num_blocks as usize, "cycle in the free list");
            prev = cur;
            cur = self.blocks[cur as usize].next_free;
        }
        assert_eq!(self.blocks[tail as usize].prev_free, prev);
        assert_eq!(seen, self.num_free, "num_free disagrees with the list");

        // Unlinked blocks must have both links cleared.
        let queued: std::collections::HashSet<u32> =
            self.get_all_free_blocks().into_iter().collect();
        for id in 0..self.num_blocks {
            let b = &self.blocks[id as usize];
            if !queued.contains(&id) {
                assert_eq!(b.prev_free, NULL, "stale prev link on unqueued {id}");
                assert_eq!(b.next_free, NULL, "stale next link on unqueued {id}");
            }
            assert_eq!(b.is_queued(), queued.contains(&id));
        }
    }
}

/// Iterator over free block ids, least recently used first.
pub struct FreeBlockIter<'a> {
    queue: &'a FreeKVCacheBlockQueue,
    cursor: u32,
}

impl Iterator for FreeBlockIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.cursor == self.queue.tail() || self.cursor == NULL {
            return None;
        }
        let id = self.cursor;
        self.cursor = self.queue.blocks[id as usize].next_free;
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_queue_is_ordered_by_id() {
        let q = FreeKVCacheBlockQueue::new(5);
        assert_eq!(q.len(), 5);
        assert_eq!(q.get_all_free_blocks(), vec![0, 1, 2, 3, 4]);
        q.check_invariants();
    }

    #[test]
    fn empty_queue_is_well_formed() {
        let mut q = FreeKVCacheBlockQueue::new(0);
        assert!(q.is_empty());
        assert_eq!(q.popleft(), None);
        assert_eq!(q.get_all_free_blocks(), Vec::<u32>::new());
        q.check_invariants();
    }

    #[test]
    fn popleft_drains_in_lru_order() {
        let mut q = FreeKVCacheBlockQueue::new(3);
        assert_eq!(q.popleft(), Some(0));
        assert_eq!(q.popleft(), Some(1));
        q.check_invariants();
        assert_eq!(q.popleft(), Some(2));
        assert_eq!(q.popleft(), None);
        assert_eq!(q.len(), 0);
        q.check_invariants();
    }

    #[test]
    fn popleft_clears_links() {
        let mut q = FreeKVCacheBlockQueue::new(3);
        let id = q.popleft().unwrap();
        assert!(!q.block(id).is_queued());
        q.check_invariants();
    }

    #[test]
    fn popleft_n_is_all_or_nothing() {
        let mut q = FreeKVCacheBlockQueue::new(4);
        assert_eq!(q.popleft_n(2), Some(vec![0, 1]));
        assert_eq!(q.len(), 2);
        // Asking for more than is free must not consume anything.
        assert_eq!(q.popleft_n(3), None);
        assert_eq!(q.len(), 2);
        assert_eq!(q.get_all_free_blocks(), vec![2, 3]);
        q.check_invariants();

        assert_eq!(q.popleft_n(2), Some(vec![2, 3]));
        assert_eq!(q.popleft_n(0), Some(vec![]));
        q.check_invariants();
    }

    #[test]
    fn remove_from_middle() {
        let mut q = FreeKVCacheBlockQueue::new(5);
        q.remove(2);
        assert_eq!(q.get_all_free_blocks(), vec![0, 1, 3, 4]);
        assert_eq!(q.len(), 4);
        q.check_invariants();
    }

    #[test]
    fn remove_from_both_ends() {
        let mut q = FreeKVCacheBlockQueue::new(4);
        q.remove(0);
        q.remove(3);
        assert_eq!(q.get_all_free_blocks(), vec![1, 2]);
        q.check_invariants();
    }

    #[test]
    fn remove_every_block_then_refill() {
        let mut q = FreeKVCacheBlockQueue::new(4);
        for id in [1, 3, 0, 2] {
            q.remove(id);
        }
        assert!(q.is_empty());
        q.check_invariants();

        q.append_n(&[2, 0]);
        assert_eq!(q.get_all_free_blocks(), vec![2, 0]);
        q.check_invariants();
    }

    #[test]
    fn append_goes_to_the_back() {
        let mut q = FreeKVCacheBlockQueue::new(3);
        let id = q.popleft().unwrap();
        q.append(id);
        assert_eq!(q.get_all_free_blocks(), vec![1, 2, 0]);
        q.check_invariants();
    }

    #[test]
    fn prepend_goes_to_the_front() {
        let mut q = FreeKVCacheBlockQueue::new(3);
        q.remove(2);
        q.prepend(2);
        assert_eq!(q.get_all_free_blocks(), vec![2, 0, 1]);
        q.check_invariants();
    }

    #[test]
    fn prepend_n_preserves_order_at_the_front() {
        let mut q = FreeKVCacheBlockQueue::new(5);
        let taken = q.popleft_n(3).unwrap();
        assert_eq!(taken, vec![0, 1, 2]);
        // LIFO reuse: the given order survives, placed ahead of everything else.
        q.prepend_n(&[2, 1, 0]);
        assert_eq!(q.get_all_free_blocks(), vec![2, 1, 0, 3, 4]);
        q.check_invariants();
    }

    #[test]
    fn append_n_preserves_order_at_the_back() {
        let mut q = FreeKVCacheBlockQueue::new(5);
        q.popleft_n(3).unwrap();
        q.append_n(&[0, 1, 2]);
        assert_eq!(q.get_all_free_blocks(), vec![3, 4, 0, 1, 2]);
        q.check_invariants();
    }

    #[test]
    fn prepend_n_and_append_n_together_match_vllm_free_order() {
        // vLLM's BlockPool::free_blocks splits a freed request into
        // "no cache value" (prepend, reused first) and "cached" (append, evicted
        // last), then flushes both. Reproduce that on a drained queue.
        let mut q = FreeKVCacheBlockQueue::new(6);
        q.popleft_n(6).unwrap();
        assert!(q.is_empty());

        let evict_first = [5u32, 4];
        let evict_last = [0u32, 1, 2];
        q.prepend_n(&evict_first);
        q.append_n(&evict_last);

        assert_eq!(q.get_all_free_blocks(), vec![5, 4, 0, 1, 2]);
        q.check_invariants();

        // The uncached blocks must be the ones handed back out first.
        assert_eq!(q.popleft(), Some(5));
        assert_eq!(q.popleft(), Some(4));
        q.check_invariants();
    }

    #[test]
    fn empty_bulk_ops_are_noops() {
        let mut q = FreeKVCacheBlockQueue::new(2);
        q.prepend_n(&[]);
        q.append_n(&[]);
        assert_eq!(q.get_all_free_blocks(), vec![0, 1]);
        q.check_invariants();
    }

    #[test]
    fn hash_roundtrip_on_a_block() {
        let mut q = FreeKVCacheBlockQueue::new(2);
        assert!(q.block(0).block_hash().is_none());
        assert_eq!(q.block(0).num_hashed_tokens(), 0);

        let h = BlockHashWithGroupId::new(BlockHash(0xabcd), 1);
        q.block_mut(0).set_block_hash(h, 16);
        assert_eq!(q.block(0).block_hash(), Some(h));
        assert_eq!(q.block(0).num_hashed_tokens(), 16);

        q.block_mut(0).reset_block_hash();
        assert!(q.block(0).block_hash().is_none());
        assert_eq!(q.block(0).num_hashed_tokens(), 0);
    }

    #[test]
    fn free_cached_hashes_reports_only_cached_free_blocks() {
        let mut q = FreeKVCacheBlockQueue::new(3);
        let h = BlockHashWithGroupId::new(BlockHash(7), 0);
        q.block_mut(1).set_block_hash(h, 16);
        // Block 2 is cached but allocated, so it must not show up.
        q.block_mut(2)
            .set_block_hash(BlockHashWithGroupId::new(BlockHash(9), 0), 16);
        q.remove(2);

        let hashes: Vec<BlockHash> = q.free_cached_hashes().collect();
        assert_eq!(hashes, vec![BlockHash(7)]);
    }

    #[test]
    fn churn_keeps_invariants() {
        // Deterministic pseudo-random churn: allocate/free in a mixed pattern and
        // assert the list never corrupts.
        let mut q = FreeKVCacheBlockQueue::new(32);
        let mut held: Vec<u32> = Vec::new();
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for step in 0..2000 {
            let take = next() % 3 != 0;
            if take && !q.is_empty() {
                let n = (next() % 4 + 1) as usize;
                let n = n.min(q.len());
                if let Some(ids) = q.popleft_n(n) {
                    held.extend(ids);
                }
            } else if !held.is_empty() {
                let n = ((next() % 4 + 1) as usize).min(held.len());
                let give: Vec<u32> = held.drain(held.len() - n..).collect();
                if next() % 2 == 0 {
                    q.prepend_n(&give);
                } else {
                    q.append_n(&give);
                }
            }
            if step % 50 == 0 {
                q.check_invariants();
            }
            assert_eq!(q.len() + held.len(), 32, "blocks leaked at step {step}");
        }
        q.check_invariants();
    }
}
