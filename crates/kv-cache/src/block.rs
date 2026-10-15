use crate::hash::{BlockHash, BlockHashWithGroupId};

pub(crate) const NULL: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub struct KVCacheBlock {
    pub ref_cnt: u32,
    block_hash: Option<BlockHashWithGroupId>,
    block_hash_num_tokens: u32,
    cache_committed: bool,
    prev_free: u32,
    next_free: u32,
    pub is_null: bool,
}

impl KVCacheBlock {
    fn new() -> Self {
        Self {
            ref_cnt: 0,
            block_hash: None,
            block_hash_num_tokens: 0,
            cache_committed: false,
            prev_free: NULL,
            next_free: NULL,
            is_null: false,
        }
    }

    #[inline]
    pub fn block_hash(&self) -> Option<BlockHashWithGroupId> {
        self.block_hash
    }

    #[inline]
    pub fn num_hashed_tokens(&self) -> u32 {
        if self.block_hash.is_some() {
            self.block_hash_num_tokens
        } else {
            0
        }
    }

    #[inline]
    pub fn set_block_hash(&mut self, hash: BlockHashWithGroupId, num_tokens: u32) {
        assert!(num_tokens > 0, "a cached block must cover >=1 token");
        self.block_hash = Some(hash);
        self.block_hash_num_tokens = num_tokens;
        self.cache_committed = true;
    }

    pub fn stage_cache(&mut self) {
        self.cache_committed = false;
    }

    pub fn commit_cache(&mut self) {
        self.cache_committed = self.block_hash.is_some();
    }

    pub fn cache_is_committed(&self) -> bool {
        self.cache_committed
    }

    #[inline]
    pub fn reset_block_hash(&mut self) {
        self.block_hash = None;
        self.block_hash_num_tokens = 0;
        self.cache_committed = false;
    }

    #[inline]
    pub fn is_queued(&self) -> bool {
        self.prev_free != NULL
    }
}

#[derive(Debug)]
pub struct FreeKVCacheBlockQueue {
    blocks: Vec<KVCacheBlock>,
    num_blocks: u32,
    num_free: usize,
}

impl FreeKVCacheBlockQueue {
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

    #[inline]
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_free
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_free == 0
    }

    #[inline]
    pub fn block(&self, block_id: u32) -> &KVCacheBlock {
        assert!(block_id < self.num_blocks, "block id out of range");
        &self.blocks[block_id as usize]
    }

    #[inline]
    pub fn block_mut(&mut self, block_id: u32) -> &mut KVCacheBlock {
        assert!(block_id < self.num_blocks, "block id out of range");
        &mut self.blocks[block_id as usize]
    }

    fn unlink(&mut self, idx: u32) {
        let (prev, next) = {
            let b = &self.blocks[idx as usize];
            (b.prev_free, b.next_free)
        };
        assert!(
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

    fn insert_after(&mut self, anchor: u32, idx: u32) {
        assert!(idx < self.num_blocks, "cannot insert a sentinel");
        assert!(
            !self.blocks[idx as usize].is_queued(),
            "block {idx} is already in the free queue (double free)"
        );
        let next = self.blocks[anchor as usize].next_free;
        assert!(next != NULL, "anchor {anchor} is not linked");
        self.blocks[anchor as usize].next_free = idx;
        self.blocks[next as usize].prev_free = idx;
        let b = &mut self.blocks[idx as usize];
        b.prev_free = anchor;
        b.next_free = next;
        self.num_free += 1;
    }

    fn insert_before(&mut self, anchor: u32, idx: u32) {
        let prev = self.blocks[anchor as usize].prev_free;
        assert!(prev != NULL, "anchor {anchor} is not linked");
        self.insert_after(prev, idx);
    }

    pub fn popleft(&mut self) -> Option<u32> {
        let first = self.blocks[self.head() as usize].next_free;
        if first == self.tail() {
            return None;
        }
        self.unlink(first);
        Some(first)
    }

    pub fn popleft_n(&mut self, n: usize) -> Option<Vec<u32>> {
        if n > self.num_free {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.popleft().expect("num_free accounting is wrong"));
        }
        Some(out)
    }

    pub fn remove(&mut self, block_id: u32) {
        assert!(block_id < self.num_blocks, "block id out of range");
        self.unlink(block_id);
    }

    pub fn append(&mut self, block_id: u32) {
        let tail = self.tail();
        self.insert_before(tail, block_id);
    }

    pub fn prepend(&mut self, block_id: u32) {
        let head = self.head();
        self.insert_after(head, block_id);
    }

    pub fn prepend_n(&mut self, block_ids: &[u32]) {
        let mut anchor = self.head();
        for &id in block_ids {
            self.insert_after(anchor, id);
            anchor = id;
        }
    }

    pub fn append_n(&mut self, block_ids: &[u32]) {
        let tail = self.tail();
        for &id in block_ids {
            self.insert_before(tail, id);
        }
    }

    pub fn iter_free(&self) -> FreeBlockIter<'_> {
        FreeBlockIter {
            queue: self,
            cursor: self.blocks[self.head() as usize].next_free,
        }
    }

    pub fn get_all_free_blocks(&self) -> Vec<u32> {
        self.iter_free().collect()
    }

    pub fn free_cached_hashes(&self) -> impl Iterator<Item = BlockHash> + '_ {
        self.iter_free()
            .filter_map(move |id| self.block(id).block_hash().map(|h| h.block_hash))
    }
}

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
