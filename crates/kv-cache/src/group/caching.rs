use crate::group::{CacheRequest, GroupKind, GroupManager, RequestId};
use crate::hash::{BlockHash, BlockHashWithGroupId};
use crate::pool::{BlockPool, NULL_BLOCK_ID};

impl GroupManager {
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
}
