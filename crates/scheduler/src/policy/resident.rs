use oh_my_vllm_kv_cache::{
    HybridCoordinator, group::RequestId, hash::BlockHash, hash::hash_request_tokens,
};

use super::{CachePolicy, PrefixHit};
use crate::{Request, SchedulerConfig};

#[derive(Debug, Clone, Default)]
pub struct ResidentState {
    pub fa_block_table: Vec<u32>,
    pub mamba_block_table: Vec<u32>,
    pub fa_block_copies: Vec<(u32, u32, u32)>,
    pub prefix_state_slot: Option<u32>,
    pub cache_checkpoint: Option<(usize, u32)>,
}

impl CachePolicy for HybridCoordinator {
    type Hit = (Vec<u32>, Vec<u32>);
    type State = ResidentState;
    type Resources = (usize, usize, usize, usize, usize, usize);

    fn empty_like(&self) -> Self {
        Self::empty_like(self)
    }

    fn resource_counts(&self) -> Self::Resources {
        Self::resource_counts(self)
    }

    fn reset_prefix_cache(&mut self) -> bool {
        Self::reset_prefix_cache(self)
    }

    fn accepts(&self, request: &Request, config: &SchedulerConfig) -> bool {
        let fa_blocks_needed = request
            .prompt_len
            .saturating_add(request.max_tokens.saturating_sub(1))
            .div_ceil(self.block_size());
        let fa_usable = self.pool().num_blocks().saturating_sub(1) as usize;
        let one_forward = request.max_tokens == 1
            && (config.max_num_seqs == 1 || request.prompt_len == 1)
            && request.prompt_len > 0
            && self.align_prefill(
                request,
                0,
                request.prompt_len.min(config.max_num_batched_tokens),
                config.max_num_batched_tokens,
            ) == request.prompt_len;
        let gdn_slots_needed = 1
            + usize::from(!one_forward)
            + if config.enable_mtp {
                config.mtp_draft_len
            } else {
                0
            };
        let gdn_usable = self
            .mamba_pool_capacity()
            .map(|capacity| capacity.saturating_sub(1) as usize);
        let fits = if let Some(gdn_usable) = gdn_usable {
            fa_blocks_needed <= fa_usable && gdn_slots_needed <= gdn_usable
        } else {
            fa_blocks_needed.saturating_add(gdn_slots_needed) <= fa_usable
        };
        if !fits {
            tracing::warn!(
                request_id = request.id,
                fa_blocks_needed,
                fa_usable,
                gdn_slots_needed,
                ?gdn_usable,
                "request rejected: FA or GDN demand exceeds pool capacity"
            );
        }
        fits
    }

    fn initialize_hashes(&self, request: &mut Request) {
        request.block_hashes = hash_request_tokens(
            request.content_salt(),
            &request.token_ids,
            self.block_size(),
            &[],
        );
    }

    fn extend_hashes(&self, request: &mut Request) {
        if request.token_ids.len() / self.block_size() > request.block_hashes.len() {
            let first = request.block_hashes.len() * self.block_size();
            let parent = request
                .block_hashes
                .last()
                .copied()
                .or(request.content_salt());
            request.block_hashes.extend(hash_request_tokens(
                parent,
                &request.token_ids[first..],
                self.block_size(),
                &[],
            ));
        }
    }

    fn prefill_block_size(&self) -> usize {
        self.block_size()
    }

    fn align_prefill(&self, request: &Request, start: usize, count: usize, _: usize) -> usize {
        let prefill_end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        if start >= prefill_end {
            return count;
        }
        let block = self.block_size();
        let mut end = start + count;
        let last_boundary = request.token_ids.len() / block * block;
        let next_boundary = (start / block + 1) * block;
        for stop in [
            last_boundary,
            self.partial_checkpoint(request).unwrap_or(0),
            if start.is_multiple_of(block) {
                0
            } else {
                next_boundary
            },
        ] {
            if start < stop && stop < end {
                end = stop;
            }
        }
        end.saturating_sub(start).max(usize::from(count > 0))
    }

    fn prefill_ranges(
        &self,
        request: &Request,
        start: usize,
        limit: usize,
        budget: usize,
    ) -> Vec<(usize, usize)> {
        let limit = self.align_prefill(request, start, limit, budget);
        if limit == 0 {
            vec![]
        } else {
            vec![(1, limit)]
        }
    }

    fn find_prefix(&self, request: &Request) -> PrefixHit<Self::Hit> {
        let (fa, mamba, length) = self.get_computed_blocks(request);
        PrefixHit {
            length,
            blocks: (fa, mamba),
        }
    }

    fn fallback_prefix(
        &self,
        request: &Request,
        previous_length: usize,
    ) -> Option<PrefixHit<Self::Hit>> {
        if previous_length.is_multiple_of(self.block_size()) {
            return None;
        }
        let (fa, mamba, length) = self.find_longest_cache_hit(
            &request.block_hashes,
            request.token_ids.len().saturating_sub(1),
        );
        Some(PrefixHit {
            length,
            blocks: (fa, mamba),
        })
    }

    fn prefix_hash(&self, request: &Request, end: usize) -> BlockHash {
        Self::prefix_hash(self, request, end)
    }

    fn new_step_starts(&mut self) {
        Self::new_step_starts(self);
    }

    fn prefix_eviction_needed(
        &self,
        request: &Request,
        hit: &PrefixHit<Self::Hit>,
        count: usize,
        lookahead: usize,
    ) -> bool {
        Self::prefix_eviction_needed(self, request, &hit.blocks, hit.length, count, lookahead)
    }

    fn allocate(
        &mut self,
        request: &Request,
        hit: PrefixHit<Self::Hit>,
        count: usize,
        lookahead: usize,
        has_running: bool,
    ) -> bool {
        self.allocate_slots(
            request,
            hit.blocks,
            hit.length,
            count,
            lookahead,
            has_running,
        )
        .is_some()
    }

    fn assignment(&self, request_id: RequestId) -> Self::State {
        ResidentState {
            fa_block_table: self.full_attn_blocks(request_id).to_vec(),
            mamba_block_table: self.mamba_blocks(request_id).to_vec(),
            fa_block_copies: self.block_copies(request_id),
            prefix_state_slot: self.prefix_state_slot(request_id),
            cache_checkpoint: self.cache_checkpoint(request_id),
        }
    }

    fn commit_cache(&mut self, request: &Request, _: &crate::execution::CacheStepResult) {
        Self::commit_cache(self, request);
    }

    fn validate_cache_step(
        &self,
        _: &Request,
        step: &crate::execution::CacheStepResult,
    ) -> Result<(), &'static str> {
        if !step.exported_checkpoints.is_empty()
            || !step.page_valid_ends.is_empty()
            || step.mtp_accepted_end.is_some()
        {
            return Err("resident reply contains Qwen resource evidence");
        }
        Ok(())
    }

    fn free(&mut self, request_id: RequestId) {
        Self::free(self, request_id);
    }
}
