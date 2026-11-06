use crate::coordinator::{HybridCoordinator, align_down, partial_hash};
use crate::group::{BlocksNeeded, CacheRequest, RequestId};

struct SlotPlan {
    computed: usize,
    main: usize,
    slots: usize,
}

impl HybridCoordinator {
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

        let plan = SlotPlan {
            computed: total_computed,
            main: num_tokens_main_model,
            slots: num_tokens_need_slot,
        };

        let processed = total_computed.saturating_sub(request.num_in_flight_tokens());
        self.full_attn
            .remove_skipped_blocks(rid, processed, &mut self.pool);
        self.mamba.remove_skipped_blocks(
            rid,
            processed,
            self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
        );

        let partial =
            self.allocation_capacity(request, &new_computed_blocks, &plan, has_scheduled_reqs)?;

        self.attach_computed_blocks(rid, total_computed, new_computed_blocks);
        self.replace_partial_sources(rid, total_computed, partial);

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

        self.stage_allocation_cache(request, num_tokens_main_model);
        Some((new_fa, new_mb))
    }

    fn allocation_capacity(
        &self,
        request: &impl CacheRequest,
        new_computed_blocks: &(Vec<u32>, Vec<u32>),
        plan: &SlotPlan,
        has_scheduled_reqs: bool,
    ) -> Option<(bool, bool)> {
        let rid = request.request_id();
        let fa_needed = self.full_attn.get_num_blocks_to_allocate(
            rid,
            plan.slots,
            &new_computed_blocks.0,
            plan.computed,
            plan.main,
            &self.pool,
        );
        let mb_needed = self.mamba.get_num_blocks_to_allocate(
            rid,
            plan.slots,
            &new_computed_blocks.1,
            plan.computed,
            plan.main,
            self.mamba_pool.as_ref().unwrap_or(&self.pool),
        );

        let fa_partial =
            self.full_attn
                .partial_source(rid, &new_computed_blocks.0, plan.computed, &self.pool);
        // Host restores have no committed cache hash until their first validated result.
        let restored_partial = self.restores.get(&rid).copied().filter(|source| {
            !plan.computed.is_multiple_of(self.block_size)
                && self.mamba.blocks(rid).get(plan.computed / self.block_size) == Some(source)
        });
        let mb_partial = self
            .mamba
            .partial_source(
                rid,
                &new_computed_blocks.1,
                plan.computed,
                self.mamba_pool.as_ref().unwrap_or(&self.pool),
            )
            .or(restored_partial);
        let fa_count = match fa_needed {
            BlocksNeeded::DeferToNextStep => return None,
            BlocksNeeded::Blocks(n) => n + usize::from(fa_partial.is_some()),
        };
        let mb_count = match mb_needed {
            BlocksNeeded::DeferToNextStep => return None,
            BlocksNeeded::Blocks(n) => n + usize::from(mb_partial.is_some()),
        };

        let watermark = if has_scheduled_reqs && request.is_waiting_or_preempted() {
            self.watermark_blocks
        } else {
            0
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

        Some((fa_partial.is_some(), mb_partial.is_some()))
    }

    fn attach_computed_blocks(
        &mut self,
        rid: RequestId,
        total_computed: usize,
        hits: (Vec<u32>, Vec<u32>),
    ) {
        let (fa_hits, mb_hits) = hits;
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
    }

    fn replace_partial_sources(
        &mut self,
        rid: RequestId,
        total_computed: usize,
        partial: (bool, bool),
    ) {
        if partial.0 {
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
        if partial.1 {
            let (source, _) = self.mamba.replace_partial(
                rid,
                total_computed,
                self.mamba_pool.as_mut().unwrap_or(&mut self.pool),
            );
            self.restores.insert(rid, source);
        }
    }

    fn stage_allocation_cache(
        &mut self,
        request: &impl CacheRequest,
        num_tokens_main_model: usize,
    ) {
        let rid = request.request_id();
        let num_tokens_to_cache = num_tokens_main_model.min(request.num_tokens());
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
    }
}
