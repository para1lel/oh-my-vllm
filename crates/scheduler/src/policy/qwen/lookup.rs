use oh_my_vllm_kv_cache::{
    hash::{BlockHash, hash_block_tokens},
    radix::KV_PAGE_TOKENS,
};

use super::*;
use crate::{
    Request, SchedulerConfig,
    policy::{CachePolicy, PrefixHit},
};

impl QwenResidentCache {
    pub(super) fn hit(&self, request: &Request) -> QwenHit {
        let mut domain = request.cache_domain.clone();
        domain.salt = request.cache_salt;
        let path = self.tree.lookup(&domain, &request.token_ids);
        let checkpoint_hit_end = path.checkpoints.last().map_or(0, |entry| entry.0);
        let mut pages = Vec::new();
        let mut contiguous = 0;
        for (end, _, id) in path.pages {
            if !self.pages.contains(id.0) {
                continue;
            }
            let page_index = (end - 1) / KV_PAGE_TOKENS;
            let binding = PageBinding {
                page_index,
                id,
                valid_end: end,
                access: PageAccess::ReadOnly,
            };
            if let Some(previous) = pages
                .last_mut()
                .filter(|entry: &&mut PageBinding| entry.page_index == page_index)
            {
                *previous = binding;
            } else {
                pages.push(binding);
            }
        }
        for page in &pages {
            if page.page_index * KV_PAGE_TOKENS != contiguous {
                break;
            }
            contiguous = page.valid_end;
            if !contiguous.is_multiple_of(KV_PAGE_TOKENS) {
                break;
            }
        }
        let resume = path
            .checkpoints
            .into_iter()
            .rev()
            .find(|(end, _, id)| {
                *end <= contiguous
                    && self.checkpoints.contains(id.0)
                    && (!request.enable_mtp
                        || self.mtp_contexts.get(id) == Some(&end.saturating_sub(1)))
            })
            .map_or(CheckpointRead::Zero, |(end, _, id)| {
                CheckpointRead::Cached { id, end }
            });
        let mtp_endpoint = match resume {
            CheckpointRead::Cached { id, .. } => self.mtp_contexts.get(&id).map(|&end| (end, id)),
            CheckpointRead::Zero => None,
        };
        QwenHit {
            evidence: PrefixEvidence {
                checkpoint_hit_end,
                kv_contiguous_end: contiguous,
            },
            resume,
            pages,
            mtp_endpoint,
        }
    }

    pub(super) fn clean_tails(&mut self) {
        self.tails.retain(|_, tail| {
            self.tree.checkpoint(tail.node).is_some()
                && self.tree.contains_node(&tail.domain, tail.node)
        });
        while self.tails.len() > self.usable as usize {
            let oldest = self
                .tails
                .iter()
                .min_by_key(|(_, tail)| tail.generation)
                .map(|(session, _)| session.clone())
                .unwrap();
            let tail = self.tails.remove(&oldest).unwrap();
            if !self.tails.values().any(|other| other.node == tail.node)
                && self
                    .tree
                    .end(tail.node)
                    .is_some_and(|end| !end.is_multiple_of(8192))
                && let Some(id) = self.tree.checkpoint(tail.node)
            {
                self.checkpoints.detach(id.0);
                self.tree.detach_checkpoint(tail.node, id);
            }
        }
        self.mtp_contexts
            .retain(|id, _| self.checkpoints.contains(id.0) || self.private.contains(id.0));
    }
}

impl CachePolicy for QwenResidentCache {
    type Hit = QwenHit;
    type State = QwenAssignment;
    type Resources = QwenResources;

    fn empty_like(&self) -> Self {
        Self::new(
            self.usable,
            self.max_seqs,
            self.max_context,
            self.batch_tokens,
        )
    }

    fn resource_counts(&self) -> Self::Resources {
        QwenResources {
            usable_checkpoints: self.checkpoints.capacity(),
            available_checkpoints: self.checkpoints.available(),
            cached_checkpoints: self.checkpoints.cached(),
            usable_pages: self.pages.capacity(),
            available_pages: self.pages.available(),
            cached_pages: self.pages.cached(),
            private_checkpoints: self.private.capacity(),
            available_private_checkpoints: self.private.available(),
            scratch_checkpoints: self.scratch,
        }
    }

    fn reset_worker_epoch(&mut self, epoch: u64) {
        assert!(
            self.active.is_empty(),
            "worker reset requires released executions"
        );
        self.tree = TokenRadix::default();
        self.tails.clear();
        self.mtp_contexts.clear();
        self.checkpoints = SlotPool::new(epoch, 1, self.usable);
        self.pages = SlotPool::new(epoch, 1, self.usable.checked_mul(64).unwrap());
        self.private = SlotPool::new(
            epoch,
            self.usable + 1,
            u32::try_from(self.max_seqs * 6).unwrap(),
        );
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if !self.active.is_empty() {
            return false;
        }
        *self = self.empty_like();
        true
    }

    fn accepts(&self, request: &Request, config: &SchedulerConfig) -> bool {
        let total = request.prompt_len.saturating_add(request.max_tokens);
        request.prompt_len > 0
            && request.max_tokens > 0
            && total <= self.max_context
            && total.div_ceil(KV_PAGE_TOKENS) <= self.pages.capacity()
            && config.mtp_draft_len <= 4
            && config.max_num_seqs <= self.max_seqs
            && config.max_num_batched_tokens <= 32768
    }

    fn batch_token_limit(&self, config: &SchedulerConfig) -> usize {
        let context_rows = if config.enable_mtp {
            config.max_num_seqs.max(4)
        } else {
            0
        };
        config
            .max_num_batched_tokens
            .min(32768_usize.saturating_sub(context_rows))
    }

    fn initialize_hashes(&self, _: &mut Request) {}

    fn extend_hashes(&self, _: &mut Request) {}

    fn align_prefill(&self, _: &Request, _: usize, count: usize, _: usize) -> usize {
        count
    }

    fn prefill_ranges(&self, _: &Request, _: usize, limit: usize, _: usize) -> Vec<(usize, usize)> {
        if limit == 0 {
            vec![]
        } else {
            vec![(1, limit)]
        }
    }

    fn find_prefix(&self, request: &Request) -> PrefixHit<Self::Hit> {
        let hit = self.hit(request);
        PrefixHit {
            length: hit.resume.end(),
            blocks: hit,
        }
    }

    fn fallback_prefix(&self, request: &Request, previous: usize) -> Option<PrefixHit<Self::Hit>> {
        let mut hit = self.hit(request);
        let partial = hit
            .pages
            .iter()
            .any(|page| page.valid_end != (page.page_index + 1) * KV_PAGE_TOKENS);
        if previous == 0 && !partial {
            return None;
        }
        hit.pages
            .retain(|page| page.valid_end == (page.page_index + 1) * KV_PAGE_TOKENS);
        hit.evidence.kv_contiguous_end = 0;
        for page in &hit.pages {
            if page.page_index * KV_PAGE_TOKENS != hit.evidence.kv_contiguous_end {
                break;
            }
            hit.evidence.kv_contiguous_end = page.valid_end;
        }
        hit.resume = CheckpointRead::Zero;
        hit.mtp_endpoint = None;
        Some(PrefixHit {
            length: 0,
            blocks: hit,
        })
    }

    fn prefix_hash(&self, request: &Request, end: usize) -> BlockHash {
        hash_block_tokens(request.cache_salt, &request.token_ids[..end], &[])
    }

    fn new_step_starts(&mut self) {}

    fn prefix_eviction_needed(
        &self,
        _: &Request,
        _: &PrefixHit<Self::Hit>,
        _: usize,
        _: usize,
    ) -> bool {
        false
    }

    fn allocate(
        &mut self,
        request: &Request,
        hit: PrefixHit<Self::Hit>,
        _: usize,
        lookahead: usize,
        _: bool,
    ) -> bool {
        self.allocate_work(request, hit.blocks, lookahead)
    }

    fn assignment(&self, request_id: RequestId) -> Self::State {
        self.active[&request_id].assignment.clone()
    }

    fn completion_plan(&self, request: &Request) -> crate::execution::CompletionPlan {
        self.plan_completion(request)
    }

    fn validate_cache_step(
        &self,
        request: &Request,
        step: &crate::execution::CacheStepResult,
    ) -> Result<(), &'static str> {
        self.validate_step(request, step)
    }

    fn commit_cache(&mut self, request: &Request, step: &crate::execution::CacheStepResult) {
        self.commit_step(request, step);
    }

    fn finish_request(&mut self, request: &Request, outcome: crate::execution::FinishOutcome) {
        self.complete_request(request, outcome);
    }

    fn free(&mut self, request_id: RequestId) {
        if let Some(active) = self.active.remove(&request_id) {
            for page in active.pages.values() {
                self.pages.unpin(page.id.0);
            }
            for id in active.copy_sources {
                self.pages.unpin(id.0);
            }
            for id in active.checkpoints {
                self.checkpoints.unpin(id.0);
            }
            for id in active.private {
                self.private.unpin(id.0);
            }
        }
    }
}
