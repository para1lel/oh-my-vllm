use std::collections::BTreeSet;

use oh_my_vllm_kv_cache::radix::{GDN_PERIOD_TOKENS, KV_PAGE_TOKENS};

use super::*;
use crate::{
    Request,
    execution::{CacheStepResult, CompletionPlan, FinishOutcome, WorkPurpose},
};

impl QwenResidentCache {
    pub(super) fn plan_completion(&self, request: &Request) -> CompletionPlan {
        let end = request.accepted_end.unwrap_or(request.token_ids.len());
        let active = self.active.get(&request.id);
        if active.is_some_and(|active| active.endpoints.contains_key(&end))
            || request.num_computed_tokens == end
        {
            return CompletionPlan::ExportEndpoint { endpoint: end };
        }
        if request.num_computed_tokens < end {
            return CompletionPlan::ForwardAccepted {
                start: request.num_computed_tokens,
                end,
            };
        }
        let private = active
            .and_then(|active| active.endpoints.range(..end).next_back().map(|(&at, _)| at))
            .unwrap_or(0);
        let cached = self.hit(request).resume.end();
        if cached == end {
            return CompletionPlan::ExportEndpoint { endpoint: end };
        }
        let earlier = private.max(cached);
        CompletionPlan::ForwardAccepted {
            start: earlier,
            end,
        }
    }

    pub(super) fn validate_step(
        &self,
        request: &Request,
        step: &CacheStepResult,
    ) -> Result<(), &'static str> {
        let active = self
            .active
            .get(&request.id)
            .ok_or("request has no cache lease")?;
        let exports = active
            .assignment
            .exports
            .iter()
            .map(|export| {
                (
                    export.end,
                    export.destination.0.slot,
                    export.destination.0.version,
                )
            })
            .collect::<BTreeSet<_>>();
        let actual = step
            .exported_checkpoints
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if actual.len() != step.exported_checkpoints.len() || exports != actual {
            return Err("checkpoint export set differs");
        }
        let pages = active
            .assignment
            .page_table
            .iter()
            .map(|page| {
                let start = page.page_index * KV_PAGE_TOKENS;
                let end = if request.work.unwrap().purpose == WorkPurpose::Finalize {
                    step.computed_end.min(start + KV_PAGE_TOKENS).max(start)
                } else {
                    page.valid_end
                        .max(step.computed_end.min(start + KV_PAGE_TOKENS).max(start))
                };
                (page.id.0.slot, page.id.0.version, end)
            })
            .collect::<BTreeSet<_>>();
        let actual = step
            .page_valid_ends
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if actual.len() != step.page_valid_ends.len() || pages != actual {
            return Err("page endpoint set differs");
        }
        if let Some(mtp) = step.mtp_accepted_end
            && mtp != step.computed_end.saturating_sub(1)
        {
            return Err("MTP accepted endpoint differs");
        }
        if request.enable_mtp && step.mtp_accepted_end.is_none() {
            return Err("missing MTP context endpoint");
        }
        Ok(())
    }

    pub(super) fn commit_step(&mut self, request: &Request, step: &CacheStepResult) {
        let mut active = self
            .active
            .remove(&request.id)
            .expect("validated active cache");
        for source in active.copy_sources.drain(..) {
            self.pages.unpin(source.0);
        }
        for export in &active.assignment.exports {
            self.mtp_contexts.remove(&export.destination);
            if let Some(context_end) = step.mtp_accepted_end
                && export.end <= step.computed_end
                && export.end.saturating_sub(1) <= context_end
            {
                self.mtp_contexts
                    .insert(export.destination, export.end.saturating_sub(1));
            }
            if export.role == ExportRole::PrivateEndpoint {
                active.endpoints.retain(|_, id| *id != export.destination);
                if export.end <= step.computed_end {
                    active.endpoints.insert(export.end, export.destination);
                }
            } else if export.role == ExportRole::Periodic && export.end <= step.computed_end {
                let node = self
                    .tree
                    .ensure(&request.cache_domain, &request.token_ids[..export.end]);
                self.attach_checkpoint(node, export.destination);
            }
        }
        for page in active.pages.values_mut() {
            let key = page.id.0;
            if let Some((_, _, end)) = step
                .page_valid_ends
                .iter()
                .find(|(slot, version, _)| *slot == key.slot && *version == key.version)
            {
                page.valid_end = *end;
            }
            if page.valid_end == (page.page_index + 1) * KV_PAGE_TOKENS
                && page.valid_end <= step.computed_end
                && page.access == PageAccess::UniqueWrite
            {
                let node = self
                    .tree
                    .ensure(&request.cache_domain, &request.token_ids[..page.valid_end]);
                if let Some(old) = self.tree.attach_page(node, page.id) {
                    self.pages.detach(old.0);
                }
                self.pages.attach(page.id.0, node);
                page.access = PageAccess::ReadOnly;
            }
        }
        if active.endpoints.contains_key(&step.computed_end) {
            let candidate = active
                .assignment
                .exports
                .iter()
                .find(|export| export.role == ExportRole::CandidateTail)
                .map(|export| export.destination);
            active.checkpoints.retain(|id| {
                if Some(*id) == candidate {
                    true
                } else {
                    self.checkpoints.unpin(id.0);
                    false
                }
            });
        }
        self.active.insert(request.id, active);
        self.clean_tails();
    }

    fn attach_checkpoint(&mut self, node: NodeId, id: CheckpointId) {
        if let Some(old) = self.tree.attach_checkpoint(node, id) {
            self.checkpoints.detach(old.0);
        }
        self.checkpoints.attach(id.0, node);
    }

    pub(super) fn complete_request(&mut self, request: &Request, outcome: FinishOutcome) {
        if !matches!(outcome, FinishOutcome::Succeeded { .. }) {
            return;
        }
        let end = request.accepted_end.expect("successful exact endpoint");
        let active = self
            .active
            .get(&request.id)
            .expect("successful active cache");
        let checkpoint = active
            .assignment
            .exports
            .iter()
            .find(|export| export.end == end && export.role == ExportRole::CandidateTail)
            .map(|export| export.destination);
        let partial = active.pages.get(&((end - 1) / KV_PAGE_TOKENS)).cloned();
        let Some(checkpoint) = checkpoint else {
            return;
        };
        let node = self
            .tree
            .ensure(&request.cache_domain, &request.token_ids[..end]);
        if let Some(page) =
            partial.filter(|page| page.valid_end == end && page.access == PageAccess::UniqueWrite)
        {
            if let Some(old) = self.tree.attach_page(node, page.id) {
                self.pages.detach(old.0);
            }
            self.pages.attach(page.id.0, node);
        }
        if let FinishOutcome::Succeeded { publish_tail: true } = outcome {
            self.attach_checkpoint(node, checkpoint);
            if let Some(ticket) = &request.ticket
                && ticket.identity.session.source != crate::session::SessionSource::Anonymous
            {
                let previous = self.tails.insert(
                    ticket.identity.session.clone(),
                    SuccessfulTail {
                        domain: request.cache_domain.clone(),
                        node,
                        generation: ticket.generation,
                    },
                );
                if let Some(previous) = previous
                    && previous.node != node
                {
                    debug_assert!(previous.generation < ticket.generation);
                    if let Some(id) = self.tree.checkpoint(previous.node)
                        && !self.is_periodic_node(previous.node, id)
                        && !self.tails.values().any(|tail| tail.node == previous.node)
                    {
                        self.checkpoints.detach(id.0);
                        self.tree.detach_checkpoint(previous.node, id);
                    }
                }
            }
        } else if end.is_multiple_of(GDN_PERIOD_TOKENS) {
            self.attach_checkpoint(node, checkpoint);
        }
        self.clean_tails();
    }

    fn is_periodic_node(&self, node: NodeId, _: CheckpointId) -> bool {
        self.tree
            .end(node)
            .is_some_and(|end| end.is_multiple_of(GDN_PERIOD_TOKENS))
    }
}
