use std::collections::VecDeque;

use crate::{
    Request, RequestStatus, ScheduledRequest, Scheduler, SchedulerOutput,
    policy::{CachePolicy, PrefixHit},
};

impl<P: CachePolicy> Scheduler<P> {
    pub fn schedule(&mut self) -> SchedulerOutput<P::State> {
        if self.detached_state {
            return SchedulerOutput::default();
        }
        let started = std::time::Instant::now();
        self.kv.new_step_starts();
        self.retry_cache_stores();

        let mut output = SchedulerOutput {
            finished_request_ids: std::mem::take(&mut self.finished_ids),
            ..Default::default()
        };
        let mut token_budget = self.config.max_num_batched_tokens;
        let mut quotas = self.batch_quotas();
        let resident_allocation_failed =
            self.schedule_running(&mut output, &mut token_budget, &quotas);
        if !resident_allocation_failed {
            self.restore_host_requests(token_budget);
        }
        self.schedule_waiting(
            &mut output,
            &mut token_budget,
            &mut quotas,
            resident_allocation_failed,
        );
        self.resolve_allocation_failure(&output, resident_allocation_failed);

        output.num_batched_tokens = output.scheduled.iter().map(|s| s.token_ids.len()).sum();
        tracing::debug!(
            elapsed_us = started.elapsed().as_micros() as u64,
            running = self.running.len(),
            waiting = self.waiting.len(),
            scheduled_tokens = output.num_batched_tokens,
            resources = ?self.kv.resource_counts(),
            "schedule"
        );
        output
    }

    fn schedule_running(
        &mut self,
        output: &mut SchedulerOutput<P::State>,
        token_budget: &mut usize,
        quotas: &rustc_hash::FxHashMap<u64, usize>,
    ) -> bool {
        self.running
            .make_contiguous()
            .sort_by_key(|request| usize::from(Self::decode_tokens(request) == 0));
        let mut still_running: VecDeque<Request> = VecDeque::new();
        let mut resident_allocation_failed = false;
        while let Some(mut req) = self.running.pop_front() {
            if self.paused.contains(&req.id) {
                still_running.push_back(req);
                continue;
            }
            let tokens_needed = req.num_tokens_with_spec() - req.num_computed_tokens;
            let limit = (*token_budget).min(quotas.get(&req.id).copied().unwrap_or(0));
            let to_schedule =
                self.aligned_prefill(&req, req.num_computed_tokens, tokens_needed.min(limit));
            if to_schedule == 0 {
                still_running.push_back(req);
                continue;
            }

            if self.host.capacity > 0
                && self.host.has_prefix_candidates()
                && self.kv.prefix_eviction_needed(
                    &req,
                    &PrefixHit::default(),
                    to_schedule,
                    self.lookahead(&req, req.num_computed_tokens, to_schedule),
                )
                && self.writeback_host_prefix()
            {
                resident_allocation_failed = true;
                still_running.push_back(req);
                continue;
            }

            let alloc = self.allocate_running(&req, to_schedule, &mut still_running, output);

            match alloc {
                true => {
                    self.push_running(&mut req, to_schedule, token_budget, output);
                    still_running.push_back(req);
                }
                false => {
                    resident_allocation_failed = true;
                    if self.host.capacity > 0 {
                        still_running.push_back(req);
                    } else {
                        self.preempt_request(req, output);
                    }
                }
            }
        }
        self.running = still_running;
        resident_allocation_failed
    }

    fn allocate_running(
        &mut self,
        req: &Request,
        to_schedule: usize,
        still_running: &mut VecDeque<Request>,
        output: &mut SchedulerOutput<P::State>,
    ) -> bool {
        loop {
            let result = self.kv.allocate(
                req,
                PrefixHit::default(),
                to_schedule,
                self.lookahead(req, req.num_computed_tokens, to_schedule),
                !still_running.is_empty() || !self.running.is_empty(),
            );
            if result {
                break result;
            }
            if self.host.capacity > 0
                && (self.host.has_request_store() || self.host.has_prefix_store())
            {
                break false;
            }
            let Some(victim) = self.running.pop_back() else {
                break false;
            };
            if self.host.capacity > 0 {
                if self.host.blocked.contains(&victim.id) {
                    still_running.push_back(victim);
                    break false;
                }
                self.begin_swap(victim);
                break false;
            }
            self.preempt_request(victim, output);
        }
    }

    fn push_running(
        &mut self,
        req: &mut Request,
        to_schedule: usize,
        token_budget: &mut usize,
        output: &mut SchedulerOutput<P::State>,
    ) {
        output.cache_hit_tokens += self.host.prefix_hits.remove(&req.id).unwrap_or(0);
        let scheduled_tokens = req
            .token_ids
            .iter()
            .chain(req.draft_token_ids.iter())
            .skip(req.num_computed_tokens)
            .take(to_schedule)
            .copied()
            .collect();
        output.scheduled.push(ScheduledRequest {
            request_id: req.id,
            token_ids: scheduled_tokens,
            prefill_token_ids: None,
            num_computed_tokens: req.num_computed_tokens,
            state: self.kv.assignment(req.id),
        });
        *token_budget -= to_schedule;
        self.prefill_cursor = Some(req.id);
        req.num_in_flight_tokens = to_schedule;
    }

    fn schedule_waiting(
        &mut self,
        output: &mut SchedulerOutput<P::State>,
        token_budget: &mut usize,
        quotas: &mut rustc_hash::FxHashMap<u64, usize>,
        resident_allocation_failed: bool,
    ) {
        let mut deferred_waiting = VecDeque::new();
        let mut unselected_waiting = VecDeque::new();
        if resident_allocation_failed || !output.preempted_request_ids.is_empty() {
            *quotas = self.retry_waiting_quotas(*token_budget, &mut unselected_waiting);
        }
        while let Some(mut req) = self.waiting.pop_front() {
            if self.paused.contains(&req.id) {
                deferred_waiting.push_back(req);
                continue;
            }
            if self.resident_count() >= self.config.max_num_seqs {
                self.waiting.push_front(req);
                break;
            }

            let hit = self.kv.find_prefix(&req);
            let mut hit_len = hit.length;

            if let Some(started) = self.begin_prefix_load(
                &req,
                hit_len,
                if resident_allocation_failed {
                    0
                } else {
                    *token_budget
                },
            ) {
                if !started {
                    deferred_waiting.push_back(req);
                }
                *quotas = self.retry_waiting_quotas(*token_budget, &mut unselected_waiting);
                continue;
            }

            let total_tokens = req.num_tokens_with_spec();
            let remaining = total_tokens - hit_len;
            req.prefill_start = hit_len;
            let limit = (*token_budget).min(quotas.get(&req.id).copied().unwrap_or(0));
            let mut to_schedule = self.aligned_prefill(&req, hit_len, remaining.min(limit));
            if to_schedule == 0 {
                unselected_waiting.push_back(req);
                continue;
            }

            let alloc = self.allocate_waiting(&mut req, hit, &mut hit_len, &mut to_schedule, limit);

            match alloc {
                true => {
                    self.push_waiting(&mut req, hit_len, to_schedule, token_budget, output);
                    self.running.push_back(req);
                }
                false => {
                    deferred_waiting.push_back(req);
                    *quotas = self.retry_waiting_quotas(*token_budget, &mut unselected_waiting);
                }
            }
        }
        deferred_waiting.append(&mut unselected_waiting);
        deferred_waiting.append(&mut self.waiting);
        self.waiting = deferred_waiting;
    }

    fn allocate_waiting(
        &mut self,
        req: &mut Request,
        hit: PrefixHit<P::Hit>,
        hit_len: &mut usize,
        to_schedule: &mut usize,
        limit: usize,
    ) -> bool {
        let has_running = !self.running.is_empty();
        if self.host.capacity > 0
            && self.host.has_prefix_candidates()
            && self.kv.prefix_eviction_needed(
                req,
                &hit,
                *to_schedule,
                self.lookahead(req, *hit_len, *to_schedule),
            )
            && self.writeback_host_prefix()
        {
            return false;
        }
        let total_tokens = req.num_tokens_with_spec();
        let mut alloc = self.kv.allocate(
            req,
            hit,
            *to_schedule,
            self.lookahead(req, *hit_len, *to_schedule),
            has_running,
        );
        if !alloc && let Some(fallback) = self.kv.fallback_prefix(req, *hit_len) {
            *hit_len = fallback.length;
            req.prefill_start = *hit_len;
            *to_schedule =
                self.aligned_prefill(req, *hit_len, (total_tokens - *hit_len).min(limit));
            alloc = *to_schedule > 0
                && self.kv.allocate(
                    req,
                    fallback,
                    *to_schedule,
                    self.lookahead(req, *hit_len, *to_schedule),
                    has_running,
                );
        }
        alloc
    }

    fn push_waiting(
        &mut self,
        req: &mut Request,
        hit_len: usize,
        to_schedule: usize,
        token_budget: &mut usize,
        output: &mut SchedulerOutput<P::State>,
    ) {
        output.cache_hit_tokens += hit_len;
        let computed_start = hit_len;
        output.scheduled.push(ScheduledRequest {
            request_id: req.id,
            token_ids: req.token_ids[computed_start..computed_start + to_schedule].to_vec(),
            prefill_token_ids: Some(req.token_ids.clone()),
            num_computed_tokens: hit_len,
            state: self.kv.assignment(req.id),
        });
        *token_budget -= to_schedule;
        req.num_computed_tokens = hit_len;
        self.prefill_cursor = Some(req.id);
        req.num_in_flight_tokens = to_schedule;
        req.status = RequestStatus::Running;
    }

    fn resolve_allocation_failure(
        &mut self,
        output: &SchedulerOutput<P::State>,
        resident_allocation_failed: bool,
    ) {
        if self.host.capacity > 0
            && resident_allocation_failed
            && output.scheduled.is_empty()
            && !self.host.has_request_store()
            && !self.host.has_prefix_store()
            && self.host.requests.is_empty()
            && let Some(id) = self
                .running
                .iter()
                .rev()
                .find(|req| !self.host.blocked.contains(&req.id))
                .map(|req| req.id)
        {
            self.swap_out(id);
        }
        if resident_allocation_failed && output.scheduled.is_empty() {
            self.fail_cache_deadlock();
        }
    }
}
