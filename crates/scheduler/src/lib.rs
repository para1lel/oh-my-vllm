pub mod cache;
pub mod output;
mod planning;
pub mod policy;
pub mod request;
pub mod transaction;

use std::collections::VecDeque;

use oh_my_vllm_kv_cache::group::RequestId;

use policy::{CachePolicy, PrefixHit};

pub use output::{RequestOutput, ScheduledRequest, SchedulerOutput, WorkerOutput};
pub use request::{Request, RequestStatus};

#[derive(Debug, thiserror::Error)]
#[error("invalid worker output for request {request_id}: {reason}")]
pub struct UpdateError {
    pub request_id: RequestId,
    pub reason: &'static str,
}

#[derive(Clone)]
pub struct SchedulerConfig {
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub enable_mtp: bool,
    pub mtp_draft_len: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 32768,
            max_num_seqs: 4,
            enable_mtp: false,
            mtp_draft_len: 4,
        }
    }
}

pub struct Scheduler<P: CachePolicy> {
    config: SchedulerConfig,
    waiting: VecDeque<Request>,
    running: VecDeque<Request>,
    paused: rustc_hash::FxHashSet<RequestId>,
    kv: P,
    finished_ids: Vec<u64>,
    prefill_cursor: Option<RequestId>,
    host: cache::HostCache<P>,
    detached_state: bool,
}

impl<P: CachePolicy> Scheduler<P> {
    pub fn empty_like(&self) -> Self {
        let mut scheduler = Self::new(self.config.clone(), self.kv.empty_like());
        scheduler.enable_host_cache(self.host.capacity);
        scheduler
    }

    pub fn cache_resource_counts(&self) -> P::Resources {
        self.kv.resource_counts()
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        if self.num_running() + self.num_waiting() != 0 || !self.host.pending.is_empty() {
            return false;
        }
        self.clear_host_prefixes();
        self.kv.reset_prefix_cache()
    }

    fn aligned_prefill(&self, request: &Request, start: usize, count: usize) -> usize {
        self.kv
            .align_prefill(request, start, count, self.config.max_num_batched_tokens)
    }

    fn decode_tokens(request: &Request) -> usize {
        let end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        if request.num_computed_tokens < end {
            0
        } else {
            request
                .num_tokens_with_spec()
                .saturating_sub(request.num_computed_tokens)
        }
    }

    pub fn new(config: SchedulerConfig, kv: P) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            paused: rustc_hash::FxHashSet::default(),
            kv,
            finished_ids: Vec::new(),
            prefill_cursor: None,
            host: cache::HostCache::default(),
            detached_state: false,
        }
    }

    fn lookahead(&self, request: &Request, start: usize, count: usize) -> usize {
        if self.config.enable_mtp {
            self.config.mtp_draft_len.min(
                request
                    .prompt_len
                    .saturating_add(request.max_tokens.saturating_sub(1))
                    .saturating_sub(start.saturating_add(count)),
            )
        } else {
            0
        }
    }

    pub fn add_request(&mut self, mut req: Request) -> bool {
        if !self.kv.accepts(&req, &self.config, self.host.capacity > 0) {
            return false;
        }
        self.kv.initialize_hashes(&mut req);
        self.waiting.push_back(req);
        true
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len() + self.host.requests.len() + self.host.request_transfers()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn pause(&mut self, request_id: RequestId) {
        self.paused.insert(request_id);
    }

    pub fn resume(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
    }

    fn preempt_request(&mut self, mut req: Request, output: &mut SchedulerOutput<P::State>) {
        self.kv.free(req.id);
        req.num_computed_tokens = 0;
        req.prefill_start = 0;
        req.num_in_flight_tokens = 0;
        req.draft_token_ids.clear();
        req.status = RequestStatus::Preempted;
        output.preempted_request_ids.push(req.id);
        self.waiting.push_front(req);
    }

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
            let limit = token_budget.min(quotas.get(&req.id).copied().unwrap_or(0));
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

            let alloc = loop {
                let result = self.kv.allocate(
                    &req,
                    PrefixHit::default(),
                    to_schedule,
                    self.lookahead(&req, req.num_computed_tokens, to_schedule),
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
                self.preempt_request(victim, &mut output);
            };

            match alloc {
                true => {
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
                    token_budget -= to_schedule;
                    self.prefill_cursor = Some(req.id);
                    req.num_in_flight_tokens = to_schedule;
                    still_running.push_back(req);
                }
                false => {
                    resident_allocation_failed = true;
                    if self.host.capacity > 0 {
                        still_running.push_back(req);
                    } else {
                        self.preempt_request(req, &mut output);
                    }
                }
            }
        }
        self.running = still_running;
        if !resident_allocation_failed {
            self.restore_host_requests(token_budget);
        }

        let mut deferred_waiting = VecDeque::new();
        let mut unselected_waiting = VecDeque::new();
        if resident_allocation_failed || !output.preempted_request_ids.is_empty() {
            quotas = self.retry_waiting_quotas(token_budget, &mut unselected_waiting);
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
                    token_budget
                },
            ) {
                if !started {
                    deferred_waiting.push_back(req);
                }
                quotas = self.retry_waiting_quotas(token_budget, &mut unselected_waiting);
                continue;
            }

            let total_tokens = req.num_tokens_with_spec();
            let remaining = total_tokens - hit_len;
            req.prefill_start = hit_len;
            let limit = token_budget.min(quotas.get(&req.id).copied().unwrap_or(0));
            let mut to_schedule = self.aligned_prefill(&req, hit_len, remaining.min(limit));
            if to_schedule == 0 {
                unselected_waiting.push_back(req);
                continue;
            }

            let has_running = !self.running.is_empty();
            if self.host.capacity > 0
                && self.host.has_prefix_candidates()
                && self.kv.prefix_eviction_needed(
                    &req,
                    &hit,
                    to_schedule,
                    self.lookahead(&req, hit_len, to_schedule),
                )
                && self.writeback_host_prefix()
            {
                deferred_waiting.push_back(req);
                quotas = self.retry_waiting_quotas(token_budget, &mut unselected_waiting);
                continue;
            }
            let mut alloc = self.kv.allocate(
                &req,
                hit,
                to_schedule,
                self.lookahead(&req, hit_len, to_schedule),
                has_running,
            );
            if !alloc && let Some(fallback) = self.kv.fallback_prefix(&req, hit_len) {
                hit_len = fallback.length;
                req.prefill_start = hit_len;
                to_schedule =
                    self.aligned_prefill(&req, hit_len, (total_tokens - hit_len).min(limit));
                alloc = to_schedule > 0
                    && self.kv.allocate(
                        &req,
                        fallback,
                        to_schedule,
                        self.lookahead(&req, hit_len, to_schedule),
                        has_running,
                    );
            }

            match alloc {
                true => {
                    output.cache_hit_tokens += hit_len;
                    let computed_start = hit_len;
                    output.scheduled.push(ScheduledRequest {
                        request_id: req.id,
                        token_ids: req.token_ids[computed_start..computed_start + to_schedule]
                            .to_vec(),
                        prefill_token_ids: Some(req.token_ids.clone()),
                        num_computed_tokens: hit_len,
                        state: self.kv.assignment(req.id),
                    });
                    token_budget -= to_schedule;
                    req.num_computed_tokens = hit_len;
                    self.prefill_cursor = Some(req.id);
                    req.num_in_flight_tokens = to_schedule;
                    req.status = RequestStatus::Running;
                    self.running.push_back(req);
                }
                false => {
                    deferred_waiting.push_back(req);
                    quotas = self.retry_waiting_quotas(token_budget, &mut unselected_waiting);
                }
            }
        }
        deferred_waiting.append(&mut unselected_waiting);
        deferred_waiting.append(&mut self.waiting);
        self.waiting = deferred_waiting;

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

    pub fn validate_output(&self, worker: &WorkerOutput) -> Result<(), UpdateError> {
        let mut seen = rustc_hash::FxHashSet::default();
        for result in &worker.outputs {
            let id = result.request_id;
            if !seen.insert(id) {
                return Err(UpdateError {
                    request_id: id,
                    reason: "duplicate result",
                });
            }
            let Some(req) = self.running.iter().find(|request| request.id == id) else {
                return Err(UpdateError {
                    request_id: id,
                    reason: "unknown request",
                });
            };
            if req.num_in_flight_tokens == 0 {
                return Err(UpdateError {
                    request_id: id,
                    reason: "request was not scheduled",
                });
            }
            let Some(scheduled_end) = req
                .num_computed_tokens
                .checked_add(req.num_in_flight_tokens)
            else {
                return Err(UpdateError {
                    request_id: id,
                    reason: "computed position overflow",
                });
            };
            let scheduled_drafts = scheduled_end.saturating_sub(req.token_ids.len());
            if result.num_accepted_draft_tokens > scheduled_drafts {
                return Err(UpdateError {
                    request_id: id,
                    reason: "accepted unscheduled drafts",
                });
            }
            let final_chunk = scheduled_end >= req.token_ids.len().max(req.prompt_len);
            let expected = if final_chunk {
                result.num_accepted_draft_tokens + 1
            } else {
                0
            };
            if result.token_ids.len() != expected {
                return Err(UpdateError {
                    request_id: id,
                    reason: "wrong verified token count",
                });
            }
            let computed_after =
                scheduled_end - scheduled_drafts + result.num_accepted_draft_tokens;
            let Some(history_after) = req.token_ids.len().checked_add(result.token_ids.len())
            else {
                return Err(UpdateError {
                    request_id: id,
                    reason: "accepted history overflow",
                });
            };
            if computed_after > history_after {
                return Err(UpdateError {
                    request_id: id,
                    reason: "computed position exceeds accepted history",
                });
            }
        }
        for req in &self.running {
            if req.num_in_flight_tokens > 0 && !seen.contains(&req.id) {
                return Err(UpdateError {
                    request_id: req.id,
                    reason: "missing result for scheduled request",
                });
            }
        }
        Ok(())
    }

    pub fn update(&mut self, mut worker: WorkerOutput) -> Result<WorkerOutput, UpdateError> {
        if self.detached_state {
            return Err(UpdateError {
                request_id: 0,
                reason: "execution state is detached",
            });
        }
        self.validate_output(&worker)?;
        if !worker.outputs.is_empty() {
            self.cache_progress();
        }
        let mut finished_ids = Vec::new();

        for result in &mut worker.outputs {
            let req = self
                .running
                .iter_mut()
                .find(|r| r.id == result.request_id)
                .expect("validated request must remain scheduled");

            let scheduled_end = req.num_computed_tokens + req.num_in_flight_tokens;
            let scheduled_drafts = scheduled_end.saturating_sub(req.token_ids.len());

            req.num_computed_tokens =
                scheduled_end - scheduled_drafts + result.num_accepted_draft_tokens;
            req.num_in_flight_tokens = 0;

            result
                .token_ids
                .truncate(req.max_tokens.saturating_sub(req.num_generated_tokens()));
            for &token in &result.token_ids {
                if req.is_finished() {
                    break;
                }
                req.append_token(token);
            }
            req.draft_token_ids.clear();
            self.kv.extend_hashes(req);

            if self.config.enable_mtp && !result.new_draft_token_ids.is_empty() {
                let mut drafts = std::mem::take(&mut result.new_draft_token_ids);
                drafts.truncate(
                    self.config.mtp_draft_len.min(
                        req.max_tokens
                            .saturating_sub(req.num_generated_tokens() + 1),
                    ),
                );
                req.set_drafts(drafts);
            }

            if req.is_finished() {
                finished_ids.push(req.id);
            }
            self.kv.commit_cache(req);
        }

        self.finished_ids.extend_from_slice(&finished_ids);

        for id in &finished_ids {
            if let Some(request) = self.running.iter().find(|r| r.id == *id).cloned() {
                self.remember_host_prefix(&request);
            }
            self.running.retain(|r| r.id != *id);
            self.paused.remove(id);
            self.kv.free(*id);
        }
        Ok(worker)
    }

    pub fn abort(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
        self.host.prefix_hits.remove(&request_id);
        if self.abort_host_request(request_id) {
            return;
        }
        if self.running.iter().any(|r| r.id == request_id) {
            self.running.retain(|r| r.id != request_id);
            self.kv.free(request_id);
            self.finished_ids.push(request_id);
        } else {
            let before = self.waiting.len();
            self.waiting.retain(|r| r.id != request_id);
            if self.waiting.len() != before {
                self.finished_ids.push(request_id);
            }
        }
    }
}

#[cfg(test)]
mod tests;
