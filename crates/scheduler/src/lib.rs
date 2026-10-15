pub mod output;
pub mod request;

use std::collections::VecDeque;

use oh_my_vllm_kv_cache::{coordinator::HybridCoordinator, group::RequestId};

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
            max_num_seqs: 256,
            enable_mtp: false,
            mtp_draft_len: 4,
        }
    }
}

pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<Request>,
    running: VecDeque<Request>,
    paused: rustc_hash::FxHashSet<RequestId>,
    kv: HybridCoordinator,
    finished_ids: Vec<u64>,
}

impl Scheduler {
    pub fn empty_like(&self) -> Self {
        Self::new(self.config.clone(), self.kv.empty_like())
    }

    pub fn cache_resource_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        self.kv.resource_counts()
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        self.running.is_empty() && self.waiting.is_empty() && self.kv.reset_prefix_cache()
    }

    fn aligned_prefill(&self, request: &Request, start: usize, count: usize) -> usize {
        let prefill_end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        if start >= prefill_end {
            return count;
        }
        let block = self.kv.block_size();
        let mut end = start + count;
        if end < prefill_end && self.config.max_num_batched_tokens >= block {
            end = end / block * block;
        }
        let last_boundary = request.token_ids.len() / block * block;
        let next_boundary = (start / block + 1) * block;
        for stop in [
            last_boundary,
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
        end.saturating_sub(start).max(if count > 0 {
            1
        } else {
            0
        })
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

    fn long_prefill(&self, request: &Request, start: usize) -> bool {
        let end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        start < end
            && end.saturating_sub(request.prefill_start) > self.config.max_num_batched_tokens
    }

    fn long_prefill_quota(&self) -> Option<usize> {
        let mut count = 0;
        let mut decode = 0_usize;
        for request in &self.running {
            if self.paused.contains(&request.id) {
                continue;
            }
            if self.long_prefill(request, request.num_computed_tokens) {
                count += 1;
            } else {
                decode = decode.saturating_add(Self::decode_tokens(request));
            }
        }
        let slots = self.config.max_num_seqs.saturating_sub(self.running.len());
        for request in self
            .waiting
            .iter()
            .filter(|r| !self.paused.contains(&r.id))
            .take(slots)
        {
            let end = request
                .prompt_len
                .max(request.token_ids.len().saturating_sub(1));
            if end <= self.config.max_num_batched_tokens {
                continue;
            }
            let (_, _, hit) = self.kv.get_computed_blocks(request);
            count += usize::from(end.saturating_sub(hit) > self.config.max_num_batched_tokens);
        }
        let budget = self.config.max_num_batched_tokens.saturating_sub(decode);
        let block = self.kv.block_size();
        if count < 2 || budget / block < count {
            return None;
        }
        Some(budget / count / block * block)
    }

    fn defer_aligned_prefill(
        &self,
        request: &Request,
        start: usize,
        count: usize,
        budget: usize,
        has_prefill_chunk: bool,
    ) -> bool {
        let block = self.kv.block_size();
        let prefill_end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        has_prefill_chunk
            && count == 1
            && self.config.max_num_batched_tokens >= block
            && prefill_end.saturating_sub(start) > budget
            && budget < block - start % block
    }

    pub fn new(config: SchedulerConfig, kv: HybridCoordinator) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            paused: rustc_hash::FxHashSet::default(),
            kv,
            finished_ids: Vec::new(),
        }
    }

    pub fn add_request(&mut self, mut req: Request) -> bool {
        let block = self.kv.block_size();
        let fa_blocks_needed = req
            .prompt_len
            .saturating_add(req.max_tokens.saturating_sub(1))
            .div_ceil(block);
        let fa_usable = self.kv.pool().num_blocks().saturating_sub(1) as usize;
        let one_forward = req.max_tokens == 1
            && req.prompt_len > 0
            && self.aligned_prefill(
                &req,
                0,
                req.prompt_len.min(self.config.max_num_batched_tokens),
            ) == req.prompt_len;
        let gdn_slots_needed = 1
            + usize::from(!one_forward)
            + if self.config.enable_mtp {
                self.config.mtp_draft_len
            } else {
                0
            };
        let gdn_usable = self
            .kv
            .mamba_pool_capacity()
            .map(|capacity| capacity.saturating_sub(1) as usize);
        let fits = if let Some(gdn_usable) = gdn_usable {
            fa_blocks_needed <= fa_usable && gdn_slots_needed <= gdn_usable
        } else {
            fa_blocks_needed.saturating_add(gdn_slots_needed) <= fa_usable
        };
        if !fits {
            tracing::warn!(
                request_id = req.id,
                fa_blocks_needed,
                fa_usable,
                gdn_slots_needed,
                ?gdn_usable,
                "request rejected: FA or GDN demand exceeds pool capacity"
            );
            return false;
        }
        req.block_hashes = self.kv.compute_block_hashes(&req.token_ids);
        self.waiting.push_back(req);
        true
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn pause(&mut self, request_id: RequestId) {
        if self.running.iter().any(|req| req.id == request_id)
            || self.waiting.iter().any(|req| req.id == request_id)
        {
            self.paused.insert(request_id);
        }
    }

    pub fn resume(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
    }

    fn preempt_request(&mut self, mut req: Request, output: &mut SchedulerOutput) {
        self.kv.free(req.id);
        req.num_computed_tokens = 0;
        req.prefill_start = 0;
        req.num_in_flight_tokens = 0;
        req.draft_token_ids.clear();
        req.status = RequestStatus::Preempted;
        output.preempted_request_ids.push(req.id);
        self.waiting.push_front(req);
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        let started = std::time::Instant::now();
        self.kv.new_step_starts();

        let mut output = SchedulerOutput {
            finished_request_ids: std::mem::take(&mut self.finished_ids),
            ..Default::default()
        };
        let mut token_budget = self.config.max_num_batched_tokens;
        let mut has_prefill_chunk = false;
        let long_quota = self.long_prefill_quota();
        let mut pending_decode = self
            .running
            .iter()
            .filter(|request| !self.paused.contains(&request.id))
            .map(Self::decode_tokens)
            .fold(0_usize, usize::saturating_add);

        let mut still_running: VecDeque<Request> = VecDeque::new();
        while let Some(mut req) = self.running.pop_front() {
            if self.paused.contains(&req.id) {
                still_running.push_back(req);
                continue;
            }
            let tokens_needed = req.num_tokens_with_spec() - req.num_computed_tokens;
            let decode = Self::decode_tokens(&req);
            pending_decode = pending_decode.saturating_sub(decode);
            let available = if decode == 0 {
                token_budget.saturating_sub(pending_decode)
            } else {
                token_budget
            };
            let limit = if self.long_prefill(&req, req.num_computed_tokens) {
                available.min(long_quota.unwrap_or(available))
            } else {
                available
            };
            let to_schedule =
                self.aligned_prefill(&req, req.num_computed_tokens, tokens_needed.min(limit));
            if to_schedule == 0
                || self.defer_aligned_prefill(
                    &req,
                    req.num_computed_tokens,
                    to_schedule,
                    limit,
                    has_prefill_chunk,
                )
            {
                still_running.push_back(req);
                continue;
            }

            let alloc = loop {
                let result = self.kv.allocate_slots(
                    &req,
                    (Vec::new(), Vec::new()),
                    0,
                    to_schedule,
                    0,
                    !still_running.is_empty() || !self.running.is_empty(),
                );
                if result.is_some() {
                    break result;
                }
                let Some(victim) = self.running.pop_back() else {
                    break None;
                };
                if !self.paused.contains(&victim.id) {
                    pending_decode = pending_decode.saturating_sub(Self::decode_tokens(&victim));
                }
                self.preempt_request(victim, &mut output);
            };

            match alloc {
                Some(_) => {
                    let prefill_end = req.prompt_len.max(req.token_ids.len().saturating_sub(1));
                    has_prefill_chunk |= req.num_computed_tokens < prefill_end
                        && to_schedule >= self.kv.block_size();
                    let fa_table = self.kv.full_attn_blocks(req.id).to_vec();
                    let mb_table = self.kv.mamba_blocks(req.id).to_vec();
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
                        fa_block_table: fa_table,
                        mamba_block_table: mb_table,
                    });
                    token_budget -= to_schedule;
                    req.num_in_flight_tokens = to_schedule;
                    still_running.push_back(req);
                }
                None => {
                    self.preempt_request(req, &mut output);
                }
            }
        }
        self.running = still_running;

        let mut paused_waiting = VecDeque::new();
        while let Some(mut req) = self.waiting.pop_front() {
            if self.paused.contains(&req.id) {
                paused_waiting.push_back(req);
                continue;
            }
            if self.running.len() >= self.config.max_num_seqs {
                self.waiting.push_front(req);
                break;
            }

            let (fa_hit, mb_hit, hit_len) = self.kv.get_computed_blocks(&req);

            let total_tokens = req.num_tokens_with_spec();
            let remaining = total_tokens - hit_len;
            req.prefill_start = hit_len;
            let limit = if self.long_prefill(&req, hit_len) {
                token_budget.min(long_quota.unwrap_or(token_budget))
            } else {
                token_budget
            };
            let to_schedule = self.aligned_prefill(&req, hit_len, remaining.min(limit));
            if to_schedule == 0
                || self.defer_aligned_prefill(&req, hit_len, to_schedule, limit, has_prefill_chunk)
            {
                self.waiting.push_front(req);
                break;
            }

            let has_running = !self.running.is_empty();
            let alloc = self.kv.allocate_slots(
                &req,
                (fa_hit, mb_hit),
                hit_len,
                to_schedule,
                0,
                has_running,
            );

            match alloc {
                Some(_) => {
                    let prefill_end = req.prompt_len.max(req.token_ids.len().saturating_sub(1));
                    has_prefill_chunk |=
                        hit_len < prefill_end && to_schedule >= self.kv.block_size();
                    let fa_table = self.kv.full_attn_blocks(req.id).to_vec();
                    let mb_table = self.kv.mamba_blocks(req.id).to_vec();

                    output.cache_hit_tokens += hit_len;
                    let computed_start = hit_len;
                    output.scheduled.push(ScheduledRequest {
                        request_id: req.id,
                        token_ids: req.token_ids[computed_start..computed_start + to_schedule]
                            .to_vec(),
                        prefill_token_ids: Some(req.token_ids.clone()),
                        num_computed_tokens: hit_len,
                        fa_block_table: fa_table,
                        mamba_block_table: mb_table,
                    });
                    token_budget -= to_schedule;
                    req.num_computed_tokens = hit_len;
                    req.num_in_flight_tokens = to_schedule;
                    req.status = RequestStatus::Running;
                    self.running.push_back(req);
                }
                None => {
                    self.waiting.push_front(req);
                    break;
                }
            }
        }
        paused_waiting.append(&mut self.waiting);
        self.waiting = paused_waiting;

        output.num_batched_tokens = output.scheduled.iter().map(|s| s.token_ids.len()).sum();
        tracing::debug!(
            elapsed_us = started.elapsed().as_micros() as u64,
            running = self.running.len(),
            waiting = self.waiting.len(),
            scheduled_tokens = output.num_batched_tokens,
            free_blocks = self.kv.pool().num_free_blocks(),
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
        self.validate_output(&worker)?;
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
            let full_blocks = req.token_ids.len() / self.kv.block_size();
            if full_blocks > req.block_hashes.len() {
                self.kv
                    .extend_block_hashes(&req.token_ids, &mut req.block_hashes);
            }

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
            self.running.retain(|r| r.id != *id);
            self.paused.remove(id);
            self.kv.free(*id);
        }
        Ok(worker)
    }

    pub fn abort(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
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
