use std::collections::VecDeque;

use crate::{
    Request, RequestStatus, ScheduledRequest, Scheduler, SchedulerOutput,
    execution::{CompletionPlan, ExecutionKey, SamplingMode, WorkAction, WorkPurpose, WorkSpec},
    policy::{CachePolicy, PrefixHit},
};

impl<P: CachePolicy> Scheduler<P> {
    pub fn schedule(&mut self) -> SchedulerOutput<P::State> {
        if self.detached_state
            || self
                .running
                .iter()
                .any(|request| request.in_flight.is_some())
        {
            return SchedulerOutput::default();
        }
        self.step_id = self
            .step_id
            .checked_add(1)
            .expect("step identity exhausted");
        self.kv.new_step_starts();
        let mut output = SchedulerOutput {
            finished_request_ids: std::mem::take(&mut self.finished_ids),
            ..Default::default()
        };
        let mut budget = self.kv.batch_token_limit(&self.config);
        let quotas = self.batch_quotas();
        self.schedule_running(&mut output, &mut budget, &quotas);
        self.schedule_waiting(&mut output, &mut budget);
        output.num_batched_tokens = output.scheduled.iter().map(|row| row.token_ids.len()).sum();
        output
    }

    fn work(&self, request: &Request, start: usize, count: usize) -> Option<WorkSpec> {
        if request.status == RequestStatus::Finalizing {
            let action = match self.kv.completion_plan(request) {
                CompletionPlan::ForwardAccepted { start, end } => {
                    let end = end.min(start + count);
                    if end == start {
                        return None;
                    }
                    WorkAction::Forward { start, end }
                }
                CompletionPlan::ExportEndpoint { endpoint }
                | CompletionPlan::Ready { endpoint } => WorkAction::ExportOnly { endpoint },
            };
            return Some(WorkSpec {
                purpose: WorkPurpose::Finalize,
                action,
                sampling: SamplingMode::None,
            });
        }
        if count == 0 && start != request.token_ids.len() {
            return None;
        }
        let end = start + count;
        let action = if count == 0 {
            WorkAction::HeadOnly { endpoint: start }
        } else {
            WorkAction::Forward { start, end }
        };
        let drafts = end.saturating_sub(request.token_ids.len());
        let sampling = if end < request.token_ids.len() {
            SamplingMode::None
        } else if drafts > 0 {
            SamplingMode::Verify { drafts }
        } else {
            SamplingMode::Target
        };
        Some(WorkSpec {
            purpose: WorkPurpose::Advance,
            action,
            sampling,
        })
    }

    fn schedule_running(
        &mut self,
        output: &mut SchedulerOutput<P::State>,
        budget: &mut usize,
        quotas: &rustc_hash::FxHashMap<u64, usize>,
    ) {
        self.running
            .make_contiguous()
            .sort_by_key(|request| usize::from(Self::decode_tokens(request) == 0));
        let mut keep = VecDeque::new();
        while let Some(mut request) = self.running.pop_front() {
            if self.paused.contains(&request.id)
                || matches!(
                    request.status,
                    RequestStatus::Finalized | RequestStatus::Retiring
                )
                || (request.status == RequestStatus::Finalizing && request.accepted_end.is_none())
            {
                keep.push_back(request);
                continue;
            }
            let start = match self.kv.completion_plan(&request) {
                CompletionPlan::ForwardAccepted { start, .. }
                    if request.status == RequestStatus::Finalizing =>
                {
                    start
                }
                _ => request.num_computed_tokens,
            };
            let available = request
                .accepted_end
                .unwrap_or(request.num_tokens_with_spec())
                .saturating_sub(start);
            let limit = if request.status == RequestStatus::Finalizing {
                *budget
            } else {
                (*budget).min(quotas.get(&request.id).copied().unwrap_or(0))
            };
            let count = self.aligned_prefill(&request, start, available.min(limit));
            let Some(work) = self.work(&request, start, count) else {
                keep.push_back(request);
                continue;
            };
            request.work = Some(work);
            if self.kv.allocate(
                &request,
                PrefixHit::default(),
                count,
                self.lookahead(&request, start, count),
                !keep.is_empty() || !self.running.is_empty(),
            ) {
                self.push(&mut request, work, false, budget, output);
                keep.push_back(request);
            } else if request.status == RequestStatus::Finalizing {
                keep.push_back(request);
            } else {
                self.preempt_request(request, output);
            }
        }
        self.running = keep;
    }

    fn schedule_waiting(&mut self, output: &mut SchedulerOutput<P::State>, budget: &mut usize) {
        let mut deferred = VecDeque::new();
        let mut unselected = VecDeque::new();
        let mut quotas = self.retry_waiting_quotas(*budget, &mut unselected);
        while let Some(mut request) = self.waiting.pop_front() {
            if self.paused.contains(&request.id) {
                deferred.push_back(request);
                continue;
            }
            if self.resident_count() >= self.config.max_num_seqs {
                self.waiting.push_front(request);
                break;
            }
            let hit = self.kv.find_prefix(&request);
            let mut start = hit.length;
            let limit = (*budget).min(quotas.get(&request.id).copied().unwrap_or(0));
            let mut count = self.aligned_prefill(
                &request,
                start,
                request
                    .num_tokens_with_spec()
                    .saturating_sub(start)
                    .min(limit),
            );
            let Some(work) = self.work(&request, start, count) else {
                deferred.push_back(request);
                continue;
            };
            request.work = Some(work);
            request.prefill_start = start;
            let mut allocated = self.kv.allocate(
                &request,
                hit,
                count,
                self.lookahead(&request, start, count),
                !self.running.is_empty(),
            );
            if !allocated && let Some(hit) = self.kv.fallback_prefix(&request, start) {
                start = hit.length;
                count = self.aligned_prefill(
                    &request,
                    start,
                    request
                        .num_tokens_with_spec()
                        .saturating_sub(start)
                        .min(*budget),
                );
                request.work = self.work(&request, start, count);
                request.prefill_start = start;
                if request.work.is_some() {
                    allocated = self.kv.allocate(
                        &request,
                        hit,
                        count,
                        self.lookahead(&request, start, count),
                        !self.running.is_empty(),
                    );
                }
            }
            if allocated {
                request.num_computed_tokens = start;
                request.status = RequestStatus::Running;
                output.cache_hit_tokens += start;
                let work = request.work.unwrap();
                self.push(&mut request, work, true, budget, output);
                self.running.push_back(request);
            } else {
                request.work = None;
                deferred.push_back(request);
                quotas = self.retry_waiting_quotas(*budget, &mut unselected);
            }
        }
        deferred.append(&mut self.waiting);
        self.waiting = deferred;
    }

    fn push(
        &mut self,
        request: &mut Request,
        work: WorkSpec,
        first: bool,
        budget: &mut usize,
        output: &mut SchedulerOutput<P::State>,
    ) {
        self.next_lease = self
            .next_lease
            .checked_add(1)
            .expect("execution lease exhausted");
        let execution = ExecutionKey {
            worker_epoch: self.worker_epoch,
            step_id: self.step_id,
            request_id: request.id,
            lease_id: self.next_lease,
        };
        let (start, end) = work.range();
        let token_ids = request
            .token_ids
            .iter()
            .chain(&request.draft_token_ids)
            .skip(start)
            .take(end - start)
            .copied()
            .collect();
        request.in_flight = Some(execution);
        request.submitted = false;
        request.work = Some(work);
        request.num_in_flight_tokens = end - start;
        output.scheduled.push(ScheduledRequest {
            request_id: request.id,
            execution,
            work,
            token_ids,
            prefill_token_ids: first.then(|| request.token_ids.clone()),
            num_computed_tokens: start,
            state: self.kv.assignment(request.id),
            accepted_end: request.accepted_end.unwrap_or(request.token_ids.len()),
        });
        *budget -= end - start;
        self.prefill_cursor = Some(request.id);
    }
}
