use crate::{
    Request, RequestOutput, RequestStatus, Scheduler, UpdateError, WorkerOutput,
    execution::{SamplingMode, WorkPurpose},
    policy::CachePolicy,
};

fn invalid(request_id: u64, reason: &'static str) -> UpdateError {
    UpdateError { request_id, reason }
}

impl<P: CachePolicy> Scheduler<P> {
    pub fn validate_output(&self, worker: &WorkerOutput) -> Result<(), UpdateError> {
        let mut seen = rustc_hash::FxHashSet::default();
        for result in &worker.outputs {
            let id = result.request_id;
            if !seen.insert(id) {
                return Err(invalid(id, "duplicate result"));
            }
            let req = self
                .running
                .iter()
                .find(|request| request.id == id)
                .ok_or_else(|| invalid(id, "unknown request"))?;
            self.validate_result(req, result)?;
        }
        for req in &self.running {
            if req.in_flight.is_some() && !seen.contains(&req.id) {
                return Err(invalid(req.id, "missing result for scheduled request"));
            }
        }
        Ok(())
    }

    fn validate_result(&self, req: &Request, result: &RequestOutput) -> Result<(), UpdateError> {
        let id = req.id;
        if !req.submitted
            || req.in_flight != Some(result.execution)
            || req.work != Some(result.work)
            || result.cache_result.execution != result.execution
            || result.execution.request_id != id
        {
            return Err(invalid(id, "execution identity or work differs"));
        }
        let work = result.work;
        work.validate().map_err(|reason| invalid(id, reason))?;
        if work.expected_tokens(result.num_accepted_draft_tokens) != Some(result.token_ids.len()) {
            return Err(invalid(id, "wrong verified token count"));
        }
        if work.sampling == SamplingMode::None && !result.new_draft_token_ids.is_empty() {
            return Err(invalid(id, "non-sampling work returned drafts"));
        }
        let (_, end) = work.range();
        let expected_computed = match work.sampling {
            SamplingMode::Verify { drafts } => end - drafts + result.num_accepted_draft_tokens,
            _ => end,
        };
        if result.cache_result.computed_end != expected_computed {
            return Err(invalid(id, "wrong computed endpoint"));
        }
        let accepted = if work.purpose == WorkPurpose::Finalize {
            req.accepted_end
                .ok_or_else(|| invalid(id, "finalize has no accepted endpoint"))?
        } else {
            req.token_ids.len().saturating_add(
                result
                    .token_ids
                    .len()
                    .min(req.max_tokens.saturating_sub(req.num_generated_tokens())),
            )
        };
        if result.cache_result.accepted_end != accepted {
            return Err(invalid(id, "wrong accepted endpoint"));
        }
        self.kv
            .validate_cache_step(req, &result.cache_result)
            .map_err(|reason| invalid(id, reason))
    }

    pub fn update(&mut self, mut worker: WorkerOutput) -> Result<WorkerOutput, UpdateError> {
        if self.detached_state {
            return Err(invalid(0, "execution state is detached"));
        }
        self.validate_output(&worker)?;
        for result in &mut worker.outputs {
            let request = self
                .running
                .iter_mut()
                .find(|request| request.id == result.request_id)
                .unwrap();
            request.in_flight = None;
            request.submitted = false;
            request.num_in_flight_tokens = 0;
            if request.status == RequestStatus::Retiring {
                continue;
            }
            request.num_computed_tokens = result.cache_result.computed_end;
            if result.work.purpose == WorkPurpose::Finalize {
                if Some(request.num_computed_tokens) == request.accepted_end {
                    request.status = RequestStatus::Finalized;
                }
            } else {
                Self::append_result(request, result, &self.config);
                self.kv.extend_hashes(request);
                if request.is_finished() {
                    request.status = RequestStatus::Finalizing;
                    request.draft_token_ids.clear();
                }
            }
            self.kv.commit_cache(request, &result.cache_result);
            if request.status == RequestStatus::Finalizing
                && request.accepted_end.is_some()
                && matches!(
                    self.kv.completion_plan(request),
                    crate::execution::CompletionPlan::Ready { .. }
                )
            {
                request.status = RequestStatus::Finalized;
            }
        }
        self.retire_completed();
        Ok(worker)
    }

    fn append_result(
        request: &mut Request,
        result: &mut RequestOutput,
        config: &crate::SchedulerConfig,
    ) {
        result.token_ids.truncate(
            request
                .max_tokens
                .saturating_sub(request.num_generated_tokens()),
        );
        for &token in &result.token_ids {
            request.append_token(token);
        }
        request.draft_token_ids.clear();
        if config.enable_mtp && !request.is_finished() {
            let mut drafts = std::mem::take(&mut result.new_draft_token_ids);
            drafts.truncate(
                config.mtp_draft_len.min(
                    request
                        .max_tokens
                        .saturating_sub(request.num_generated_tokens()),
                ),
            );
            request.set_drafts(drafts);
        }
    }
}
