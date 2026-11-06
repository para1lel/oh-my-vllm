use crate::{Scheduler, UpdateError, WorkerOutput, policy::CachePolicy};

impl<P: CachePolicy> Scheduler<P> {
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
}
