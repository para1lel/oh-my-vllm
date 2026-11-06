use crate::{RequestStatus, Scheduler, UpdateError, execution::FinishOutcome, policy::CachePolicy};

impl<P: CachePolicy> Scheduler<P> {
    pub fn finalize_request(
        &mut self,
        request_id: u64,
        accepted_end: usize,
    ) -> Result<(), UpdateError> {
        let request = self
            .running
            .iter_mut()
            .find(|row| row.id == request_id)
            .ok_or(UpdateError {
                request_id,
                reason: "unknown finalizing request",
            })?;
        if request.status == RequestStatus::Finalized && request.accepted_end == Some(accepted_end)
        {
            return Ok(());
        }
        if request.in_flight.is_some()
            || accepted_end < request.prompt_len
            || accepted_end > request.token_ids.len()
            || matches!(
                request.status,
                RequestStatus::Retiring | RequestStatus::Finalized
            )
        {
            return Err(UpdateError {
                request_id,
                reason: "invalid accepted endpoint",
            });
        }
        if let Some(frozen) = request.accepted_end
            && frozen != accepted_end
        {
            return Err(UpdateError {
                request_id,
                reason: "accepted endpoint is already frozen",
            });
        }
        request.token_ids.truncate(accepted_end);
        request.draft_token_ids.clear();
        request.accepted_end = Some(accepted_end);
        request.status = RequestStatus::Finalizing;
        self.kv.initialize_hashes(request);
        if matches!(
            self.kv.completion_plan(request),
            crate::execution::CompletionPlan::Ready { .. }
        ) {
            request.status = RequestStatus::Finalized;
        }
        self.paused.remove(&request_id);
        Ok(())
    }

    pub fn is_finalized(&self, request_id: u64) -> bool {
        self.running
            .iter()
            .any(|row| row.id == request_id && row.status == RequestStatus::Finalized)
    }

    pub fn finish_success(&mut self, request_id: u64) -> Result<(), UpdateError> {
        let index = self
            .running
            .iter()
            .position(|row| {
                row.id == request_id
                    && row.status == RequestStatus::Finalized
                    && row.in_flight.is_none()
            })
            .ok_or(UpdateError {
                request_id,
                reason: "exact endpoint is not finalized",
            })?;
        let request = self.running.remove(index).expect("validated request");
        let publish_tail = request
            .ticket
            .as_ref()
            .is_some_and(|ticket| self.sessions.is_latest(ticket));
        self.kv
            .finish_request(&request, FinishOutcome::Succeeded { publish_tail });
        self.release_request(&request);
        Ok(())
    }

    fn release_request(&mut self, request: &crate::Request) {
        self.kv.free(request.id);
        self.paused.remove(&request.id);
        if let Some(ticket) = &request.ticket {
            self.sessions.release(ticket);
        }
        self.finished_ids.push(request.id);
    }

    pub fn abort(&mut self, request_id: u64) {
        self.paused.remove(&request_id);
        if let Some(index) = self.running.iter().position(|row| row.id == request_id) {
            if self.running[index].in_flight.is_some() && self.running[index].submitted {
                self.running[index].status = RequestStatus::Retiring;
                return;
            }
            let request = self.running.remove(index).unwrap();
            self.kv.finish_request(&request, FinishOutcome::Cancelled);
            self.release_request(&request);
        } else if let Some(index) = self.waiting.iter().position(|row| row.id == request_id) {
            let request = self.waiting.remove(index).unwrap();
            self.kv.finish_request(&request, FinishOutcome::Cancelled);
            self.release_request(&request);
        }
    }

    pub fn set_worker_epoch(&mut self, epoch: u64) -> Result<(), UpdateError> {
        if self.num_running() + self.num_waiting() > 0 || epoch < self.worker_epoch {
            return Err(UpdateError {
                request_id: 0,
                reason: "worker epoch requires empty scheduler and monotonic epoch",
            });
        }
        if epoch == self.worker_epoch {
            return Ok(());
        }
        self.kv.reset_worker_epoch(epoch);
        self.worker_epoch = epoch;
        Ok(())
    }

    pub(crate) fn retire_completed(&mut self) {
        let ids = self
            .running
            .iter()
            .filter(|request| {
                request.status == RequestStatus::Retiring && request.in_flight.is_none()
            })
            .map(|request| request.id)
            .collect::<Vec<_>>();
        for id in ids {
            self.abort(id);
        }
    }
}
