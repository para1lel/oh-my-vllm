use std::collections::BTreeSet;

use oh_my_vllm_kv_cache::state::{StateError, StateTransaction};

use crate::{
    Scheduler, SchedulerOutput, UpdateError, WorkerOutput,
    policy::{CachePolicy, TransactionalCachePolicy},
};

#[derive(Debug, thiserror::Error)]
pub enum StateUpdateError {
    #[error(transparent)]
    Output(#[from] UpdateError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error("invalid scheduler state handoff: {0}")]
    Invalid(&'static str),
}

impl<P: CachePolicy> Scheduler<P> {
    pub fn has_detached_state(&self) -> bool {
        self.detached_state
    }

    pub fn fail_execution(&mut self) {
        self.detached_state = true;
    }

    fn in_flight_ids(&self) -> BTreeSet<u64> {
        self.running
            .iter()
            .filter(|request| request.num_in_flight_tokens > 0)
            .map(|request| request.id)
            .collect()
    }

    pub fn scheduled_prefill(&self, request_id: u64) -> Option<bool> {
        self.running
            .iter()
            .find(|request| request.id == request_id)
            .map(|request| {
                request.num_computed_tokens
                    < request
                        .prompt_len
                        .max(request.token_ids.len().saturating_sub(1))
            })
    }

    pub fn validate_schedule(
        &self,
        batch: &SchedulerOutput<P::State>,
    ) -> Result<(), StateUpdateError> {
        let mut ids = BTreeSet::new();
        let mut total = 0_usize;
        for scheduled in &batch.scheduled {
            let Some(request) = self
                .running
                .iter()
                .find(|request| request.id == scheduled.request_id)
            else {
                return Err(StateUpdateError::Invalid("unknown scheduled request"));
            };
            if !ids.insert(request.id)
                || scheduled.num_computed_tokens != request.num_computed_tokens
                || scheduled.token_ids.len() != request.num_in_flight_tokens
                || !scheduled.token_ids.iter().copied().eq(request
                    .token_ids
                    .iter()
                    .chain(&request.draft_token_ids)
                    .skip(request.num_computed_tokens)
                    .take(request.num_in_flight_tokens)
                    .copied())
            {
                return Err(StateUpdateError::Invalid(
                    "schedule differs from in-flight request",
                ));
            }
            total = total
                .checked_add(scheduled.token_ids.len())
                .ok_or(StateUpdateError::Invalid("scheduled row overflow"))?;
        }
        if ids != self.in_flight_ids() || total != batch.num_batched_tokens {
            return Err(StateUpdateError::Invalid(
                "incomplete schedule or wrong row total",
            ));
        }
        Ok(())
    }
}

impl<P: TransactionalCachePolicy> Scheduler<P> {
    pub fn detach_state(
        &mut self,
        batch: &SchedulerOutput<P::State>,
    ) -> Result<Vec<StateTransaction>, StateUpdateError> {
        if self.detached_state {
            return Err(StateUpdateError::Invalid(
                "execution state is already detached",
            ));
        }
        self.validate_schedule(batch)?;
        self.detached_state = true;
        let ids = batch
            .scheduled
            .iter()
            .map(|request| request.request_id)
            .collect::<Vec<_>>();
        let transactions = self.kv.detach_transactions(&ids)?;
        self.validate_transaction_ids(&transactions)?;
        Ok(transactions)
    }

    fn validate_transaction_ids(
        &self,
        transactions: &[StateTransaction],
    ) -> Result<(), StateUpdateError> {
        let ids = transactions
            .iter()
            .map(|transaction| transaction.key().request_id)
            .collect::<BTreeSet<_>>();
        if ids.len() != transactions.len() || ids != self.in_flight_ids() {
            return Err(StateUpdateError::Invalid(
                "transaction set differs from in-flight requests",
            ));
        }
        Ok(())
    }

    // The caller must prove device completion before returning these owners.
    pub fn apply_completed(
        &mut self,
        worker: WorkerOutput,
        transactions: Vec<StateTransaction>,
    ) -> Result<WorkerOutput, StateUpdateError> {
        if !self.detached_state {
            return Err(StateUpdateError::Invalid("execution state is not detached"));
        }
        self.validate_output(&worker)?;
        self.validate_transaction_ids(&transactions)?;
        self.kv.restore_transactions(transactions)?;
        self.detached_state = false;
        Ok(self.update(worker)?)
    }
}
