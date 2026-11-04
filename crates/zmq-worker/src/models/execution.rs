use std::sync::{Arc, Mutex};

use oh_my_vllm_model_contract::{ModelDescriptor, ParallelStrategy};
use oh_my_vllm_scheduler::{Request, output::ScheduledRequest};

use super::{ModelFamily, deepseek_v41::ownership::RequestOwners};
use crate::error::Result;
use crate::protocol::{RankCompletion, RankSchedule};

pub(crate) struct ModelExecution {
    pub(crate) owners: Option<Arc<Mutex<RequestOwners>>>,
    snapshots: std::collections::BTreeMap<u64, u32>,
    pending_drops: std::collections::BTreeMap<u64, u64>,
}

impl ModelExecution {
    pub(crate) fn new(family: ModelFamily, descriptor: &ModelDescriptor) -> Self {
        Self {
            owners: (family == ModelFamily::DeepSeekV41
                && descriptor.parallel_plan.strategy == ParallelStrategy::Dep4)
                .then(|| Arc::new(Mutex::new(RequestOwners::default()))),
            snapshots: Default::default(),
            pending_drops: Default::default(),
        }
    }

    pub(crate) fn owner(&mut self, id: u64) -> Option<u32> {
        self.owners
            .as_mut()
            .map(|owners| owners.lock().unwrap().assign(id))
    }

    pub(crate) fn needs_rank_ack(&self) -> bool {
        self.owners.is_some()
    }

    pub(crate) fn abort(&mut self, id: u64, ranks: &[u32]) -> Result<()> {
        if let Some(owners) = &self.owners {
            owners.lock().unwrap().abort(id, ranks)?;
        }
        Ok(())
    }

    pub(crate) fn bind_cache(&mut self, request: &mut Request) {
        if let Some(owners) = &mut self.owners {
            owners.lock().unwrap().bind_cache(request);
        }
    }

    pub(crate) fn schedule(
        &mut self,
        step: u64,
        scheduled: &[ScheduledRequest],
        finished: &[u64],
        preempted: &[u64],
    ) -> Result<Option<Vec<RankSchedule>>> {
        self.owners
            .as_mut()
            .map(|owners| {
                owners
                    .lock()
                    .unwrap()
                    .schedule(step, scheduled, finished, preempted)
            })
            .transpose()
    }

    pub(crate) fn route_cache(
        &mut self,
        operations: &mut [oh_my_vllm_scheduler::cache::CacheOperation],
    ) -> Result<()> {
        if self.owners.is_none() {
            return Ok(());
        }
        for operation in operations {
            let rank = operation
                .state
                .owner_rank
                .or_else(|| self.snapshots.get(&operation.snapshot_id).copied())
                .ok_or_else(|| {
                    crate::error::Error::UnexpectedMessageType("DEP4 snapshot has no owner".into())
                })?;
            if rank >= 4
                || self
                    .snapshots
                    .get(&operation.snapshot_id)
                    .is_some_and(|saved| *saved != rank)
            {
                return Err(crate::error::Error::UnexpectedMessageType(
                    "DEP4 snapshot changed owner".into(),
                ));
            }
            self.snapshots.insert(operation.snapshot_id, rank);
            if operation.kind == oh_my_vllm_scheduler::cache::CacheKind::Drop {
                self.pending_drops
                    .insert(operation.transfer_id, operation.snapshot_id);
            }
            operation.state.owner_rank = Some(rank);
        }
        Ok(())
    }

    pub(crate) fn complete_cache(
        &mut self,
        ranks: &[u32],
        completed: &[oh_my_vllm_scheduler::cache::CacheCompletion],
    ) -> Result<()> {
        if self.owners.is_some() && ranks != [0, 1, 2, 3] {
            return Err(crate::error::Error::UnexpectedMessageType(
                "DEP4 cache completion is missing a rank".into(),
            ));
        }
        for completion in completed {
            if completion.status == oh_my_vllm_scheduler::cache::CacheStatus::Complete
                && let Some(snapshot) = self.pending_drops.remove(&completion.transfer_id)
            {
                self.snapshots.remove(&snapshot);
            }
        }
        Ok(())
    }

    pub(crate) fn complete(&mut self, step: u64, replies: &[RankCompletion]) -> Result<()> {
        if let Some(owners) = &mut self.owners {
            owners.lock().unwrap().complete(step, replies)?;
        }
        Ok(())
    }
}
