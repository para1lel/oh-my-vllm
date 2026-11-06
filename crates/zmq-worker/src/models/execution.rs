use std::sync::{Arc, Mutex};

use oh_my_vllm_model_contract::{ModelDescriptor, ParallelStrategy};
use oh_my_vllm_scheduler::{Request, output::ScheduledRequest};

use super::{ModelFamily, deepseek_v41::ownership::RequestOwners};
use crate::error::Result;
use crate::protocol::{RankCompletion, RankSchedule};

pub(crate) struct ModelExecution {
    pub(crate) owners: Option<Arc<Mutex<RequestOwners>>>,
}

impl ModelExecution {
    pub(crate) fn new(family: ModelFamily, descriptor: &ModelDescriptor) -> Self {
        Self {
            owners: (family == ModelFamily::DeepSeekV41
                && descriptor.parallel_plan.strategy == ParallelStrategy::Dep4)
                .then(|| Arc::new(Mutex::new(RequestOwners::default()))),
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

    pub(crate) fn schedule<S>(
        &mut self,
        step: u64,
        scheduled: &[ScheduledRequest<S>],
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

    pub(crate) fn complete(&mut self, step: u64, replies: &[RankCompletion]) -> Result<()> {
        if let Some(owners) = &mut self.owners {
            owners.lock().unwrap().complete(step, replies)?;
        }
        Ok(())
    }
}
