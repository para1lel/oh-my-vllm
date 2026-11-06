use crate::group::model::{ModelPayload, ModelState};
use crate::group::protocol::StepPlan;
use crate::rank_client::GroupClient;
use anyhow::{Context, Result, ensure};
use oh_my_vllm_kv_cache::state::{StateBlock, StateTransaction};
use oh_my_vllm_model_contract::{CacheSource, RequestPlacement};
use oh_my_vllm_scheduler::policy::TransactionalCachePolicy;
use oh_my_vllm_scheduler::{Scheduler, SchedulerOutput};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

impl GroupClient {
    pub(super) fn validate_model_schedule<P, L>(
        &self,
        scheduler: &Scheduler<P>,
        batch: &SchedulerOutput<ModelState<L>>,
        plan: &StepPlan,
    ) -> Result<()>
    where
        P: TransactionalCachePolicy<State = ModelState<L>>,
        L: Clone + Debug,
    {
        scheduler.validate_schedule(batch)?;
        ensure!(
            plan.requests.len() == batch.scheduled.len(),
            "model request count differs"
        );
        for (work, request) in plan.requests.iter().zip(&batch.scheduled) {
            ensure!(
                work.request_id == request.request_id
                    && work.rows as usize == request.token_ids.len()
                    && Some(work.prefill) == scheduler.scheduled_prefill(request.request_id),
                "model dispatch differs from scheduled rows"
            );
        }
        Ok(())
    }

    pub(super) fn validate_model_inputs<L>(
        &self,
        payload: &ModelPayload<'_, L>,
        plan: &StepPlan,
    ) -> Result<()> {
        ensure!(
            plan.requests.len() == payload.requests.len(),
            "model request count differs"
        );
        let descriptor = self.gate.descriptor();
        for (work, request) in plan.requests.iter().zip(&payload.requests) {
            ensure!(
                work.request_id == request.request_id
                    && work.rows as usize == request.token_ids.len(),
                "model dispatch differs from input rows"
            );
            ensure!(
                request
                    .num_computed_tokens
                    .checked_add(request.token_ids.len())
                    .is_some_and(|end| end as u64 <= descriptor.max_context_tokens)
                    && request
                        .token_ids
                        .iter()
                        .all(|&token| token < descriptor.vocabulary_size),
                "model input exceeds context or vocabulary"
            );
        }
        Ok(())
    }

    pub(super) fn validate_bindings<'a, L>(
        &self,
        payload: &ModelPayload<'_, L>,
        plan: &StepPlan,
        transactions: impl Iterator<Item = &'a StateTransaction>,
    ) -> Result<()> {
        let transactions: BTreeMap<_, _> = transactions
            .map(|transaction| (transaction.key().request_id, transaction))
            .collect();
        for (work, request) in plan.requests.iter().zip(&payload.requests) {
            let transaction = transactions
                .get(&request.request_id)
                .context("missing model transaction")?;
            ensure!(
                request.state.transaction == transaction.key(),
                "model transaction differs"
            );
            ensure!(
                request.state.row_grants == transaction.row_grants(),
                "model row grants differ from live transaction grants"
            );
            for grant in &request.state.row_grants {
                self.validate_block(&grant.root, transaction, work.owner_rank)?;
            }
            let writes = request.state.writes.iter().collect::<BTreeSet<_>>();
            ensure!(
                writes.len() == request.state.writes.len()
                    && writes == transaction.blocks().iter().collect(),
                "model writes differ from transaction allocations"
            );
            let reads = request.state.reads.iter().collect::<BTreeSet<_>>();
            let authorized = transaction
                .read_blocks()
                .iter()
                .chain(transaction.blocks())
                .collect::<BTreeSet<_>>();
            ensure!(
                reads.len() == request.state.reads.len() && reads.is_subset(&authorized),
                "model reads exceed transaction capabilities"
            );
            for block in reads.union(&writes) {
                self.validate_block(block, transaction, work.owner_rank)?;
            }
        }
        Ok(())
    }

    fn validate_block(
        &self,
        block: &StateBlock,
        transaction: &StateTransaction,
        owner_rank: u32,
    ) -> Result<()> {
        let descriptor = self.gate.descriptor();
        let schema = &self
            .capacities()
            .context("group has no capacities")?
            .common_schema;
        let group = schema
            .groups
            .iter()
            .find(|group| group.name == block.group)
            .context("model state has an unknown group")?;
        ensure!(
            matches!(group.source, CacheSource::Owned {})
                && block.rank < descriptor.parallel_plan.world_size
                && (descriptor.parallel_plan.request_placement == RequestPlacement::Replicated
                    || block.rank == owner_rank)
                && block.slot > 0
                && u64::from(block.slot) < group.capacity_blocks
                && block.generation > 0
                && transaction.key().epoch == self.epoch,
            "model state differs from negotiated storage"
        );
        Ok(())
    }
}
