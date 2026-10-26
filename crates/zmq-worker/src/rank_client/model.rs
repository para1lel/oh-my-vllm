use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::io::{self, Write};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use oh_my_vllm_kv_cache::state::{StateBlock, StateTransaction};
use oh_my_vllm_model_contract::{CacheSource, ModelDescriptor, RequestPlacement};
use oh_my_vllm_scheduler::policy::TransactionalCachePolicy;
use oh_my_vllm_scheduler::{RequestOutput, Scheduler, SchedulerOutput, WorkerOutput};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::group::model::{
    MAX_MODEL_PAYLOAD_BYTES, ModelCommand, ModelInput, ModelPayload, ModelState,
};
use crate::group::protocol::{StepDispatch, StepPlan};

use super::{GroupClient, deadline, protocol};

/// A complete rank exchange, not an accepted-state or generation decision.
pub struct BorrowedModelResult {
    epoch: u64,
    step_id: u64,
    payload_sha256: String,
    output: WorkerOutput,
}

impl BorrowedModelResult {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn step_id(&self) -> u64 {
        self.step_id
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn output(&self) -> &WorkerOutput {
        &self.output
    }
}

fn model_payload<L>(batch: &SchedulerOutput<ModelState<L>>) -> ModelPayload<'_, L> {
    ModelPayload {
        requests: batch
            .scheduled
            .iter()
            .map(|request| ModelInput {
                request_id: request.request_id,
                token_ids: &request.token_ids,
                num_computed_tokens: request.num_computed_tokens,
                state: &request.state,
            })
            .collect(),
        finished_request_ids: &batch.finished_request_ids,
        preempted_request_ids: &batch.preempted_request_ids,
    }
}

struct BoundedBytes {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("model message exceeds byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode(value: &impl Serialize, limit: usize) -> Result<Vec<u8>> {
    let mut writer = BoundedBytes {
        bytes: Vec::new(),
        limit,
    };
    value.serialize(super::encoding::Checked(
        &mut rmp_serde::Serializer::new(&mut writer).with_struct_map(),
    ))?;
    Ok(writer.bytes)
}

fn validate_outputs(descriptor: &ModelDescriptor, outputs: &[RequestOutput]) -> Result<()> {
    let drafts = descriptor
        .speculation
        .iter()
        .map(|spec| spec.max_draft_tokens as usize)
        .max()
        .unwrap_or(0);
    for output in outputs {
        ensure!(
            output.token_ids.len() <= drafts + 1
                && output.new_draft_token_ids.len() <= drafts
                && output.num_accepted_draft_tokens <= drafts,
            "model result exceeds speculation limits"
        );
        ensure!(
            output
                .token_ids
                .iter()
                .chain(&output.new_draft_token_ids)
                .all(|&token| token < descriptor.vocabulary_size),
            "model result token exceeds vocabulary"
        );
    }
    Ok(())
}

impl GroupClient {
    /// Exchange model commands without taking or publishing the caller's transactions.
    ///
    /// The caller supplies the model phase (`prefill`); this method has no prompt history.
    /// Success proves rank completion only. It does not validate stop or RNG decisions.
    /// On error or cancellation, keep transactions, steps, snapshots and device owners
    /// alive until `close().await` succeeds. If cleanup fails, retain them and stop the epoch.
    /// The Drop reaper cannot retain these borrowed owners. A cancelled exchange blocks
    /// further execution on this group, even if its rank replies arrive later.
    pub async fn exchange_model_borrowed<L: Serialize>(
        &mut self,
        plan: StepPlan,
        batch: &SchedulerOutput<ModelState<L>>,
        transactions: &[&StateTransaction],
    ) -> Result<BorrowedModelResult> {
        ensure!(
            !self.closed
                && self.pending.is_empty()
                && !self.borrowed_in_flight
                && self.gate.is_ready(),
            "group is closed or a step is pending"
        );
        self.validate_transactions(&plan, transactions.iter().copied())?;
        let payload = model_payload(batch);
        self.validate_model_inputs(&payload, &plan)?;
        self.validate_bindings(&payload, &plan, transactions.iter().copied())?;
        let rows = payload
            .requests
            .iter()
            .try_fold(0_usize, |total, request| {
                total
                    .checked_add(request.token_ids.len())
                    .context("model row overflow")
            })?;
        ensure!(
            rows == batch.num_batched_tokens,
            "model batch row total differs"
        );
        let mut ids = plan
            .requests
            .iter()
            .map(|work| work.request_id)
            .collect::<BTreeSet<_>>();
        for id in batch
            .finished_request_ids
            .iter()
            .chain(&batch.preempted_request_ids)
        {
            ensure!(ids.insert(*id), "model request lifecycle sets overlap");
        }
        let step_id = plan.step_id;
        // Cancellation must not permit reuse after the gate receives the last rank reply.
        self.borrowed_in_flight = true;
        let (output, payload_sha256) = self.exchange_model_payload(plan, &payload).await?;
        self.borrowed_in_flight = false;
        Ok(BorrowedModelResult {
            epoch: self.epoch,
            step_id,
            payload_sha256,
            output,
        })
    }

    pub async fn execute_scheduled<P, L>(
        &mut self,
        scheduler: &mut Scheduler<P>,
        batch: SchedulerOutput<ModelState<L>>,
        plan: StepPlan,
    ) -> Result<WorkerOutput>
    where
        P: TransactionalCachePolicy<State = ModelState<L>>,
        L: Serialize + Clone + Debug,
    {
        let result = self.execute_model(scheduler, batch, plan).await;
        if result.is_err() {
            scheduler.fail_execution();
            self.close()
                .await
                .context("model execution cleanup failed")?;
        }
        result
    }

    async fn execute_model<P, L>(
        &mut self,
        scheduler: &mut Scheduler<P>,
        batch: SchedulerOutput<ModelState<L>>,
        plan: StepPlan,
    ) -> Result<WorkerOutput>
    where
        P: TransactionalCachePolicy<State = ModelState<L>>,
        L: Serialize + Clone + Debug,
    {
        ensure!(
            !self.closed
                && self.pending.is_empty()
                && !self.borrowed_in_flight
                && self.gate.is_ready(),
            "group is closed or a step is pending"
        );
        self.validate_model_schedule(scheduler, &batch, &plan)?;
        let payload = model_payload(&batch);
        self.validate_model_inputs(&payload, &plan)?;
        // The group retains these owners if any subsequent await is cancelled.
        self.pending = scheduler.detach_state(&batch)?;
        self.validate_transactions(&plan, self.pending.iter())?;
        self.validate_bindings(&payload, &plan, self.pending.iter())?;
        let (worker, _) = self.exchange_model_payload(plan, &payload).await?;
        Ok(scheduler.apply_completed(worker, std::mem::take(&mut self.pending))?)
    }

    async fn exchange_model_payload<L: Serialize>(
        &mut self,
        plan: StepPlan,
        payload: &ModelPayload<'_, L>,
    ) -> Result<(WorkerOutput, String)> {
        let bytes = encode(&payload, MAX_MODEL_PAYLOAD_BYTES)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let end = deadline(self.step_timeout)?;
        let dispatch = self.gate.begin(plan, end, Instant::now())?;
        let command = ModelCommand {
            kind: "execute_model",
            dispatch: &dispatch,
            payload_sha256: &digest,
            payload: &bytes,
        };
        let command = encode(&command, MAX_MODEL_PAYLOAD_BYTES + (1 << 20))?;
        let mut outputs = self
            .collect_model_results(&dispatch, &command, &digest, end)
            .await?;
        let worker = WorkerOutput {
            outputs: payload
                .requests
                .iter()
                .map(|request| {
                    outputs
                        .remove(&request.request_id)
                        .context("missing model result")
                })
                .collect::<Result<_>>()?,
        };
        Ok((worker, digest))
    }

    fn validate_model_schedule<P, L>(
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

    fn validate_model_inputs<L>(
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

    fn validate_bindings<'a, L>(
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

    async fn collect_model_results(
        &mut self,
        dispatch: &StepDispatch,
        command: &[u8],
        digest: &str,
        end: Instant,
    ) -> Result<BTreeMap<u64, RequestOutput>> {
        for rank in 0..self.senders.len() as u32 {
            self.send_bytes(rank, command, end, false).await?;
        }
        let mut outputs = BTreeMap::new();
        loop {
            let (rank, reply) = self.receive().await?;
            let result = match reply {
                protocol::Reply::ModelResult(result) => result,
                protocol::Reply::Error { header, message } => {
                    bail!("rank {rank} ({header:?}): {message}")
                }
                _ => bail!("unexpected reply during model execution"),
            };
            ensure!(
                result.payload_sha256 == digest,
                "model payload digest differs"
            );
            let expected: BTreeSet<_> = dispatch.ranks[rank as usize]
                .requests
                .iter()
                .filter(|request| request.owner_rank == rank)
                .map(|request| request.request_id)
                .collect();
            let actual: BTreeSet<_> = result
                .outputs
                .iter()
                .map(|output| output.request_id)
                .collect();
            ensure!(
                actual.len() == result.outputs.len() && actual == expected,
                "model result set differs from rank ownership"
            );
            validate_outputs(self.gate.descriptor(), &result.outputs)?;
            for output in result.outputs {
                ensure!(
                    outputs.insert(output.request_id, output).is_none(),
                    "duplicate model output"
                );
            }
            if self
                .gate
                .accept_result(rank, result.completion, Instant::now())?
                .is_some()
            {
                return Ok(outputs);
            }
        }
    }
}

#[cfg(test)]
mod encoding_tests {
    use std::cell::Cell;

    use serde::ser::{SerializeMap, SerializeSeq};

    use super::*;

    struct Unknown<'a> {
        visits: &'a Cell<usize>,
        map: bool,
    }

    impl Serialize for Unknown<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if self.map {
                let mut map = serializer.serialize_map(None)?;
                for value in 0..4096_u32 {
                    self.visits.set(self.visits.get() + 1);
                    map.serialize_entry(&value, &value)?;
                }
                map.end()
            } else {
                let mut seq = serializer.serialize_seq(None)?;
                for value in 0..4096_u32 {
                    self.visits.set(self.visits.get() + 1);
                    seq.serialize_element(&value)?;
                }
                seq.end()
            }
        }
    }

    #[test]
    fn bounded_encoding_rejects_unknown_nested_containers_before_buffering() {
        #[derive(Serialize)]
        struct Nested<T> {
            value: T,
        }

        for map in [false, true] {
            let visits = Cell::new(0);
            let value = Nested {
                value: Unknown {
                    visits: &visits,
                    map,
                },
            };
            assert!(encode(&value, 1024).is_err());
            assert_eq!(
                visits.get(),
                0,
                "unknown container buffered before rejection"
            );
        }
        let value = Nested {
            value: vec![1_u32, 2, 3],
        };
        let expected = rmp_serde::to_vec_named(&value).unwrap();
        assert_eq!(encode(&value, expected.len()).unwrap(), expected);
        assert!(encode(&value, expected.len() - 1).is_err());
    }
}
