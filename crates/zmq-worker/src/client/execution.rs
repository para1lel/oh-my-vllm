use crate::client::{WorkerClient, validate_reply_ids, worker_error};
use crate::error::{Error, Result};
use crate::protocol::{ExecuteMsg, RustMessage, ScheduledRequestMsg};
use oh_my_vllm_scheduler::output::{RequestOutput, ScheduledRequest, WorkerOutput};
use std::time::Instant;
use tracing::debug;

impl WorkerClient {
    pub async fn execute_one_step(
        &mut self,
        scheduled: &[ScheduledRequest],
        finished_request_ids: &[u64],
        preempted_request_ids: &[u64],
        num_batched_tokens: usize,
    ) -> Result<WorkerOutput> {
        if scheduled.is_empty()
            && finished_request_ids.is_empty()
            && preempted_request_ids.is_empty()
        {
            debug!("empty batch — skipping execute");
            return Ok(WorkerOutput {
                outputs: Vec::new(),
            });
        }

        self.step_id += 1;
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        let started = Instant::now();
        let msg = RustMessage::Execute(ExecuteMsg {
            rpc_id,
            step_id: self.step_id,
            scheduled: scheduled
                .iter()
                .map(|s| ScheduledRequestMsg {
                    request_id: s.request_id,
                    token_ids: s.token_ids.clone(),
                    prefill_token_ids: s.prefill_token_ids.clone(),
                    num_computed_tokens: s.num_computed_tokens as u32,
                    fa_block_table: s.state.fa_block_table.clone(),
                    mamba_block_table: s.state.mamba_block_table.clone(),
                    fa_block_copies: s.state.fa_block_copies.clone(),
                    prefix_state_slot: s.state.prefix_state_slot,
                    cache_checkpoint: s.state.cache_checkpoint,
                })
                .collect(),
            finished_request_ids: finished_request_ids.to_vec(),
            preempted_request_ids: preempted_request_ids.to_vec(),
            num_batched_tokens: num_batched_tokens as u32,
            rank_plan: self.execution.schedule(
                self.step_id,
                scheduled,
                finished_request_ids,
                preempted_request_ids,
            )?,
        });
        Self::send_raw(&mut self.sock, &msg).await?;

        let reply = self.recv_for(rpc_id).await?;
        debug!(
            step_id = self.step_id,
            requests = scheduled.len(),
            num_batched_tokens,
            elapsed_us = started.elapsed().as_micros() as u64,
            "execute_round_trip"
        );
        match reply {
            crate::protocol::WorkerMessage::ExecuteResult(r) => self.finish_execution(scheduled, r),
            crate::protocol::WorkerMessage::Error(e) => Err(worker_error(e)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    pub async fn cache_step<P: crate::models::ServiceCache>(
        &mut self,
        scheduler: &mut oh_my_vllm_scheduler::Scheduler<P>,
    ) -> Result<()> {
        if !scheduler.cache_pending() {
            return Ok(());
        }
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        let mut operations = scheduler.cache_operations();
        self.execution.route_cache(&mut operations)?;
        Self::send_raw(&mut self.sock, &RustMessage::Cache { rpc_id, operations }).await?;
        match self.recv_for(rpc_id).await? {
            crate::protocol::WorkerMessage::CacheResult {
                completed,
                host_cache_bytes,
                cache_rank_completions,
                ..
            } => {
                self.execution
                    .complete_cache(&cache_rank_completions, &completed)?;
                if host_cache_bytes != self.host_cache_bytes {
                    return Err(Error::UnexpectedMessageType(
                        "host cache capacity changed".into(),
                    ));
                }
                scheduler
                    .complete_cache(&completed)
                    .map_err(|error| Error::UnexpectedMessageType(error.to_string()))
            }
            crate::protocol::WorkerMessage::Error(error) => Err(Error::WorkerError(error.message)),
            reply => Err(Error::UnexpectedMessageType(format!("{reply:?}"))),
        }
    }

    pub async fn query_stats(&mut self) -> Result<serde_json::Value> {
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        Self::send_raw(&mut self.sock, &RustMessage::Stats { rpc_id }).await?;
        match self.recv_for(rpc_id).await? {
            crate::protocol::WorkerMessage::StatsResult { stats, .. } => {
                if serde_json::to_vec(&stats)
                    .map_err(|error| Error::UnexpectedMessageType(error.to_string()))?
                    .len()
                    > 65536
                {
                    return Err(Error::UnexpectedMessageType(
                        "oversized worker stats".to_owned(),
                    ));
                }
                Ok(stats)
            }
            crate::protocol::WorkerMessage::Error(error) => Err(worker_error(error)),
            other => Err(Error::UnexpectedMessageType(format!(
                "wrong stats reply: {other:?}"
            ))),
        }
    }

    fn finish_execution(
        &mut self,
        scheduled: &[ScheduledRequest],
        r: crate::protocol::ExecuteResultMsg,
    ) -> Result<WorkerOutput> {
        validate_reply_ids(scheduled, &r.outputs)?;
        self.execution.complete(self.step_id, &r.rank_completions)?;
        if !self.serving_enabled
            && let Some(error) = r.outputs.iter().find_map(|output| output.error.as_ref())
        {
            return Err(Error::WorkerError(error.clone()));
        }
        if self.serving_enabled {
            for output in &r.outputs {
                if let Some(error) = &output.error {
                    self.serving_errors.insert(output.request_id, error.clone());
                } else {
                    self.serving_outputs.insert(
                        output.request_id,
                        (
                            output.text.clone(),
                            output.finish_reason.clone(),
                            output.reasoning_tokens,
                        ),
                    );
                }
            }
        }
        Ok(WorkerOutput {
            outputs: r
                .outputs
                .into_iter()
                .filter(|o| o.error.is_none())
                .map(|o| RequestOutput {
                    request_id: o.request_id,
                    token_ids: o.token_ids,
                    num_accepted_draft_tokens: o.num_accepted_draft_tokens as usize,
                    new_draft_token_ids: o.new_draft_token_ids,
                })
                .collect(),
        })
    }
}
