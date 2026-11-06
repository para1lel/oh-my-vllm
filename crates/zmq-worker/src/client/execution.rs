use crate::client::{WorkerClient, worker_error};
use crate::error::{Error, Result};
use crate::models::ServiceCache;
use crate::protocol::{ExecuteMsg, RustMessage, ScheduledRequestMsg};
use oh_my_vllm_scheduler::output::{RequestOutput, ScheduledRequest, WorkerOutput};
use std::time::Instant;
use tracing::debug;

mod codec;
mod resources;
mod validation;

#[cfg(test)]
mod tests;

fn wire_digest(value: &impl serde::Serialize) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_value(value)
        .and_then(|value| serde_json::to_vec(&value))
        .map_err(|error| Error::UnexpectedMessageType(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

impl WorkerClient {
    pub async fn execute_one_step<P: ServiceCache>(
        &mut self,
        scheduled: &[ScheduledRequest<P::State>],
        finished_request_ids: &[u64],
        preempted_request_ids: &[u64],
        num_batched_tokens: usize,
    ) -> Result<WorkerOutput> {
        if scheduled.is_empty()
            && finished_request_ids.is_empty()
            && preempted_request_ids.is_empty()
        {
            return Ok(WorkerOutput {
                outputs: Vec::new(),
            });
        }
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        let rows = scheduled
            .iter()
            .map(codec::encode_row::<P>)
            .collect::<Result<Vec<ScheduledRequestMsg>>>()?;
        self.step_id =
            validation::execution_step(scheduled, self.config.worker_epoch, self.step_id)?;
        let msg = RustMessage::Execute(ExecuteMsg {
            rpc_id,
            step_id: self.step_id,
            worker_epoch: self.config.worker_epoch,
            scheduled: rows,
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
        let started = Instant::now();
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
            crate::protocol::WorkerMessage::ExecuteResult(result) => {
                let RustMessage::Execute(sent) = msg else {
                    unreachable!()
                };
                validation::validate_reply(
                    scheduled,
                    &result,
                    self.config.worker_epoch,
                    self.step_id,
                    self.descriptor.vocabulary_size,
                    self.config.num_speculative_tokens,
                    &self.verified_reasoning_tokens,
                )?;
                resources::validate(
                    &sent.scheduled,
                    &result.outputs,
                    self.config.num_speculative_tokens > 0,
                )?;
                self.finish_execution(scheduled, finished_request_ids, &sent, result)
            }
            crate::protocol::WorkerMessage::Error(error) => Err(worker_error(error)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
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

    fn finish_execution<S>(
        &mut self,
        scheduled: &[ScheduledRequest<S>],
        finished_request_ids: &[u64],
        sent: &ExecuteMsg,
        result: crate::protocol::ExecuteResultMsg,
    ) -> Result<WorkerOutput> {
        if !self.serving_enabled
            && let Some(error) = result
                .outputs
                .iter()
                .find_map(|output| output.error.as_ref())
        {
            return Err(Error::WorkerError(error.clone()));
        }
        self.execution
            .complete(self.step_id, &result.rank_completions)?;
        if std::env::var_os("OH_MY_VLLM_RUST_EXECUTION_TRACE").is_some() {
            self.execution_trace = Some(serde_json::json!({
                "worker_epoch": result.worker_epoch, "step_id": result.step_id,
                "scheduled_sha256": wire_digest(&sent.scheduled)?,
                "outputs_sha256": wire_digest(&result.outputs)?,
            }));
        }
        for request_id in finished_request_ids {
            self.verified_reasoning_tokens.remove(request_id);
        }
        for output in &result.outputs {
            if output.error.is_none()
                && output.purpose == oh_my_vllm_scheduler::execution::WorkPurpose::Advance
            {
                self.verified_reasoning_tokens
                    .insert(output.request_id, output.reasoning_tokens);
            }
        }
        if self.serving_enabled {
            for output in &result.outputs {
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
        let rows: std::collections::BTreeMap<_, _> =
            scheduled.iter().map(|row| (row.request_id, row)).collect();
        Ok(WorkerOutput {
            outputs: result
                .outputs
                .into_iter()
                .filter(|output| output.error.is_none())
                .map(|output| {
                    let row = rows[&output.request_id];
                    RequestOutput {
                        request_id: output.request_id,
                        execution: row.execution,
                        work: row.work,
                        cache_result: codec::cache_result(&output, row.execution),
                        token_ids: output.token_ids,
                        num_accepted_draft_tokens: output.num_accepted_draft_tokens as usize,
                        new_draft_token_ids: output.new_draft_token_ids,
                    }
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod trace_binding {
    #[test]
    fn wire_hash_matches_python_sorted_utf8_json() {
        let value = serde_json::json!({"b": "实际输出", "a": 7});
        assert_eq!(
            super::wire_digest(&value).unwrap(),
            "78a8a0581641da657df7ff0c6f6f6110947aeaeed544d5953519c08b7c7df3ec"
        );
    }
}
