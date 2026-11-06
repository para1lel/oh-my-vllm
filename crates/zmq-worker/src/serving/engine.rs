pub(super) mod output;
mod trace;

use crate::client::WorkerClient;
use crate::models::ServiceCache;
use crate::serving::admission::{begin_admit, cancel_admit, complete_admit, wait_admit};
use crate::serving::resources::unix_seconds;
use crate::serving::{Active, EngineContext, PendingAdmission, Submission};
use anyhow::{Context, Result};
use oh_my_vllm_scheduler::Scheduler;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

async fn execute<P: ServiceCache>(
    client: &mut WorkerClient,
    batch: &oh_my_vllm_scheduler::SchedulerOutput<P::State>,
    step_timeout: Duration,
) -> Result<oh_my_vllm_scheduler::WorkerOutput> {
    if batch.scheduled.len() > 1 {
        let request_ids: Vec<_> = batch.scheduled.iter().map(|r| r.request_id).collect();
        debug!(?request_ids, "scheduled mixed batch");
    }
    let output = tokio::time::timeout(
        step_timeout,
        client.execute_one_step::<P>(
            &batch.scheduled,
            &batch.finished_request_ids,
            &batch.preempted_request_ids,
            batch.num_batched_tokens,
        ),
    )
    .await
    .with_context(|| format!("worker step exceeded {} seconds", step_timeout.as_secs()))??;
    if !batch.finished_request_ids.is_empty() {
        debug!(request_ids = ?batch.finished_request_ids, "worker requests released");
    }
    Ok(output)
}

pub(super) async fn run_engine<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
    incoming: &mut mpsc::Receiver<Submission>,
    context: &EngineContext,
    epoch: u64,
    active: &mut BTreeMap<u64, Active>,
    pending: &mut Option<PendingAdmission>,
) -> Result<()> {
    let max_running = (client.configuration().max_num_seqs as usize).min(64);
    let admission_limit = max_running.saturating_add(1).min(64);
    let trace = trace::ExecutionTrace::open(client);
    let mut state = EngineLoop {
        trace,
        client,
        scheduler,
        incoming,
        context,
        epoch,
        active,
        pending,
        max_running,
        admission_limit,
    };
    let mut next_diagnostics = Instant::now();
    loop {
        let was_idle = state.active.is_empty();
        if was_idle {
            let cleanup = state.scheduler.schedule();
            state.scheduler.mark_submitted(&cleanup)?;
            execute::<P>(state.client, &cleanup, context.step_timeout).await?;
        }
        update_resource_counts(
            &context.health,
            state.scheduler,
            state.active.len(),
            usize::from(state.pending.is_some()),
        );
        if Instant::now() >= next_diagnostics {
            state.sample_resources().await?;
            next_diagnostics = Instant::now() + context.diagnostics_interval;
        }
        if was_idle && state.pending.is_none() {
            match state.receive_idle(next_diagnostics).await? {
                IdleReceive::Closed => break,
                IdleReceive::Retry => continue,
                IdleReceive::Received => {}
            }
        }
        state.admit_ready(was_idle).await?;
        state.flush_streams();
        state.cancel_expired();
        state.start_queued(was_idle).await?;
        state.execute_step().await?;
    }
    Ok(())
}

fn update_resource_counts<P: ServiceCache>(
    health: &Mutex<Value>,
    scheduler: &Scheduler<P>,
    active: usize,
    pending: usize,
) {
    let mut status = health.lock().unwrap();
    status["active_requests"] = json!(active);
    status["pending_admissions"] = json!(pending);
    status["scheduler_running"] = json!(scheduler.num_running());
    status["scheduler_waiting"] = json!(scheduler.num_waiting());
    status["kv"] = scheduler.cache_policy().resource_snapshot();
}

struct EngineLoop<'a, P: ServiceCache> {
    trace: trace::ExecutionTrace,
    client: &'a mut WorkerClient,
    scheduler: &'a mut Scheduler<P>,
    incoming: &'a mut mpsc::Receiver<Submission>,
    context: &'a EngineContext,
    epoch: u64,
    active: &'a mut BTreeMap<u64, Active>,
    pending: &'a mut Option<PendingAdmission>,
    max_running: usize,
    admission_limit: usize,
}

enum IdleReceive {
    Closed,
    Retry,
    Received,
}

impl<P: ServiceCache> EngineLoop<'_, P> {
    async fn sample_resources(&mut self) -> Result<()> {
        let worker =
            tokio::time::timeout(Duration::from_secs(10), self.client.query_stats()).await??;
        let mut status = self.context.health.lock().unwrap();
        status["worker"] = worker;
        status["worker_pid"] = json!(self.client.process_id());
        status["worker_snapshot_unix_s"] = json!(unix_seconds());
        status["sampled_unix_s"] = json!(unix_seconds());
        info!("SERVING_RESOURCE_SNAPSHOT {}", *status);
        drop(status);
        self.context.store.lock().unwrap().prune();
        Ok(())
    }

    async fn receive_idle(&mut self, next_diagnostics: Instant) -> Result<IdleReceive> {
        let diagnostics_at = tokio::time::Instant::from_std(next_diagnostics);
        let submission = tokio::select! {
            received = self.incoming.recv() => match received {
                Some(submission) => submission,
                None => return Ok(IdleReceive::Closed),
            },
            _ = tokio::time::sleep_until(diagnostics_at) => {
                return Ok(IdleReceive::Retry);
            }
        };
        *self.pending = begin_admit(
            self.client,
            self.scheduler,
            self.active.len(),
            self.admission_limit,
            submission,
            self.epoch,
        )
        .await?;
        Ok(IdleReceive::Received)
    }

    async fn admit_ready(&mut self, was_idle: bool) -> Result<()> {
        if was_idle {
            wait_admit(self.client, self.scheduler, self.active, self.pending).await?;
            // Fix the cohort before waiting: later arrivals must not postpone execution.
            let queued = self
                .incoming
                .len()
                .min(self.max_running.saturating_sub(self.active.len()));
            for _ in 0..queued {
                let Ok(submission) = self.incoming.try_recv() else {
                    break;
                };
                *self.pending = begin_admit(
                    self.client,
                    self.scheduler,
                    self.active.len(),
                    self.admission_limit,
                    submission,
                    self.epoch,
                )
                .await?;
                wait_admit(self.client, self.scheduler, self.active, self.pending).await?;
            }
        } else if let Some(preparing) = self.pending.as_ref() {
            if preparing.submission.ready.is_closed() || Instant::now() >= preparing.deadline {
                cancel_admit(self.client, self.scheduler, self.pending.take().unwrap()).await?;
            } else if let Some(result) = self.client.poll_prepare_request()? {
                let preparing = self.pending.take().unwrap();
                complete_admit(self.client, self.scheduler, self.active, preparing, result).await?;
            }
        }
        Ok(())
    }

    fn flush_streams(&mut self) {
        let mut backpressure_failures = Vec::new();
        for (&id, request) in self.active.iter_mut() {
            if !request.output.request.stream {
                continue;
            }
            if let Err(error) = request.stream_buffer.flush(&request.events) {
                backpressure_failures.push((id, error));
            } else if request.stream_buffer.is_blocked() {
                self.scheduler.pause(id);
            } else {
                self.scheduler.resume(id);
            }
        }
        for (id, error) in backpressure_failures {
            self.scheduler.abort(id);
            if let Some(request) = self.active.remove(&id) {
                request.fail_delivery(&error);
                info!(request_id = id, %error, "stream aborted");
            }
        }
    }

    fn cancel_expired(&mut self) {
        let cancelled: Vec<_> = self
            .active
            .iter()
            .filter(|(_, request)| {
                request.events.is_closed() || Instant::now() >= request.lease.deadline()
            })
            .map(|(&id, _)| id)
            .collect();
        for id in cancelled {
            self.scheduler.abort(id);
            if let Some(request) = self.active.remove(&id) {
                request.fail_with_status("request cancelled or timed out", "cancelled");
                info!(request_id = id, "request cancelled");
            }
        }
    }

    async fn start_queued(&mut self, was_idle: bool) -> Result<()> {
        if !was_idle
            && self.active.len() < self.admission_limit
            && self.pending.is_none()
            && let Ok(submission) = self.incoming.try_recv()
        {
            *self.pending = begin_admit(
                self.client,
                self.scheduler,
                self.active.len(),
                self.admission_limit,
                submission,
                self.epoch,
            )
            .await?;
        }
        Ok(())
    }

    async fn execute_step(&mut self) -> Result<()> {
        for event in self.client.take_prepare_traces() {
            self.trace.prepared(event);
        }
        let batch = self.scheduler.schedule();
        self.finish_finalized();
        self.trace.retain(self.active.keys().copied());
        if batch.cache_hit_tokens > 0 {
            let mut status = self.context.health.lock().unwrap();
            status["prefix_hit_tokens"] = json!(
                status["prefix_hit_tokens"].as_u64().unwrap_or(0) + batch.cache_hit_tokens as u64
            );
        }
        if batch.scheduled.is_empty()
            && batch.finished_request_ids.is_empty()
            && batch.preempted_request_ids.is_empty()
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
            return Ok(());
        }
        self.scheduler.mark_submitted(&batch)?;
        let worker = execute::<P>(self.client, &batch, self.context.step_timeout).await?;
        for scheduled in &batch.scheduled {
            let id = scheduled.request_id;
            if let Some(error) = self.client.serving_errors.remove(&id) {
                self.scheduler.abort(id);
                self.client.serving_outputs.remove(&id);
                if let Some(request) = self.active.remove(&id) {
                    request.fail(&error);
                }
                warn!(request_id = id, %error, "request failed in worker");
            }
        }
        self.scheduler.validate_output(&worker)?;
        let worker = self
            .scheduler
            .update(worker)
            .context("validated worker output changed")?;
        self.trace.validated(
            &batch.scheduled,
            &worker.outputs,
            self.client.take_execution_trace(),
        );
        if batch.scheduled.is_empty() && !self.active.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.process_results(&batch, worker);
        Ok(())
    }
}
