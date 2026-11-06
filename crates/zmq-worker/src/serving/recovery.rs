use crate::client::WorkerClient;
use crate::models::ServiceCache;
use crate::serving::engine::run_engine;
use crate::serving::{Active, EngineContext, PendingAdmission, Submission};
use axum::http::StatusCode;
use oh_my_vllm_scheduler::Scheduler;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

pub(super) async fn engine<P: ServiceCache>(
    mut client: WorkerClient,
    mut scheduler: Scheduler<P>,
    mut incoming: mpsc::Receiver<Submission>,
    context: EngineContext,
    mut shutdown: oneshot::Receiver<()>,
) {
    scheduler.cache_policy().attach_client(&mut client);
    let health = &context.health;
    let configuration = client.configuration();
    scheduler.enable_host_cache(client.host_cache_bytes);
    let expected_capacities = (
        client.logical_num_blocks,
        client.mamba_blocks,
        client.host_cache_bytes,
    );
    let model_policy = client.model_policy();
    let template = scheduler.empty_like();
    let mut client = Some(client);
    let mut epoch = 1u64;
    let mut restarts = 0u64;
    let mut recovery = Recovery {
        configuration,
        model_policy,
        expected_capacities,
        launches: 0,
    };
    loop {
        scheduler
            .cache_policy()
            .attach_client(client.as_mut().unwrap());
        let mut active = BTreeMap::new();
        let mut pending = None;
        let (result, stopping) = tokio::select! {
            _ = &mut shutdown => (Ok(()), true),
            result = run_engine(
                client.as_mut().unwrap(), &mut scheduler, &mut incoming,
                &context, epoch, &mut active, &mut pending,
            ) => (result, false),
        };
        if let Err(error) = &result {
            warn!(%error, epoch, "inference engine failed; replacing worker");
        }
        health.lock().unwrap()["status"] = json!(if stopping {
            "stopped"
        } else {
            "recovering"
        });
        reject_epoch(active, pending, &mut incoming, stopping);
        if stopping || result.is_ok() {
            if let Some(worker) = client.as_mut() {
                let _ = worker.shutdown().await;
            }
            break;
        }
        if !terminate_failed(client.as_mut().unwrap(), health, &mut shutdown).await {
            break;
        }
        drop(client.take());
        scheduler = template.empty_like();
        clear_worker_status(health);
        match recovery
            .replace_worker(context.restart_attempts, health, &mut shutdown)
            .await
        {
            Ok(Some(worker)) => {
                client = Some(worker);
                mark_recovered(health, &mut epoch, &mut restarts);
            }
            Err(RecoveryStopped) => return,
            Ok(None) => {
                stop_after_attempts(health, &mut shutdown).await;
                break;
            }
        }
    }
}

fn reject_epoch(
    active: BTreeMap<u64, Active>,
    pending: Option<PendingAdmission>,
    incoming: &mut mpsc::Receiver<Submission>,
    stopping: bool,
) {
    for (_, request) in active {
        request.fail(if stopping {
            "server shutting down"
        } else {
            "worker failed"
        });
    }
    if let Some(pending) = pending {
        let _ = pending.submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "worker unavailable".to_owned(),
        )));
    }
    while let Ok(submission) = incoming.try_recv() {
        let _ = submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "worker unavailable".to_owned(),
        )));
    }
}

async fn terminate_failed(
    worker: &mut WorkerClient,
    health: &Mutex<Value>,
    shutdown: &mut oneshot::Receiver<()>,
) -> bool {
    let worker_pid = worker.process_id();
    if let Err(error) = worker.terminate().await {
        warn!(
            worker_pid,
            %error,
            "could not confirm old worker exit; recovery stopped without replacement"
        );
        health.lock().unwrap()["status"] = json!("stopped");
        let _ = (&mut *shutdown).await;
        return false;
    }
    true
}

fn clear_worker_status(health: &Mutex<Value>) {
    let mut status = health.lock().unwrap();
    status["worker"] = Value::Null;
    status["worker_snapshot_unix_s"] = Value::Null;
    status["active_requests"] = json!(0);
    status["pending_admissions"] = json!(0);
    status["scheduler_running"] = json!(0);
    status["scheduler_waiting"] = json!(0);
}

struct Recovery {
    configuration: crate::client::WorkerConfig,
    model_policy: crate::models::ModelPolicy,
    expected_capacities: (u32, u32, u64),
    launches: u64,
}

struct RecoveryStopped;

impl Recovery {
    async fn replace_worker(
        &mut self,
        attempts: usize,
        health: &Mutex<Value>,
        shutdown: &mut oneshot::Receiver<()>,
    ) -> std::result::Result<Option<WorkerClient>, RecoveryStopped> {
        for attempt in 0..attempts {
            let delay = Duration::from_secs((1u64 << attempt.min(5)).min(30));
            tokio::select! {
                _ = &mut *shutdown => return Err(RecoveryStopped),
                _ = tokio::time::sleep(delay) => {}
            }
            let reaped = tokio::select! {
                _ = &mut *shutdown => return Err(RecoveryStopped),
                result = WorkerClient::wait_for_reapers() => result,
            };
            if reaped.is_err() {
                break;
            }
            self.launches += 1;
            let mut config = self.configuration.clone();
            let mut socket = config.socket_path.into_os_string();
            socket.push(format!(".recovery-{}", self.launches));
            config.socket_path = socket.into();
            let launched = tokio::select! {
                _ = &mut *shutdown => return Err(RecoveryStopped),
                result = WorkerClient::launch_with_model(config, self.model_policy) => result,
            };
            match launched {
                Ok(mut worker) => {
                    if (
                        worker.logical_num_blocks,
                        worker.mamba_blocks,
                        worker.host_cache_bytes,
                    ) != self.expected_capacities
                    {
                        let worker_pid = worker.process_id();
                        warn!(
                            worker_pid,
                            "replacement worker reported changed cache capacities"
                        );
                        if let Err(error) = worker.terminate().await {
                            warn!(
                                worker_pid,
                                %error,
                                "could not confirm replacement worker exit; \
                                 recovery stopped without replacement"
                            );
                            health.lock().unwrap()["status"] = json!("stopped");
                            let _ = (&mut *shutdown).await;
                            return Err(RecoveryStopped);
                        }
                        continue;
                    }
                    return Ok(Some(worker));
                }
                Err(error) => warn!(%error, attempt, "GPU worker replacement failed"),
            }
        }
        Ok(None)
    }
}

fn mark_recovered(health: &Mutex<Value>, epoch: &mut u64, restarts: &mut u64) {
    *epoch += 1;
    *restarts += 1;
    let mut status = health.lock().unwrap();
    status["status"] = json!("running");
    status["epoch"] = json!(*epoch);
    status["restarts"] = json!(*restarts);
    info!(
        epoch = *epoch,
        restarts = *restarts,
        "GPU worker recovery complete"
    );
}

async fn stop_after_attempts(health: &Mutex<Value>, shutdown: &mut oneshot::Receiver<()>) {
    health.lock().unwrap()["status"] = json!("stopped");
    warn!("GPU worker recovery attempt limit exhausted");
    let _ = (&mut *shutdown).await;
}
