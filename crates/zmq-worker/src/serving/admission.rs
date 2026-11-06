use crate::client::WorkerClient;
use crate::error::Error as WorkerError;
use crate::models::ServiceCache;
use crate::models::output::Parser;
use crate::serving::events::Output;
use crate::serving::request::Request;
use crate::serving::resources::{Lease, unix_seconds};
use crate::serving::stream::StreamBuffer;
use crate::serving::{Active, IDS, PendingAdmission, Submission, media, now, send_events};
use anyhow::Result;
use axum::http::StatusCode;
use oh_my_vllm_scheduler::{Request as EngineRequest, Scheduler};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

pub(super) async fn begin_admit(
    client: &mut WorkerClient,
    active_count: usize,
    max_running: usize,
    submission: Submission,
    epoch: u64,
) -> Result<Option<PendingAdmission>> {
    if submission.ready.is_closed() {
        return Ok(None);
    }
    if submission.lease.epoch() != epoch {
        let _ = submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "request belongs to a failed worker epoch".to_owned(),
        )));
        return Ok(None);
    }
    if active_count >= max_running {
        let _ = submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "active request limit exceeded".to_owned(),
        )));
        return Ok(None);
    }
    if let Err(error) = submission.lease.start() {
        let _ = submission
            .ready
            .send(Err((StatusCode::SERVICE_UNAVAILABLE, error.to_string())));
        return Ok(None);
    }
    let id = IDS.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let started_unix_s = unix_seconds();
    let prepare_timeout_ms = std::env::var("OH_MY_VLLM_PREPARE_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(120_000);
    let deadline = submission
        .lease
        .deadline()
        .min(started + Duration::from_millis(prepare_timeout_ms));
    match tokio::time::timeout_at(
        tokio::time::Instant::from_std(deadline),
        client.begin_prepare_request(id, submission.request.normalized.clone()),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let _ = submission.ready.send(Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "worker unavailable".to_owned(),
            )));
            return Err(error.into());
        }
        Err(_) => {
            client.cancel_prepare(id).await?;
            let _ = submission.ready.send(Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "worker preparation timeout".to_owned(),
            )));
            return Ok(None);
        }
    }
    Ok(Some(PendingAdmission {
        submission,
        id,
        started,
        deadline,
        started_unix_s,
    }))
}

pub(super) async fn cancel_admit(
    client: &mut WorkerClient,
    pending: PendingAdmission,
) -> Result<()> {
    let disconnected = pending.submission.ready.is_closed();
    client.cancel_prepare(pending.id).await?;
    if !disconnected {
        let _ = pending.submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "worker preparation timeout".to_owned(),
        )));
    }
    Ok(())
}

pub(super) async fn wait_admit<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
    active: &mut BTreeMap<u64, Active>,
    pending: &mut Option<PendingAdmission>,
) -> Result<()> {
    let Some(preparing) = pending.as_mut() else {
        return Ok(());
    };
    let result = tokio::select! {
        biased;
        _ = preparing.submission.ready.closed() => None,
        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(preparing.deadline)) => None,
        result = client.wait_prepare_request() => Some(result?),
    };
    let preparing = pending.take().unwrap();
    if let Some(result) = result {
        complete_admit(client, scheduler, active, preparing, result).await
    } else {
        cancel_admit(client, preparing).await
    }
}

pub(super) async fn complete_admit<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
    active: &mut BTreeMap<u64, Active>,
    pending: PendingAdmission,
    prepare_result: std::result::Result<Vec<u32>, WorkerError>,
) -> Result<()> {
    let PendingAdmission {
        submission:
            Submission {
                request,
                enqueued,
                events,
                done,
                ready,
                lease,
            },
        id,
        started,
        started_unix_s,
        ..
    } = pending;
    let pending_times = (enqueued, started, started_unix_s);
    let Some((tokens, ready)) = prepared_tokens(client, id, prepare_result, ready).await? else {
        return Ok(());
    };
    if ready.is_closed() {
        client.cancel_prepare(id).await?;
        return Ok(());
    }
    let max_tokens = request.normalized["max_tokens"].as_u64().unwrap() as usize;
    let mut running = active_request(
        client,
        id,
        tokens.len(),
        pending_times,
        (request, events, done, lease),
    );
    let start_events = running.output.start();
    let mut engine_request = EngineRequest::new(id, tokens, max_tokens, vec![]);
    isolate_media_prefix(&mut engine_request, &running.output.request, id);
    client.bind_request_cache(&mut engine_request);
    if !scheduler.add_request(engine_request) {
        client.cancel_prepare(id).await?;
        let _ = ready.send(Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "request FA/GDN or output budget exceeds KV pool capacity".to_owned(),
        )));
        return Ok(());
    }
    if ready.send(Ok(())).is_err() {
        scheduler.abort(id);
        client.cancel_scheduled_request(id).await?;
        running.fail_with_status("client disconnected", "cancelled");
        return Ok(());
    }
    if let Err(error) = send_events(&mut running, start_events) {
        scheduler.abort(id);
        client.cancel_scheduled_request(id).await?;
        running.fail_delivery(&error);
        return Ok(());
    }
    if running.stream_buffer.is_blocked() {
        scheduler.pause(id);
    }
    info!(
        request_id = id,
        input_tokens = running.output.input_tokens,
        max_tokens,
        queue_ms = started.duration_since(enqueued).as_millis(),
        prepare_ms = started.elapsed().as_millis(),
        "request admitted"
    );
    active.insert(id, running);
    Ok(())
}

async fn prepared_tokens(
    client: &mut WorkerClient,
    id: u64,
    prepare_result: std::result::Result<Vec<u32>, WorkerError>,
    ready: ReadySender,
) -> Result<Option<(Vec<u32>, ReadySender)>> {
    match prepare_result {
        Err(e) => {
            if matches!(e, WorkerError::WorkerOom(_) | WorkerError::WorkerDevice(_)) {
                let _ = ready.send(Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    "worker recovery required".to_owned(),
                )));
                return Err(e.into());
            }
            let status = if matches!(e, WorkerError::WorkerValidation(_)) {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            client.cancel_prepare(id).await?;
            let _ = ready.send(Err((status, e.to_string())));
            Ok(None)
        }
        Ok(tokens) => Ok(Some((tokens, ready))),
    }
}

fn active_request(
    client: &WorkerClient,
    id: u64,
    input_tokens: usize,
    times: (Instant, Instant, f64),
    delivery: ActiveDelivery,
) -> Active {
    let (request, events, done, lease) = delivery;
    let (enqueued, started, started_unix_s) = times;
    let parser = Parser::new(
        client.model_family(),
        request.normalized["effort"] != "off",
        if request.normalized["tool_choice"] == "none"
            || request.normalized["format"]["type"] != "text"
        {
            vec![]
        } else {
            request.normalized["tools"].as_array().unwrap().clone()
        },
    );
    let mut output = Output::new(
        format!(
            "{}_{}_{:x}_{id}",
            if request.responses {
                "resp"
            } else {
                "chatcmpl"
            },
            std::process::id(),
            now()
        ),
        request,
    );
    output.input_tokens = input_tokens;
    Active {
        output,
        parser,
        events,
        stream_buffer: StreamBuffer::default(),
        done,
        started: enqueued,
        first_token: None,
        steps: 0,
        proposed: 0,
        accepted: 0,
        lease,
        processing_started: started,
        processing_started_unix_s: started_unix_s,
    }
}

fn isolate_media_prefix(engine_request: &mut EngineRequest, request: &Request, id: u64) {
    if request.normalized["messages"]
        .as_array()
        .is_some_and(|messages| {
            messages
                .iter()
                .any(|message| media::count(&message["content"]) > 0)
        })
    {
        engine_request.record_partial_prefix = false;
        engine_request.cache_salt = Some(oh_my_vllm_kv_cache::hash::hash_block_tokens(
            None,
            &[],
            &[0x6d65646961, id],
        ));
    }
}

type ReadySender = oneshot::Sender<std::result::Result<(), (StatusCode, String)>>;
type ActiveDelivery = (
    Request,
    mpsc::Sender<Value>,
    oneshot::Sender<std::result::Result<Value, String>>,
    Lease,
);
