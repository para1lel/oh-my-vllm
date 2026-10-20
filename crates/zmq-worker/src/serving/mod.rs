mod events;
mod media;
mod parser;
mod request;
mod resources;
mod sampling;
#[cfg(all(test, unix))]
mod tests;

use crate::{client::WorkerClient, error::Error as WorkerError};
use anyhow::{Context, Result};
use axum::{
    Extension, Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection},
    http::StatusCode,
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use clap::Args;
use events::Output;
use oh_my_vllm_scheduler::{Request as EngineRequest, Scheduler};
use parser::Parser;
use request::Request;
use resources::{Budget, Lease, serialized_bytes, unix_seconds};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    convert::Infallible,
    future::IntoFuture,
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tracing::{debug, info, warn};

const STREAM_BACKPRESSURE_GRACE: Duration = Duration::from_secs(30);
const STREAM_LOW_WATER: usize = 128;
const MAX_PENDING_STREAM_EVENTS: usize = 1024;
const MAX_PENDING_STREAM_BYTES: usize = 8 << 20;

#[derive(Debug, thiserror::Error)]
#[error("client disconnected")]
struct ClientDisconnected;

#[derive(Args, Clone)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub listen: SocketAddr,
    #[arg(long, default_value = "qwen3.8-27b-fp8")]
    pub served_model_name: String,
    #[arg(long, default_value_t = 3600)]
    pub response_ttl_seconds: u64,
    #[arg(long, default_value_t = 1000)]
    pub response_capacity: usize,
    #[arg(long, default_value_t = 268435456)]
    pub response_max_bytes: usize,
    #[arg(long, default_value_t = 3600)]
    pub request_timeout_seconds: u64,
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..))]
    pub worker_step_timeout_seconds: u64,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true,
        value_parser = clap::value_parser!(i64).range(-1..=1_048_576))]
    pub thinking_token_budget: i64,
    #[arg(long, default_value = "{}", value_parser = sampling::parse_overrides)]
    pub override_generation_config: Value,
    #[arg(long, default_value_t = 35)]
    pub shutdown_grace_seconds: u64,
    #[arg(long, default_value_t = 536870912)]
    pub request_state_max_bytes: usize,
    #[arg(long, default_value_t = 30)]
    pub diagnostics_interval_seconds: u64,
    #[arg(long, default_value_t = 5)]
    pub worker_restart_attempts: usize,
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

static IDS: AtomicU64 = AtomicU64::new(1);

struct Stored {
    response: Value,
    history: Vec<Value>,
    saved: Instant,
    bytes: usize,
}

struct Store {
    entries: BTreeMap<String, Stored>,
    ttl: Duration,
    capacity: usize,
    max_bytes: usize,
    bytes: usize,
}

impl Store {
    fn prune(&mut self) {
        self.entries
            .retain(|_, value| value.saved.elapsed() < self.ttl);
        self.bytes = self.entries.values().map(|s| s.bytes).sum();
    }

    fn insert(&mut self, id: String, response: Value, history: Vec<Value>) -> bool {
        self.prune();
        let bytes =
            response.to_string().len() + serde_json::to_vec(&history).map_or(0, |v| v.len());
        if bytes > self.max_bytes {
            return false;
        }
        while self.entries.len() >= self.capacity || self.bytes + bytes > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, v)| v.saved)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.bytes -= self.entries.remove(&oldest).unwrap().bytes;
        }
        self.bytes += bytes;
        self.entries.insert(
            id,
            Stored {
                response,
                history,
                saved: Instant::now(),
                bytes,
            },
        );
        true
    }
}

#[derive(Clone)]
struct App {
    submit: mpsc::Sender<Submission>,
    store: Arc<Mutex<Store>>,
    model: String,
    thinking_token_budget: i64,
    sampling_defaults: Value,
    budget: Budget,
    health: Arc<Mutex<Value>>,
    request_timeout: Duration,
}

struct Submission {
    request: Request,
    enqueued: Instant,
    events: mpsc::Sender<Value>,
    done: oneshot::Sender<std::result::Result<Value, String>>,
    ready: oneshot::Sender<std::result::Result<(), (StatusCode, String)>>,
    lease: Lease,
}

struct Active {
    output: Output,
    parser: Parser,
    events: mpsc::Sender<Value>,
    stream_buffer: StreamBuffer,
    done: oneshot::Sender<std::result::Result<Value, String>>,
    started: Instant,
    first_token: Option<Instant>,
    steps: usize,
    proposed: usize,
    accepted: usize,
    lease: Lease,
    processing_started: Instant,
    processing_started_unix_s: f64,
}

#[derive(Default)]
struct StreamBuffer {
    pending: VecDeque<(Value, usize)>,
    pending_bytes: usize,
    blocked_since: Option<Instant>,
}

impl StreamBuffer {
    fn is_blocked(&self) -> bool {
        self.blocked_since.is_some()
    }

    fn queue(&mut self, event: Value) -> Result<()> {
        let bytes = event.to_string().len();
        anyhow::ensure!(
            self.pending.len() < MAX_PENDING_STREAM_EVENTS
                && self.pending_bytes.saturating_add(bytes) <= MAX_PENDING_STREAM_BYTES,
            "stream pending buffer exceeds its memory limit"
        );
        self.blocked_since.get_or_insert_with(Instant::now);
        self.pending.push_back((event, bytes));
        self.pending_bytes += bytes;
        Ok(())
    }

    fn flush(&mut self, sender: &mpsc::Sender<Value>) -> Result<()> {
        while let Some((event, bytes)) = self.pending.pop_front() {
            match sender.try_send(event) {
                Ok(()) => self.pending_bytes -= bytes,
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.pending.push_front((event, bytes));
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(ClientDisconnected.into());
                }
            }
        }
        if self.pending.is_empty() && sender.capacity() >= STREAM_LOW_WATER {
            self.blocked_since = None;
        }
        anyhow::ensure!(
            self.blocked_since
                .is_none_or(|since| since.elapsed() < STREAM_BACKPRESSURE_GRACE),
            "stream backpressure exceeded 30-second grace"
        );
        Ok(())
    }

    fn send(&mut self, sender: &mpsc::Sender<Value>, events: Vec<Value>) -> Result<()> {
        self.flush(sender)?;
        for event in events {
            if self.pending.is_empty() {
                match sender.try_send(event) {
                    Ok(()) => continue,
                    Err(mpsc::error::TrySendError::Full(event)) => self.queue(event)?,
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        return Err(ClientDisconnected.into());
                    }
                }
            } else {
                self.queue(event)?;
            }
        }
        self.flush(sender)
    }

    fn drain_after_completion(
        self,
        sender: mpsc::Sender<Value>,
        lease: Option<Lease>,
    ) -> Option<tokio::task::JoinHandle<()>> {
        if self.pending.is_empty() {
            return None;
        }
        let deadline = tokio::time::Instant::from_std(
            self.blocked_since
                .expect("pending stream must have a deadline")
                + STREAM_BACKPRESSURE_GRACE,
        );
        Some(tokio::spawn(async move {
            let _drain = lease.map(|lease| lease.drain());
            for (event, _) in self.pending {
                match tokio::time::timeout_at(deadline, sender.send(event)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => return,
                    Err(_) => {
                        warn!("completed stream exceeded backpressure grace");
                        return;
                    }
                }
            }
        }))
    }
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    let body = json!({
        "error": {
            "message": message.to_string(),
            "type": kind,
            "param": null,
            "code": status.as_u16().to_string(),
        },
    });
    (status, Json(body)).into_response()
}

fn error_response(reason: impl ToString) -> Response {
    error(StatusCode::SERVICE_UNAVAILABLE, reason)
}

async fn reserve_http(
    State(app): State<App>,
    mut request: axum::extract::Request,
    next: Next,
) -> Response {
    let generation = request.method() == axum::http::Method::POST
        && matches!(
            request.uri().path(),
            "/v1/chat/completions" | "/v1/responses"
        );
    let retrieval = request.method() == axum::http::Method::GET
        && request.uri().path().starts_with("/v1/responses/");
    if !generation && !retrieval {
        return next.run(request).await;
    }
    let (status, epoch) = {
        let health = app.health.lock().unwrap();
        (
            health["status"] == "running",
            health["epoch"].as_u64().unwrap(),
        )
    };
    if generation && !status {
        return error_response("worker recovering or unavailable");
    }
    let lease = match app.budget.admit(app.request_timeout, epoch) {
        Ok(lease) => lease,
        Err(error) => return error_response(error),
    };
    if retrieval && let Err(error) = lease.start() {
        return error_response(error);
    }
    let deadline = tokio::time::Instant::from_std(lease.deadline());
    request.extensions_mut().insert(lease.clone());
    let response = match tokio::time::timeout_at(deadline, next.run(request)).await {
        Ok(response) => response,
        Err(_) => return error_response("request deadline exceeded"),
    };
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream().map(move |chunk| {
        let _ownership = &lease;
        chunk
    });
    Response::from_parts(parts, Body::from_stream(stream))
}

async fn diagnostics(State(app): State<App>) -> Json<Value> {
    let mut snapshot = app.health.lock().unwrap().clone();
    snapshot
        .as_object_mut()
        .unwrap()
        .extend(app.budget.snapshot().as_object().unwrap().clone());
    let mut store = app.store.lock().unwrap();
    store.prune();
    snapshot["stored_responses"] = json!(store.entries.len());
    snapshot["stored_response_bytes"] = json!(store.bytes);
    snapshot["cpu"] = json!({"rss_bytes": resources::rss_bytes()});
    Json(snapshot)
}

async fn chat(
    State(app): State<App>,
    Extension(lease): Extension<Lease>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(app, lease, body, false).await
}

async fn responses(
    State(app): State<App>,
    Extension(lease): Extension<Lease>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(app, lease, body, true).await
}

async fn generate(
    app: App,
    lease: Lease,
    body: std::result::Result<Json<Value>, JsonRejection>,
    responses: bool,
) -> Response {
    let body = match body {
        Ok(Json(v)) => v,
        Err(e) => return error(e.status(), e.body_text()),
    };
    let body_bytes = match serialized_bytes(&body) {
        Ok(bytes) => bytes,
        Err(error) => return error_response(error),
    };
    let previous = if let Some(id) = body.get("previous_response_id").filter(|v| !v.is_null()) {
        if !responses {
            return error(
                StatusCode::BAD_REQUEST,
                "previous_response_id requires Responses",
            );
        }
        let Some(id) = id.as_str() else {
            return error(
                StatusCode::BAD_REQUEST,
                "previous_response_id must be a string",
            );
        };
        let mut store = app.store.lock().unwrap();
        store.prune();
        match store.entries.get(id) {
            Some(s) => {
                let history_bytes = serialized_bytes(&s.history).unwrap_or(usize::MAX);
                let reservation = body_bytes
                    .saturating_mul(6)
                    .saturating_add(history_bytes.saturating_mul(2));
                if let Err(error) = lease.resize(reservation) {
                    return error_response(error);
                }
                Some(s.history.clone())
            }
            None => return error(StatusCode::NOT_FOUND, "response not found or expired"),
        }
    } else {
        if let Err(error) = lease.resize(body_bytes.saturating_mul(6)) {
            return error_response(error);
        }
        None
    };
    let mut request = match request::normalize(body, responses, &app.model, previous) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if request.normalized["thinking_token_budget"].is_null() {
        request.normalized["thinking_token_budget"] = json!(app.thinking_token_budget);
    }
    sampling::apply_defaults(&mut request.normalized["sampling"], &app.sampling_defaults);
    let streaming = request.stream;
    let request_bytes = serialized_bytes(&request.original)
        .unwrap_or(usize::MAX)
        .saturating_mul(4)
        .saturating_add(
            serialized_bytes(&request.normalized)
                .unwrap_or(usize::MAX)
                .saturating_mul(2),
        );
    if let Err(error) = lease.resize(request_bytes) {
        return error_response(error);
    }
    let (tx, rx) = mpsc::channel(256);
    let (done, result) = oneshot::channel();
    let (ready, prepared) = oneshot::channel();
    let submission = Submission {
        request,
        enqueued: lease.received(),
        events: tx,
        done,
        ready,
        lease,
    };
    if app.submit.try_send(submission).is_err() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference queue unavailable or full",
        );
    }
    match prepared.await {
        Ok(Ok(())) => {}
        Ok(Err((status, e))) => return error(status, e),
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "worker unavailable"),
    }
    if !streaming {
        let response = match result.await {
            Ok(Ok(v)) => Json(v).into_response(),
            Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
            Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "worker stopped"),
        };
        drop(rx);
        return response;
    }
    let stream = ReceiverStream::new(rx).map(move |value| {
        let event = if value == "[DONE]" {
            Event::default().data("[DONE]")
        } else if responses {
            Event::default()
                .event(value["type"].as_str().unwrap_or("error"))
                .data(value.to_string())
        } else {
            Event::default().data(value.to_string())
        };
        Ok::<_, Infallible>(event)
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": app.model,
                "object": "model",
                "created": 0,
                "owned_by": "oh-my-vllm",
            },
        ],
    }))
}

async fn retrieve(
    State(app): State<App>,
    Extension(lease): Extension<Lease>,
    Path(id): Path<String>,
) -> Response {
    let mut store = app.store.lock().unwrap();
    store.prune();
    match store.entries.get(&id) {
        Some(s) => {
            let bytes = serialized_bytes(&s.response)
                .unwrap_or(usize::MAX)
                .saturating_mul(2);
            if let Err(error) = lease.resize(bytes) {
                return error_response(error);
            }
            Json(s.response.clone()).into_response()
        }
        None => error(StatusCode::NOT_FOUND, "response not found or expired"),
    }
}

async fn delete(State(app): State<App>, Path(id): Path<String>) -> Response {
    let mut store = app.store.lock().unwrap();
    store.prune();
    match store.entries.remove(&id) {
        Some(s) => {
            store.bytes -= s.bytes;
            Json(json!({
                "id": id,
                "object": "response.deleted",
                "deleted": true,
            }))
            .into_response()
        }
        None => error(StatusCode::NOT_FOUND, "response not found or expired"),
    }
}

pub async fn serve(client: WorkerClient, scheduler: Scheduler, args: ServeArgs) -> Result<()> {
    let sampling_defaults = sampling::service_defaults(&args.override_generation_config)?;
    anyhow::ensure!(
        args.response_capacity > 0
            && args.response_max_bytes > 0
            && args.response_ttl_seconds > 0
            && args.request_timeout_seconds > 0
            && args.worker_step_timeout_seconds > 0
            && args.shutdown_grace_seconds > 0,
        "serving limits must be positive"
    );
    anyhow::ensure!(
        args.request_state_max_bytes >= resources::HTTP_BODY_BYTES
            && args.diagnostics_interval_seconds > 0
            && args.worker_restart_attempts > 0,
        "request state, diagnostics, and recovery limits must be positive"
    );
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    let (tx, rx) = mpsc::channel(64);
    let store = Arc::new(Mutex::new(Store {
        entries: BTreeMap::new(),
        ttl: Duration::from_secs(args.response_ttl_seconds),
        capacity: args.response_capacity,
        max_bytes: args.response_max_bytes,
        bytes: 0,
    }));
    let budget = Budget::new(args.request_state_max_bytes);
    let health = Arc::new(Mutex::new(json!({
        "status": "running", "epoch": 1, "restarts": 0,
        "active_requests": 0, "pending_admissions": 0,
        "scheduler_running": 0, "scheduler_waiting": 0,
        "worker": null, "worker_snapshot_unix_s": null,
        "server_pid": std::process::id(), "worker_pid": null, "prefix_hit_tokens": 0,
        "limits": {
            "queued": 64, "live": 64, "http_body_bytes": resources::HTTP_BODY_BYTES,
            "request_state_max_bytes": args.request_state_max_bytes,
            "response_max_bytes": args.response_max_bytes,
            "response_capacity": args.response_capacity,
            "request_timeout_seconds": args.request_timeout_seconds,
            "worker_step_timeout_seconds": args.worker_step_timeout_seconds,
            "thinking_token_budget": args.thinking_token_budget,
            "override_generation_config": sampling_defaults,
            "worker_restart_attempts": args.worker_restart_attempts,
            "diagnostics_interval_seconds": args.diagnostics_interval_seconds,
        },
    })));
    let app = App {
        submit: tx,
        store: store.clone(),
        model: args.served_model_name.clone(),
        thinking_token_budget: args.thinking_token_budget,
        sampling_defaults,
        budget: budget.clone(),
        health: health.clone(),
        request_timeout: Duration::from_secs(args.request_timeout_seconds),
    };
    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{id}", get(retrieve).delete(delete))
        .route("/diagnostics", get(diagnostics))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "unknown endpoint") })
        .layer(DefaultBodyLimit::max(resources::HTTP_BODY_BYTES))
        .layer(middleware::from_fn_with_state(app.clone(), reserve_http))
        .with_state(app);
    info!(listen = %args.listen, model = %args.served_model_name, "HTTP service ready");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let mut engine = tokio::spawn(engine(
        client,
        scheduler,
        rx,
        EngineContext {
            store,
            health,
            diagnostics_interval: Duration::from_secs(args.diagnostics_interval_seconds),
            restart_attempts: args.worker_restart_attempts,
            step_timeout: Duration::from_secs(args.worker_step_timeout_seconds),
        },
        shutdown_rx,
    ));
    let (drain_tx, mut drain_rx) = oneshot::channel();
    let mut server = Box::pin(
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = async {
                        #[cfg(unix)]
                        {
                            let _ = tokio::signal::unix::signal(
                                tokio::signal::unix::SignalKind::terminate(),
                            )
                            .unwrap()
                            .recv()
                            .await;
                        }
                        #[cfg(not(unix))]
                        std::future::pending::<()>().await;
                    } => {}
                }
                let _ = shutdown_tx.send(());
                let _ = drain_tx.send(());
            })
            .into_future(),
    );
    let grace = Duration::from_secs(args.shutdown_grace_seconds);
    let (result, deadline) = tokio::select! {
        result = &mut server => (Some(result), tokio::time::Instant::now() + grace),
        _ = &mut drain_rx => {
            let deadline = tokio::time::Instant::now() + grace;
            let result = tokio::time::timeout_at(deadline, &mut server).await.ok();
            if result.is_none() {
                warn!("HTTP connections did not drain before shutdown deadline");
            }
            (result, deadline)
        }
    };
    drop(server);
    if tokio::time::timeout_at(deadline, &mut engine)
        .await
        .is_err()
    {
        warn!("inference engine did not stop after graceful shutdown");
        engine.abort();
        let _ = engine.await;
    }
    result.map_or(Ok(()), |result| result.context("HTTP service"))
}

fn send_events(active: &mut Active, events: Vec<Value>) -> Result<()> {
    if active.output.request.stream {
        active.stream_buffer.send(&active.events, events)?;
    }
    Ok(())
}

struct PendingAdmission {
    submission: Submission,
    id: u64,
    started: Instant,
    deadline: Instant,
    started_unix_s: f64,
}

async fn begin_admit(
    client: &mut WorkerClient,
    active_count: usize,
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
    if active_count >= 64 {
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

async fn cancel_admit(client: &mut WorkerClient, pending: PendingAdmission) -> Result<()> {
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

async fn wait_admit(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
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

async fn complete_admit(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
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
    let tokens = match prepare_result {
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
            return Ok(());
        }
        Ok(tokens) => tokens,
    };
    if ready.is_closed() {
        client.cancel_prepare(id).await?;
        return Ok(());
    }
    let max_tokens = request.normalized["max_tokens"].as_u64().unwrap() as usize;
    let parser = Parser::new(
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
    output.input_tokens = tokens.len();
    let start_events = output.start();
    let mut running = Active {
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
    };
    let mut engine_request = EngineRequest::new(id, tokens, max_tokens, vec![]);
    if running.output.request.normalized["messages"]
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
        client.cancel_prepare(id).await?;
        running.fail_with_status("client disconnected", "cancelled");
        return Ok(());
    }
    if let Err(error) = send_events(&mut running, start_events) {
        scheduler.abort(id);
        client.cancel_prepare(id).await?;
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

impl Active {
    fn fail(self, message: &str) {
        self.fail_with_status(message, "failed");
    }

    fn fail_delivery(self, error: &anyhow::Error) {
        let status = if error.is::<ClientDisconnected>() {
            "cancelled"
        } else {
            "failed"
        };
        self.fail_with_status(&error.to_string(), status);
    }

    fn fail_with_status(mut self, message: &str, status: &str) {
        self.metric(status);
        let event = self.output.error_event(message);
        let _ = self.events.try_send(event);
        let _ = self.done.send(Err(message.to_owned()));
    }

    fn metric(&self, status: &str) {
        let request_id = self
            .output
            .id
            .rsplit('_')
            .next()
            .and_then(|id| id.parse::<u64>().ok());
        info!(
            "SERVING_REQUEST_METRIC {}",
            json!({
                "request_id": request_id,
                "response_id": self.output.id,
                "api": if self.output.request.responses { "responses" } else { "chat" },
                "status": status,
                "epoch": self.lease.epoch(),
                "received_unix_s": self.lease.received_unix_s(),
                "processing_started_unix_s": self.processing_started_unix_s,
                "generation_completed_unix_s": unix_seconds(),
                "thinking_token_budget": self.output.request.normalized["thinking_token_budget"],
                "sampling": self.output.request.normalized["sampling"],
                "output_tokens": self.output.output_tokens,
                "prefix_hit_tokens": self.output.cached_tokens,
                "queue_seconds": self.processing_started.duration_since(self.started).as_secs_f64(),
                "processing_seconds": self.processing_started.elapsed().as_secs_f64(),
            })
        );
    }
}

#[derive(Clone)]
struct EngineContext {
    store: Arc<Mutex<Store>>,
    health: Arc<Mutex<Value>>,
    diagnostics_interval: Duration,
    restart_attempts: usize,
    step_timeout: Duration,
}

async fn engine(
    client: WorkerClient,
    mut scheduler: Scheduler,
    mut incoming: mpsc::Receiver<Submission>,
    context: EngineContext,
    mut shutdown: oneshot::Receiver<()>,
) {
    let health = &context.health;
    let configuration = client.configuration();
    let expected_capacities = (client.logical_num_blocks, client.mamba_blocks);
    let template = scheduler.empty_like();
    let mut client = Some(client);
    let mut epoch = 1u64;
    let mut restarts = 0u64;
    let mut launches = 0u64;
    loop {
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
        if stopping || result.is_ok() {
            if let Some(worker) = client.as_mut() {
                let _ = worker.shutdown().await;
            }
            break;
        }
        if let Err(error) = client.as_mut().unwrap().terminate().await {
            warn!(%error, "could not confirm old worker exit; recovery stopped");
            health.lock().unwrap()["status"] = json!("stopped");
            let _ = shutdown.await;
            break;
        }
        drop(client.take());
        scheduler = template.empty_like();
        {
            let mut status = health.lock().unwrap();
            status["worker"] = Value::Null;
            status["worker_snapshot_unix_s"] = Value::Null;
            status["active_requests"] = json!(0);
            status["pending_admissions"] = json!(0);
            status["scheduler_running"] = json!(0);
            status["scheduler_waiting"] = json!(0);
        }
        for attempt in 0..context.restart_attempts {
            let delay = Duration::from_secs((1u64 << attempt.min(5)).min(30));
            tokio::select! {
                _ = &mut shutdown => return,
                _ = tokio::time::sleep(delay) => {}
            }
            let reaped = tokio::select! {
                _ = &mut shutdown => return,
                result = WorkerClient::wait_for_reapers() => result,
            };
            if reaped.is_err() {
                break;
            }
            launches += 1;
            let mut config = configuration.clone();
            let mut socket = config.socket_path.into_os_string();
            socket.push(format!(".recovery-{launches}"));
            config.socket_path = socket.into();
            let launched = tokio::select! {
                _ = &mut shutdown => return,
                result = WorkerClient::launch(config) => result,
            };
            match launched {
                Ok(mut worker) => {
                    if (worker.logical_num_blocks, worker.mamba_blocks) != expected_capacities {
                        let _ = worker.terminate().await;
                        warn!("replacement worker reported changed cache capacities");
                        continue;
                    }
                    client = Some(worker);
                    epoch += 1;
                    restarts += 1;
                    let mut status = health.lock().unwrap();
                    status["status"] = json!("running");
                    status["epoch"] = json!(epoch);
                    status["restarts"] = json!(restarts);
                    info!(epoch, restarts, "GPU worker recovery complete");
                    break;
                }
                Err(error) => warn!(%error, attempt, "GPU worker replacement failed"),
            }
        }
        if client.is_none() {
            health.lock().unwrap()["status"] = json!("stopped");
            warn!("GPU worker recovery attempt limit exhausted");
            let _ = shutdown.await;
            break;
        }
    }
}

async fn execute(
    client: &mut WorkerClient,
    batch: &oh_my_vllm_scheduler::SchedulerOutput,
    step_timeout: Duration,
) -> Result<oh_my_vllm_scheduler::WorkerOutput> {
    if batch.scheduled.len() > 1 {
        let request_ids: Vec<_> = batch.scheduled.iter().map(|r| r.request_id).collect();
        debug!(?request_ids, "scheduled mixed batch");
    }
    let output = tokio::time::timeout(
        step_timeout,
        client.execute_one_step(
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

async fn run_engine(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
    incoming: &mut mpsc::Receiver<Submission>,
    context: &EngineContext,
    epoch: u64,
    active: &mut BTreeMap<u64, Active>,
    pending: &mut Option<PendingAdmission>,
) -> Result<()> {
    let store = &context.store;
    let health = &context.health;
    let diagnostics_interval = context.diagnostics_interval;
    let max_running = (client.configuration().max_num_seqs as usize).min(64);
    let mut next_diagnostics = Instant::now();
    loop {
        let was_idle = active.is_empty();
        if was_idle {
            execute(client, &scheduler.schedule(), context.step_timeout).await?;
        }
        update_resource_counts(
            health,
            scheduler,
            active.len(),
            usize::from(pending.is_some()),
        );
        if Instant::now() >= next_diagnostics {
            let worker =
                tokio::time::timeout(Duration::from_secs(10), client.query_stats()).await??;
            let mut status = health.lock().unwrap();
            status["worker"] = worker;
            status["worker_pid"] = json!(client.process_id());
            status["worker_snapshot_unix_s"] = json!(unix_seconds());
            status["sampled_unix_s"] = json!(unix_seconds());
            info!("SERVING_RESOURCE_SNAPSHOT {}", *status);
            drop(status);
            store.lock().unwrap().prune();
            next_diagnostics = Instant::now() + diagnostics_interval;
        }
        if was_idle && pending.is_none() {
            let diagnostics_at = tokio::time::Instant::from_std(next_diagnostics);
            let submission = tokio::select! {
                received = incoming.recv() => match received {
                    Some(submission) => submission,
                    None => break,
                },
                _ = tokio::time::sleep_until(diagnostics_at) => {
                    continue;
                }
            };
            *pending = begin_admit(client, active.len(), submission, epoch).await?;
        }
        if was_idle {
            wait_admit(client, scheduler, active, pending).await?;
            // Fix the cohort before waiting: later arrivals must not postpone execution.
            let queued = incoming.len().min(max_running.saturating_sub(active.len()));
            for _ in 0..queued {
                let Ok(submission) = incoming.try_recv() else {
                    break;
                };
                *pending = begin_admit(client, active.len(), submission, epoch).await?;
                wait_admit(client, scheduler, active, pending).await?;
            }
        } else if let Some(preparing) = pending.as_ref() {
            if preparing.submission.ready.is_closed() || Instant::now() >= preparing.deadline {
                cancel_admit(client, pending.take().unwrap()).await?;
            } else if let Some(result) = client.poll_prepare_request()? {
                let preparing = pending.take().unwrap();
                complete_admit(client, scheduler, active, preparing, result).await?;
            }
        }
        let mut backpressure_failures = Vec::new();
        for (&id, request) in active.iter_mut() {
            if !request.output.request.stream {
                continue;
            }
            if let Err(error) = request.stream_buffer.flush(&request.events) {
                backpressure_failures.push((id, error));
            } else if request.stream_buffer.is_blocked() {
                scheduler.pause(id);
            } else {
                scheduler.resume(id);
            }
        }
        for (id, error) in backpressure_failures {
            scheduler.abort(id);
            if let Some(request) = active.remove(&id) {
                request.fail_delivery(&error);
                info!(request_id = id, %error, "stream aborted");
            }
        }
        let cancelled: Vec<_> = active
            .iter()
            .filter(|(_, request)| {
                request.events.is_closed() || Instant::now() >= request.lease.deadline()
            })
            .map(|(&id, _)| id)
            .collect();
        for id in cancelled {
            scheduler.abort(id);
            if let Some(request) = active.remove(&id) {
                request.fail_with_status("request cancelled or timed out", "cancelled");
                info!(request_id = id, "request cancelled");
            }
        }
        if !was_idle
            && pending.is_none()
            && let Ok(submission) = incoming.try_recv()
        {
            *pending = begin_admit(client, active.len(), submission, epoch).await?;
        }
        let batch = scheduler.schedule();
        if batch.cache_hit_tokens > 0 {
            let mut status = health.lock().unwrap();
            status["prefix_hit_tokens"] = json!(
                status["prefix_hit_tokens"].as_u64().unwrap_or(0) + batch.cache_hit_tokens as u64
            );
        }
        if batch.scheduled.is_empty()
            && batch.finished_request_ids.is_empty()
            && batch.preempted_request_ids.is_empty()
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
            continue;
        }
        let worker = execute(client, &batch, context.step_timeout).await?;
        for scheduled in &batch.scheduled {
            let id = scheduled.request_id;
            if let Some(error) = client.serving_errors.remove(&id) {
                scheduler.abort(id);
                client.serving_outputs.remove(&id);
                if let Some(request) = active.remove(&id) {
                    request.fail(&error);
                }
                warn!(request_id = id, %error, "request failed in worker");
            }
        }
        let mut worker = worker;
        while let Err(invalid) = scheduler.validate_output(&worker) {
            let id = invalid.request_id;
            let message = invalid.to_string();
            worker.outputs.retain(|output| output.request_id != id);
            scheduler.abort(id);
            client.serving_outputs.remove(&id);
            if let Some(request) = active.remove(&id) {
                request.fail(&message);
            }
            warn!(request_id = id, %message, "invalid worker output");
        }
        let worker = scheduler
            .update(worker)
            .context("validated worker output changed")?;
        if batch.scheduled.is_empty() && !active.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for result in worker.outputs {
            let id = result.request_id;
            let Some(request) = active.get_mut(&id) else {
                continue;
            };
            if let Some(scheduled) = batch.scheduled.iter().find(|s| s.request_id == id) {
                if request.steps == 0 {
                    request.output.cached_tokens = scheduled
                        .num_computed_tokens
                        .min(request.output.input_tokens);
                }
                request.proposed += (scheduled.num_computed_tokens + scheduled.token_ids.len())
                    .saturating_sub(request.output.input_tokens + request.output.output_tokens);
            }
            request.steps += 1;
            request.accepted += result.num_accepted_draft_tokens;
            request.output.output_tokens += result.token_ids.len();
            if !result.token_ids.is_empty() && request.first_token.is_none() {
                request.first_token = Some(Instant::now());
            }
            let (text, mut reason, reasoning_tokens) =
                client.serving_outputs.remove(&id).unwrap_or_default();
            if let Err(error) = request.lease.grow(
                text.len()
                    .saturating_mul(8)
                    .saturating_add(result.token_ids.len().saturating_mul(16)),
            ) {
                scheduler.abort(id);
                active.remove(&id).unwrap().fail(&error.to_string());
                continue;
            }
            request.output.reasoning_tokens = reasoning_tokens;
            if request.output.output_tokens
                >= request.output.request.normalized["max_tokens"]
                    .as_u64()
                    .unwrap() as usize
            {
                reason.get_or_insert("length".to_owned());
            }
            let parsed = if reason.as_deref() == Some("length") {
                request.parser.feed_length(&text)
            } else {
                request.parser.feed(&text, reason.is_some())
            };
            let events = parsed.map(|deltas| {
                deltas
                    .into_iter()
                    .flat_map(|d| request.output.delta(d))
                    .collect()
            });
            if let Err(error) = events.and_then(|events| send_events(request, events)) {
                scheduler.abort(id);
                active.remove(&id).unwrap().fail_delivery(&error);
                info!(request_id = id, error = %error, "request aborted");
                continue;
            }
            if request.stream_buffer.is_blocked() {
                scheduler.pause(id);
            }
            if let Some(reason) = reason {
                scheduler.abort(id);
                let mut request = active.remove(&id).unwrap();
                let (response, events) = request.output.finish(&reason);
                if request.output.request.store
                    && !store.lock().unwrap().insert(
                        request.output.id.clone(),
                        response.clone(),
                        request.output.history(),
                    )
                {
                    request.fail("response exceeds configured storage capacity");
                    continue;
                }
                if let Err(error) = send_events(&mut request, events) {
                    request.fail_delivery(&error);
                    continue;
                }
                if !request.output.request.responses
                    && request.output.request.stream
                    && let Err(error) = send_events(&mut request, vec![json!("[DONE]")])
                {
                    request.fail_delivery(&error);
                    continue;
                }
                let pending = std::mem::take(&mut request.stream_buffer);
                let _ = pending
                    .drain_after_completion(request.events.clone(), Some(request.lease.clone()));
                let elapsed = request.started.elapsed().as_secs_f64();
                let ttft_ms = request
                    .first_token
                    .map(|t| t.duration_since(request.started).as_millis());
                info!(request_id = id,
                    output_tokens = request.output.output_tokens,
                    input_tokens = request.output.input_tokens,
                    cached_tokens = request.output.cached_tokens,
                    reasoning_tokens = request.output.reasoning_tokens,
                    thinking_token_budget = request.output.request.normalized
                        ["thinking_token_budget"].as_i64(),
                    elapsed_s = elapsed,
                    ttft_ms,
                    output_tps = request.output.output_tokens as f64 / elapsed,
                    steps = request.steps,
                    proposed_draft_tokens = request.proposed,
                    verified_draft_tokens = request.proposed,
                    accepted_draft_tokens = request.accepted,
                    finish_reason = %reason, "request completed");
                request.metric("completed");
                let _ = request.done.send(Ok(response));
            }
        }
    }
    Ok(())
}

fn update_resource_counts(
    health: &Mutex<Value>,
    scheduler: &Scheduler,
    active: usize,
    pending: usize,
) {
    let mut status = health.lock().unwrap();
    status["active_requests"] = json!(active);
    status["pending_admissions"] = json!(pending);
    status["scheduler_running"] = json!(scheduler.num_running());
    status["scheduler_waiting"] = json!(scheduler.num_waiting());
    let (fa, fa_free, fa_cached, gdn, gdn_free, gdn_cached) = scheduler.cache_resource_counts();
    status["kv"] = json!({
        "fa_blocks": fa, "fa_free": fa_free, "fa_cached": fa_cached,
        "gdn_blocks": gdn, "gdn_free": gdn_free, "gdn_cached": gdn_cached,
    });
}
