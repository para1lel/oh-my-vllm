//! Rust-owned OpenAI HTTP adapters, bounded response state, and online scheduling.
mod events;
mod parser;
mod request;

use crate::{client::WorkerClient, error::Error as WorkerError};
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection},
    http::StatusCode,
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
    #[arg(long, default_value_t = 600)]
    pub request_timeout_seconds: u64,
    #[arg(long, default_value_t = 35)]
    pub shutdown_grace_seconds: u64,
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
}
struct Submission {
    request: Request,
    enqueued: Instant,
    events: mpsc::Sender<Value>,
    done: oneshot::Sender<std::result::Result<Value, String>>,
    ready: oneshot::Sender<std::result::Result<(), (StatusCode, String)>>,
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
                    anyhow::bail!("client disconnected");
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
                        anyhow::bail!("client disconnected");
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
async fn chat(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(app, body, false).await
}
async fn responses(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(app, body, true).await
}
async fn generate(
    app: App,
    body: std::result::Result<Json<Value>, JsonRejection>,
    responses: bool,
) -> Response {
    let body = match body {
        Ok(Json(v)) => v,
        Err(e) => return error(e.status(), e.body_text()),
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
            Some(s) => Some(s.history.clone()),
            None => return error(StatusCode::NOT_FOUND, "response not found or expired"),
        }
    } else {
        None
    };
    let request = match request::normalize(body, responses, &app.model, previous) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let streaming = request.stream;
    let (tx, rx) = mpsc::channel(256);
    let (done, result) = oneshot::channel();
    let (ready, prepared) = oneshot::channel();
    let submission = Submission {
        request,
        enqueued: Instant::now(),
        events: tx,
        done,
        ready,
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
        // Keep the event receiver alive until completion; its closure is cancellation.
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
async fn retrieve(State(app): State<App>, Path(id): Path<String>) -> Response {
    let mut store = app.store.lock().unwrap();
    store.prune();
    match store.entries.get(&id) {
        Some(s) => Json(s.response.clone()).into_response(),
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
    anyhow::ensure!(
        args.response_capacity > 0
            && args.response_max_bytes > 0
            && args.response_ttl_seconds > 0
            && args.request_timeout_seconds > 0
            && args.shutdown_grace_seconds > 0,
        "serving limits must be positive"
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
    let app = App {
        submit: tx,
        store: store.clone(),
        model: args.served_model_name.clone(),
    };
    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{id}", get(retrieve).delete(delete))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "unknown endpoint") })
        .layer(DefaultBodyLimit::max(8 << 20))
        .with_state(app);
    info!(listen = %args.listen, model = %args.served_model_name, "HTTP service ready");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let mut engine = tokio::spawn(engine(
        client,
        scheduler,
        rx,
        store,
        Duration::from_secs(args.request_timeout_seconds),
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
}

async fn begin_admit(
    client: &mut WorkerClient,
    active_count: usize,
    submission: Submission,
) -> Result<Option<PendingAdmission>> {
    if submission.ready.is_closed() {
        return Ok(None);
    }
    if active_count >= 64 {
        let _ = submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "active request limit exceeded".to_owned(),
        )));
        return Ok(None);
    }
    let id = IDS.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let prepare_timeout_ms = std::env::var("OH_MY_VLLM_PREPARE_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(120_000);
    let deadline = started + Duration::from_millis(prepare_timeout_ms);
    // A stalled DEALER send must not hold active inference past the same
    // preparation deadline. Abort also covers a send that reached Python just
    // before the local future timed out.
    match tokio::time::timeout(
        Duration::from_millis(prepare_timeout_ms),
        client.begin_prepare_request(id, submission.request.normalized.clone()),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(error.into()),
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
    }))
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
            },
        id,
        started,
        ..
    } = pending;
    let tokens = match prepare_result {
        Err(e) => {
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
    };
    if !scheduler.add_request(EngineRequest::new(id, tokens, max_tokens, vec![])) {
        client.cancel_prepare(id).await?;
        let _ = ready.send(Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "request FA/GDN or output budget exceeds KV pool capacity".to_owned(),
        )));
        return Ok(());
    }
    if ready.send(Ok(())).is_err() || send_events(&mut running, start_events).is_err() {
        scheduler.abort(id);
        client.cancel_prepare(id).await?;
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
    fn fail(mut self, message: &str) {
        let event = self.output.error_event(message);
        let _ = self.events.try_send(event);
        let _ = self.done.send(Err(message.to_owned()));
    }
}

async fn engine(
    mut client: WorkerClient,
    mut scheduler: Scheduler,
    mut incoming: mpsc::Receiver<Submission>,
    store: Arc<Mutex<Store>>,
    request_timeout: Duration,
    mut shutdown: oneshot::Receiver<()>,
) {
    let mut active = BTreeMap::new();
    let mut pending = None;
    let (result, stopping) = tokio::select! {
        _ = &mut shutdown => (Ok(()), true),
        result = run_engine(
            &mut client,
            &mut scheduler,
            &mut incoming,
            &store,
            request_timeout,
            &mut active,
            &mut pending,
        ) => (result, false),
    };
    if let Err(error) = result {
        warn!(%error, "inference engine failed");
    }
    for (_, request) in active {
        request.fail(if stopping {
            "server shutting down"
        } else {
            "worker failed"
        });
    }
    if let Some(pending) = pending {
        if let Err(error) = client.cancel_prepare(pending.id).await {
            warn!(request_id = pending.id, %error, "shutdown prepare cancellation failed");
        }
        let _ = pending.submission.ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".to_owned(),
        )));
    }
    // Fail pending admission immediately rather than holding handlers during shutdown.
    drop(incoming);
    let _ = client.shutdown().await;
}

async fn execute(
    client: &mut WorkerClient,
    batch: &oh_my_vllm_scheduler::SchedulerOutput,
) -> Result<oh_my_vllm_scheduler::WorkerOutput> {
    if batch.scheduled.len() > 1 {
        let request_ids: Vec<_> = batch.scheduled.iter().map(|r| r.request_id).collect();
        debug!(?request_ids, "scheduled mixed batch");
    }
    let output = tokio::time::timeout(
        Duration::from_secs(120),
        client.execute_one_step(
            &batch.scheduled,
            &batch.finished_request_ids,
            &batch.preempted_request_ids,
            batch.num_batched_tokens,
        ),
    )
    .await??;
    if !batch.finished_request_ids.is_empty() {
        debug!(request_ids = ?batch.finished_request_ids, "worker requests released");
    }
    Ok(output)
}

async fn run_engine(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
    incoming: &mut mpsc::Receiver<Submission>,
    store: &Mutex<Store>,
    request_timeout: Duration,
    active: &mut BTreeMap<u64, Active>,
    pending: &mut Option<PendingAdmission>,
) -> Result<()> {
    loop {
        let was_idle = active.is_empty();
        if was_idle {
            // Finish native cleanup before going idle, including cancelled admissions.
            execute(client, &scheduler.schedule()).await?;
            if pending.is_none() {
                let Some(submission) = incoming.recv().await else {
                    break;
                };
                *pending = begin_admit(client, active.len(), submission).await?;
            }
        }
        if let Some(preparing) = pending.as_ref() {
            if preparing.submission.ready.is_closed() || Instant::now() >= preparing.deadline {
                let preparing = pending.take().unwrap();
                let disconnected = preparing.submission.ready.is_closed();
                client.cancel_prepare(preparing.id).await?;
                if !disconnected {
                    let _ = preparing.submission.ready.send(Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        "worker preparation timeout".to_owned(),
                    )));
                }
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
                backpressure_failures.push((id, error.to_string()));
            } else if request.stream_buffer.is_blocked() {
                scheduler.pause(id);
            } else {
                scheduler.resume(id);
            }
        }
        for (id, error) in backpressure_failures {
            scheduler.abort(id);
            if let Some(request) = active.remove(&id) {
                request.fail(&error);
                info!(request_id = id, %error, "stream aborted");
            }
        }
        let cancelled: Vec<_> = active
            .iter()
            .filter(|(_, request)| {
                request.events.is_closed() || request.started.elapsed() > request_timeout
            })
            .map(|(&id, _)| id)
            .collect();
        for id in cancelled {
            scheduler.abort(id);
            if let Some(request) = active.remove(&id) {
                request.fail("request cancelled or timed out");
                info!(request_id = id, "request cancelled");
            }
        }
        if !was_idle
            && pending.is_none()
            && let Ok(submission) = incoming.try_recv()
        {
            *pending = begin_admit(client, active.len(), submission).await?;
        }
        let batch = scheduler.schedule();
        if batch.scheduled.is_empty()
            && batch.finished_request_ids.is_empty()
            && batch.preempted_request_ids.is_empty()
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
            continue;
        }
        let worker = execute(client, &batch).await?;
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
            // A constrained logical pool can temporarily prevent admission. Avoid a
            // CPU busy loop while retaining request timeout/cancellation handling.
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
                // Count drafts actually scheduled, excluding the unverified next batch.
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
            request.output.reasoning_tokens = reasoning_tokens;
            if request.output.output_tokens
                >= request.output.request.normalized["max_tokens"]
                    .as_u64()
                    .unwrap() as usize
            {
                reason.get_or_insert("length".to_owned());
            }
            // Length truncation may leave an incomplete call. Never deliver that call
            // to a tool executor; retain length/incomplete rather than reporting 500.
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
                active.remove(&id).unwrap().fail(&error.to_string());
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
                    request.fail(&error.to_string());
                    continue;
                }
                if !request.output.request.responses
                    && request.output.request.stream
                    && let Err(error) = send_events(&mut request, vec![json!("[DONE]")])
                {
                    request.fail(&error.to_string());
                    continue;
                }
                let pending = std::mem::take(&mut request.stream_buffer);
                let _ = pending.drain_after_completion(request.events.clone());
                let elapsed = request.started.elapsed().as_secs_f64();
                let ttft_ms = request
                    .first_token
                    .map(|t| t.duration_since(request.started).as_millis());
                info!(request_id = id,
                    output_tokens = request.output.output_tokens,
                    input_tokens = request.output.input_tokens,
                    cached_tokens = request.output.cached_tokens,
                    reasoning_tokens = request.output.reasoning_tokens,
                    elapsed_s = elapsed,
                    ttft_ms,
                    output_tps = request.output.output_tokens as f64 / elapsed,
                    steps = request.steps,
                    proposed_draft_tokens = request.proposed,
                    accepted_draft_tokens = request.accepted,
                    finish_reason = %reason, "request completed");
                let _ = request.done.send(Ok(response));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stalled_stream_preserves_more_than_256_events_and_recovers_in_order() {
        let (sender, mut receiver) = mpsc::channel(256);
        let mut buffer = StreamBuffer::default();
        buffer
            .send(&sender, (0..300).map(|index| json!(index)).collect())
            .unwrap();
        assert!(buffer.is_blocked());
        assert_eq!(buffer.pending.len(), 44);
        for expected in 0..200 {
            assert_eq!(receiver.recv().await.unwrap(), json!(expected));
        }
        buffer.flush(&sender).unwrap();
        assert!(!buffer.is_blocked());
        for expected in 200..300 {
            assert_eq!(receiver.recv().await.unwrap(), json!(expected));
        }
    }

    #[tokio::test]
    async fn intermittent_one_event_reads_do_not_reset_stream_grace() {
        let (sender, mut receiver) = mpsc::channel(256);
        let mut buffer = StreamBuffer::default();
        buffer
            .send(&sender, (0..257).map(|index| json!(index)).collect())
            .unwrap();
        assert!(buffer.is_blocked());
        assert_eq!(receiver.recv().await.unwrap(), json!(0));
        buffer.flush(&sender).unwrap();
        assert!(buffer.pending.is_empty());
        assert!(
            buffer.is_blocked(),
            "one free slot must not resume generation"
        );
        buffer.blocked_since = Some(Instant::now() - Duration::from_secs(29));
        buffer.flush(&sender).unwrap();
        buffer.blocked_since = Some(Instant::now() - Duration::from_secs(31));
        assert!(buffer.flush(&sender).is_err());
    }

    #[tokio::test]
    async fn completed_stream_drains_pending_events_and_done_in_order() {
        let (sender, mut receiver) = mpsc::channel(256);
        let mut buffer = StreamBuffer::default();
        buffer
            .send(&sender, (0..258).map(|index| json!(index)).collect())
            .unwrap();
        buffer.send(&sender, vec![json!("[DONE]")]).unwrap();
        let drain = buffer.drain_after_completion(sender.clone()).unwrap();
        for expected in 0..258 {
            let item = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(item, json!(expected));
        }
        assert_eq!(receiver.recv().await.unwrap(), json!("[DONE]"));
        drain.await.unwrap();
    }

    #[tokio::test]
    async fn completed_stream_drain_exits_on_timeout_or_receiver_drop() {
        let (sender, receiver) = mpsc::channel(1);
        sender.try_send(json!("occupied")).unwrap();
        let mut buffer = StreamBuffer::default();
        buffer.queue(json!("pending")).unwrap();
        buffer.blocked_since = Some(Instant::now() - Duration::from_secs(31));
        let drain = buffer.drain_after_completion(sender.clone()).unwrap();
        tokio::time::timeout(Duration::from_millis(100), drain)
            .await
            .unwrap()
            .unwrap();
        drop(receiver);

        let (sender, receiver) = mpsc::channel(1);
        let mut buffer = StreamBuffer::default();
        buffer.queue(json!("pending")).unwrap();
        drop(receiver);
        let drain = buffer.drain_after_completion(sender).unwrap();
        tokio::time::timeout(Duration::from_millis(100), drain)
            .await
            .unwrap()
            .unwrap();
    }

    #[test]
    fn stream_pending_cap_and_disconnect_fail_closed() {
        let (sender, receiver) = mpsc::channel(256);
        let mut buffer = StreamBuffer::default();
        assert!(
            buffer
                .send(&sender, (0..1281).map(|index| json!(index)).collect())
                .is_err()
        );
        assert_eq!(buffer.pending.len(), MAX_PENDING_STREAM_EVENTS);
        drop(receiver);
        assert!(buffer.flush(&sender).is_err());
    }

    #[test]
    fn store_capacity_and_expiry() {
        let mut store = Store {
            entries: BTreeMap::new(),
            ttl: Duration::from_secs(60),
            capacity: 1,
            max_bytes: 1000,
            bytes: 0,
        };
        store.insert("a".to_owned(), json!({}), vec![]);
        store.insert("b".to_owned(), json!({}), vec![]);
        assert!(!store.entries.contains_key("a"));
        store.entries.get_mut("b").unwrap().saved = Instant::now() - Duration::from_secs(61);
        store.prune();
        assert_eq!(store.bytes, 0);
    }
}
