//! Rust-owned OpenAI HTTP adapters, bounded response state, and online scheduling.
mod events;
mod parser;
mod request;

use crate::client::WorkerClient;
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
    collections::BTreeMap,
    convert::Infallible,
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
    done: oneshot::Sender<std::result::Result<Value, String>>,
    started: Instant,
    first_token: Option<Instant>,
    steps: usize,
    proposed: usize,
    accepted: usize,
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    let kind = if status == StatusCode::INTERNAL_SERVER_ERROR {
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
            && args.request_timeout_seconds > 0,
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
    let engine = tokio::spawn(engine(
        client,
        scheduler,
        rx,
        store,
        Duration::from_secs(args.request_timeout_seconds),
    ));
    let abort_engine = engine.abort_handle();
    let result = axum::serve(listener, router)
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
            abort_engine.abort();
        })
        .await;
    engine.abort();
    let _ = engine.await;
    result.context("HTTP service")
}

fn send_events(active: &Active, events: Vec<Value>) -> Result<()> {
    if active.output.request.stream {
        for event in events {
            active
                .events
                .try_send(event)
                .map_err(|_| anyhow::anyhow!("client disconnected or stream buffer full"))?;
        }
    }
    Ok(())
}

async fn admit(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
    active: &mut BTreeMap<u64, Active>,
    submission: Submission,
) -> Result<()> {
    let Submission {
        request,
        enqueued,
        events,
        done,
        ready,
    } = submission;
    if ready.is_closed() {
        return Ok(());
    }
    if active.len() >= 64 {
        let _ = ready.send(Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "active request limit exceeded".to_owned(),
        )));
        return Ok(());
    }
    let id = IDS.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let prepare_result = tokio::time::timeout(
        Duration::from_secs(120),
        client.prepare_request(id, request.normalized.clone()),
    )
    .await;
    let tokens = match prepare_result {
        Err(_) => {
            let _ = ready.send(Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "worker preparation timeout".to_owned(),
            )));
            return Ok(());
        }
        Ok(Err(e)) => {
            let _ = ready.send(Err((StatusCode::BAD_REQUEST, e.to_string())));
            return Ok(());
        }
        Ok(Ok(tokens)) => tokens,
    };
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
    let running = Active {
        output,
        parser,
        events,
        done,
        started: enqueued,
        first_token: None,
        steps: 0,
        proposed: 0,
        accepted: 0,
    };
    if !scheduler.add_request(EngineRequest::new(id, tokens, max_tokens, vec![])) {
        let _ = ready.send(Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "prompt exceeds KV pool capacity".to_owned(),
        )));
        return Ok(());
    }
    if ready.send(Ok(())).is_err() || send_events(&running, start_events).is_err() {
        scheduler.abort(id);
        return Ok(());
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
) {
    let mut active = BTreeMap::new();
    let result = run_engine(
        &mut client,
        &mut scheduler,
        &mut incoming,
        &store,
        request_timeout,
        &mut active,
    )
    .await;
    if let Err(error) = result {
        warn!(%error, "inference engine failed");
        for (_, request) in active {
            request.fail("worker failed");
        }
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
) -> Result<()> {
    loop {
        if active.is_empty() {
            // Finish native cleanup before going idle, including cancelled admissions.
            execute(client, &scheduler.schedule()).await?;
            let Some(submission) = incoming.recv().await else {
                break;
            };
            admit(client, scheduler, active, submission).await?;
        } else if let Ok(submission) = incoming.try_recv() {
            // Bound admission work to one request between model steps.
            admit(client, scheduler, active, submission).await?;
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
        let batch = scheduler.schedule();
        let worker = execute(client, &batch).await?;
        let worker = scheduler.update(worker);
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
            let parsed = request.parser.feed(&text, reason.is_some());
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
                if let Err(error) = send_events(&request, events) {
                    request.fail(&error.to_string());
                    continue;
                }
                if !request.output.request.responses && request.output.request.stream {
                    let _ = request.events.try_send(json!("[DONE]"));
                }
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
