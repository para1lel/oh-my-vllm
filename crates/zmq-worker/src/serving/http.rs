use crate::serving::resources::{Lease, serialized_bytes};
use crate::serving::{App, Submission, request, resources, sampling};
use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use axum::{Extension, Json};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

pub(super) fn error(status: StatusCode, message: impl ToString) -> Response {
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

pub(super) fn error_response(reason: impl ToString) -> Response {
    error(StatusCode::SERVICE_UNAVAILABLE, reason)
}

pub(super) async fn reserve_http(
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

pub(super) async fn diagnostics(State(app): State<App>) -> Json<Value> {
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

pub(super) async fn chat(
    State(app): State<App>,
    Extension(lease): Extension<Lease>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    generate(app, lease, body, false).await
}

pub(super) async fn responses(
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
    let previous = match previous_history(&app, &lease, &body, body_bytes, responses) {
        Ok(previous) => previous,
        Err((status, message)) => return error(status, message),
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
    deliver(result, rx, streaming, responses).await
}

pub(super) async fn models(State(app): State<App>) -> Json<Value> {
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

pub(super) async fn retrieve(
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

pub(super) async fn delete(State(app): State<App>, Path(id): Path<String>) -> Response {
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

fn previous_history(
    app: &App,
    lease: &Lease,
    body: &Value,
    body_bytes: usize,
    responses: bool,
) -> std::result::Result<Option<Vec<Value>>, (StatusCode, String)> {
    if let Some(id) = body.get("previous_response_id").filter(|v| !v.is_null()) {
        if !responses {
            return Err((
                StatusCode::BAD_REQUEST,
                "previous_response_id requires Responses".to_owned(),
            ));
        }
        let Some(id) = id.as_str() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "previous_response_id must be a string".to_owned(),
            ));
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
                    return Err((StatusCode::SERVICE_UNAVAILABLE, error.to_string()));
                }
                Ok(Some(s.history.clone()))
            }
            None => Err((
                StatusCode::NOT_FOUND,
                "response not found or expired".to_owned(),
            )),
        }
    } else {
        if let Err(error) = lease.resize(body_bytes.saturating_mul(6)) {
            return Err((StatusCode::SERVICE_UNAVAILABLE, error.to_string()));
        }
        Ok(None)
    }
}

async fn deliver(
    result: oneshot::Receiver<std::result::Result<Value, String>>,
    rx: mpsc::Receiver<Value>,
    streaming: bool,
    responses: bool,
) -> Response {
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
