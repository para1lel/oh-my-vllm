mod admission;
mod engine;
mod events;
mod http;
mod media;
mod recovery;
mod request;
mod resources;
mod sampling;
mod server;
mod stream;
#[cfg(all(test, unix))]
mod tests;

use stream::StreamBuffer;

pub use server::serve;

use crate::models::output::Parser;
use anyhow::Result;
use axum::http::StatusCode;
use clap::Args;
use events::Output;
use request::Request;
use resources::{Budget, Lease, unix_seconds};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{Arc, Mutex, atomic::AtomicU64},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

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
    #[arg(long)]
    pub served_model_name: Option<String>,
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
