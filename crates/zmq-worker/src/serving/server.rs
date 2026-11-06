use crate::client::WorkerClient;
use crate::models::ServiceCache;
use crate::serving::http::{
    chat, delete, diagnostics, error, models, reserve_http, responses, retrieve,
};
use crate::serving::recovery::engine;
use crate::serving::resources::Budget;
use crate::serving::{App, EngineContext, ServeArgs, Store, Submission, resources, sampling};
use anyhow::{Context, Result};
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Router, middleware};
use oh_my_vllm_scheduler::Scheduler;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::future::IntoFuture;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

pub async fn serve<P: ServiceCache>(
    client: WorkerClient,
    scheduler: Scheduler<P>,
    args: ServeArgs,
) -> Result<()> {
    let model_name = args
        .served_model_name
        .clone()
        .unwrap_or_else(|| client.model_policy().served_model_name().to_owned());
    let sampling_defaults = sampling::service_defaults(&args.override_generation_config)?;
    validate_limits(&args)?;
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    let (app, rx) = create_app(model_name.clone(), sampling_defaults, &args);
    let store = app.store.clone();
    let health = app.health.clone();
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
    info!(listen = %args.listen, model = %model_name, "HTTP service ready");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let engine = tokio::spawn(engine(
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
    drain_server(
        listener,
        router,
        shutdown_tx,
        engine,
        Duration::from_secs(args.shutdown_grace_seconds),
    )
    .await
}

fn validate_limits(args: &ServeArgs) -> Result<()> {
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
    Ok(())
}

fn create_app(
    model_name: String,
    sampling_defaults: Value,
    args: &ServeArgs,
) -> (App, mpsc::Receiver<Submission>) {
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
        scope: crate::serving::identity::random_id().expect("service identity entropy"),
        model: model_name,
        thinking_token_budget: args.thinking_token_budget,
        sampling_defaults,
        budget: budget.clone(),
        health: health.clone(),
        request_timeout: Duration::from_secs(args.request_timeout_seconds),
    };
    (app, rx)
}

async fn drain_server(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown_tx: oneshot::Sender<()>,
    mut engine: tokio::task::JoinHandle<()>,
    grace: Duration,
) -> Result<()> {
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
