//! Async client handle for the Python model worker.
//!
//! `WorkerClient` owns the ZMQ DEALER socket and the Python child process.
//! The scheduler calls `execute_one_step` once per inference step.

use std::collections::BTreeSet;
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use oh_my_vllm_scheduler::output::{RequestOutput, ScheduledRequest, WorkerOutput};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{debug, info, warn};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSendHalf, DealerSocket, ZmqMessage};

use crate::error::{Error, Result};
use crate::protocol::{
    AbortMsg, ExecuteMsg, InitMsg, RegisterMsg, RustMessage, ScheduledRequestMsg, decode, encode,
};

/// Configuration for launching the Python worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Path to the model weights directory.
    pub model_path: PathBuf,
    /// ZMQ IPC socket path (without `ipc://` prefix).
    pub socket_path: PathBuf,
    /// Number of KV cache blocks pre-allocated on the GPU.
    pub num_gpu_blocks: u32,
    pub mamba_blocks: Option<u32>,
    /// Block size in tokens (Qwen3.8: 784).
    pub block_size: u32,
    /// Tensor parallel size (1 for single-GPU B200).
    pub tensor_parallel_size: u32,
    /// Maximum sequence length in tokens.
    pub max_model_len: u32,
    pub num_speculative_tokens: usize,
    /// Process mode: none, mtp, or dspark. None retains the legacy token-count selection.
    pub speculative_mode: Option<String>,
    /// Path to DSpark weights. Required for DSpark or comparison execution.
    pub draft_model_path: Option<PathBuf>,
    /// Keep both proposers for an idle-only benchmark switch on shared target caches.
    pub comparison: bool,
    /// DSpark cumulative confidence cutoff. Zero selects fixed-length proposals.
    pub dspark_confidence_threshold: f64,
    /// Python executable to use.
    pub python_executable: PathBuf,
    /// Timeout waiting for the worker to become ready after launch.
    pub init_timeout: Duration,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model_path: std::env::var_os("OH_MY_VLLM_MODEL")
                .map(PathBuf::from)
                .unwrap_or_default(),
            socket_path: PathBuf::from("/tmp/oh-my-vllm.ipc"),
            num_gpu_blocks: 1024,
            mamba_blocks: None,
            block_size: 784,
            tensor_parallel_size: 1,
            max_model_len: 65536,
            num_speculative_tokens: 0,
            speculative_mode: None,
            draft_model_path: std::env::var_os("OH_MY_VLLM_DRAFT_MODEL").map(PathBuf::from),
            comparison: false,
            dspark_confidence_threshold: 0.2,
            python_executable: std::env::var_os("OH_MY_VLLM_WORKER_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("python3")),
            init_timeout: Duration::from_secs(300),
        }
    }
}

/// Handle to the running Python model worker.
pub struct WorkerClient {
    sock: DealerSendHalf,
    inbox: mpsc::Receiver<Result<crate::protocol::PythonMessage>>,
    reader: JoinHandle<()>,
    _child: ManagedChild,
    step_id: u64,
    rpc_id: u64,
    pending_prepare: Option<(u64, u64)>,
    last_prepare: Option<(u64, u64)>,
    cancelled_prepares: BTreeSet<u64>,
    prepared_reply: Option<crate::protocol::PythonMessage>,
    serving_enabled: bool,
    comparison: bool,
    pub logical_num_blocks: u32,
    pub mamba_blocks: u32,
    pub serving_outputs: std::collections::BTreeMap<u64, (String, Option<String>, usize)>,
    pub serving_errors: std::collections::BTreeMap<u64, String>,
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// Reap the owned worker on errors as well as normal shutdown.
struct ManagedChild(Option<Child>);

impl ManagedChild {
    fn child(&mut self) -> &mut Child {
        self.0
            .as_mut()
            .expect("worker child remains owned until drop")
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            if child.try_wait().ok().flatten().is_some() {
                return;
            }
            let _ = child.kill();
            // Waiting for a child can block; reap it off the Tokio executor.
            let child = Arc::new(Mutex::new(child));
            let waiter = Arc::clone(&child);
            if std::thread::Builder::new()
                .name("oh-my-vllm-worker-reap".to_owned())
                .spawn(move || {
                    let _ = waiter.lock().unwrap().wait();
                })
                .is_err()
            {
                // If no waiter thread can start, still satisfy the reaping contract.
                let _ = child.lock().unwrap().wait();
            }
        }
    }
}

impl WorkerClient {
    /// Launch the Python worker and wait for the `ready` handshake.
    pub async fn launch(config: WorkerConfig) -> Result<Self> {
        if config.model_path.as_os_str().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "set OH_MY_VLLM_MODEL or WorkerConfig.model_path",
            )
            .into());
        }
        if (config.speculative_mode.as_deref() == Some("dspark") || config.comparison)
            && config
                .draft_model_path
                .as_ref()
                .is_none_or(|p| p.as_os_str().is_empty())
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "set OH_MY_VLLM_DRAFT_MODEL or WorkerConfig.draft_model_path",
            )
            .into());
        }
        let addr = format!("ipc://{}", config.socket_path.display());

        // Bind on the Rust side; the Python side connects.
        let mut sock = DealerSocket::new();
        sock.bind(&addr).await?;
        info!("ZMQ DEALER socket bound at {addr}");

        // Fork the Python worker process.
        let mut child = ManagedChild(Some({
            let mut cmd = Command::new(&config.python_executable);
            cmd.args(["-m", "oh_my_vllm.worker.zmq_bridge", "--socket", &addr])
                .env(
                    "OH_MY_VLLM_RUN_ID",
                    std::env::var("OH_MY_VLLM_RUN_ID")
                        .unwrap_or_else(|_| format!("pid-{}", std::process::id())),
                );
            // SAFETY: prctl is async-signal-safe; no Rust objects are touched.
            // The child receives SIGKILL when the Rust parent exits, preventing
            // orphaned GPU worker processes.
            #[cfg(target_os = "linux")]
            unsafe {
                let parent_pid = libc::getpid();
                cmd.pre_exec(move || {
                    if libc::prctl(
                        libc::PR_SET_PDEATHSIG,
                        libc::SIGKILL as libc::c_ulong,
                        0,
                        0,
                        0,
                    ) == -1
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The parent may have died between fork and prctl.
                    if libc::getppid() != parent_pid {
                        return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                    }
                    Ok(())
                });
            }
            cmd.spawn()?
        }));
        info!("Python worker launched (pid {})", child.child().id());

        // Send the init message.  The Python process needs a moment to import
        // and connect its DEALER socket before we can deliver the first message,
        // so retry with exponential backoff until the send succeeds.
        let init = RustMessage::Init(InitMsg {
            model_path: config.model_path.to_string_lossy().into_owned(),
            num_gpu_blocks: config.num_gpu_blocks,
            mamba_blocks: config.mamba_blocks,
            block_size: config.block_size,
            tensor_parallel_size: config.tensor_parallel_size,
            max_model_len: config.max_model_len,
            num_speculative_tokens: config.num_speculative_tokens,
            speculative_mode: config.speculative_mode.clone(),
            draft_model_path: config
                .draft_model_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            comparison: config.comparison,
            dspark_confidence_threshold: config.dspark_confidence_threshold,
        });
        let mut delay_ms = 100u64;
        let deadline = std::time::Instant::now() + config.init_timeout;
        loop {
            if child.child().try_wait()?.is_some() {
                return Err(Error::WorkerDied);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let sent = timeout(remaining, Self::send_raw(&mut sock, &init))
                .await
                .map_err(|_| Error::Timeout)?;
            match sent {
                Ok(()) => break,
                Err(e) => {
                    if std::time::Instant::now() >= deadline {
                        warn!("init send failed at deadline: {e}");
                        return Err(Error::Timeout);
                    }
                    warn!("Init send failed ({e}), retrying in {delay_ms}ms…");
                    tokio::time::sleep(
                        Duration::from_millis(delay_ms)
                            .min(deadline.saturating_duration_since(Instant::now())),
                    )
                    .await;
                    delay_ms = (delay_ms * 2).min(2000);
                }
            }
        }

        // Wait for `ready`.
        let ready_reply = timeout(
            deadline.saturating_duration_since(Instant::now()),
            Self::recv_with_child(&mut sock, &mut child),
        )
        .await
        .map_err(|_| Error::Timeout)??;

        let (logical_num_blocks, mamba_blocks) = match ready_reply {
            crate::protocol::PythonMessage::Ready {
                logical_num_blocks,
                mamba_blocks,
            } => {
                info!(logical_num_blocks, mamba_blocks, "Python worker is ready");
                (logical_num_blocks, mamba_blocks)
            }
            crate::protocol::PythonMessage::Error(e) => {
                return Err(Error::WorkerError(e.message));
            }
            other => {
                return Err(Error::UnexpectedMessageType(format!("{other:?}")));
            }
        };

        let (send, mut receive) = sock.split();
        let (inbox_tx, inbox) = mpsc::channel(16);
        let reader = tokio::spawn(async move {
            loop {
                let reply = Self::recv_raw(&mut receive).await;
                let failed = reply.is_err();
                if inbox_tx.send(reply).await.is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            sock: send,
            inbox,
            reader,
            _child: child,
            step_id: 0,
            rpc_id: 0,
            pending_prepare: None,
            last_prepare: None,
            cancelled_prepares: BTreeSet::new(),
            prepared_reply: None,
            serving_enabled: false,
            comparison: config.comparison,
            logical_num_blocks,
            mamba_blocks,
            serving_outputs: Default::default(),
            serving_errors: Default::default(),
        })
    }

    /// Tell the Python worker about a new request before it is first scheduled.
    /// Send failures are non-fatal: the worker may have already exited, and the
    /// next `execute_one_step` will detect the dead worker via `recv_with_child`.
    pub async fn register_request(&mut self, request_id: u64, prompt_token_ids: Vec<u32>) {
        self.send_registration(request_id, prompt_token_ids, None)
            .await;
    }

    /// Register a request with the same output budget as the Rust scheduler.
    pub async fn register_request_with_limit(
        &mut self,
        request_id: u64,
        prompt_token_ids: Vec<u32>,
        max_tokens: usize,
    ) {
        self.send_registration(request_id, prompt_token_ids, Some(max_tokens))
            .await;
    }

    async fn send_registration(
        &mut self,
        request_id: u64,
        prompt_token_ids: Vec<u32>,
        max_tokens: Option<usize>,
    ) {
        debug!(
            request_id,
            input_tokens = prompt_token_ids.len(),
            "register_request"
        );
        let msg = RustMessage::Register(RegisterMsg {
            request_id,
            prompt_token_ids,
            max_tokens,
        });
        if let Err(e) = Self::send_raw(&mut self.sock, &msg).await {
            warn!(request_id, "register send failed: {e}");
        }
    }

    /// Select a preloaded proposer after all benchmark requests have finished.
    pub async fn set_speculative_mode(&mut self, mode: &str) -> Result<()> {
        if !self.comparison
            || !matches!(mode, "mtp" | "dspark")
            || self.serving_enabled
            || self.pending_prepare.is_some()
        {
            return Err(Error::WorkerError(
                "cannot switch a serving worker".to_owned(),
            ));
        }
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        Self::send_raw(
            &mut self.sock,
            &RustMessage::SetSpeculativeMode {
                rpc_id,
                mode: mode.to_owned(),
            },
        )
        .await?;
        match self.recv_for(rpc_id).await? {
            crate::protocol::PythonMessage::ModeChanged { mode: received, .. }
                if received == mode =>
            {
                Ok(())
            }
            crate::protocol::PythonMessage::Error(error) => Err(Error::WorkerError(error.message)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    /// Release state from a prepare whose HTTP caller timed out or disconnected.
    pub async fn cancel_prepare(&mut self, request_id: u64) -> Result<()> {
        let msg = RustMessage::Abort(AbortMsg { request_id });
        timeout(Duration::from_secs(1), Self::send_raw(&mut self.sock, &msg))
            .await
            .map_err(|_| Error::Timeout)??;
        if let Some((id, rpc_id)) = self.last_prepare
            && id == request_id
        {
            if self.prepared_reply.is_none() {
                self.cancelled_prepares.insert(rpc_id);
                // An evicted late reply fails closed instead of contaminating
                // a newer RPC; cancellation history cannot grow without bound.
                if self.cancelled_prepares.len() > 1024 {
                    self.cancelled_prepares.pop_first();
                }
            }
            self.last_prepare = None;
        }
        if self.pending_prepare.is_some_and(|(id, _)| id == request_id) {
            self.pending_prepare = None;
            self.prepared_reply = None;
        }
        Ok(())
    }

    /// Start one CPU preparation while inference steps continue on this connection.
    pub async fn begin_prepare_request(
        &mut self,
        request_id: u64,
        request: serde_json::Value,
    ) -> Result<()> {
        if self.pending_prepare.is_some() {
            return Err(Error::UnexpectedMessageType(
                "a preparation is already in flight".to_owned(),
            ));
        }
        self.serving_enabled = true;
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        // Record the ID before send: a send timeout may race with delivery.
        self.last_prepare = Some((request_id, rpc_id));
        Self::send_raw(
            &mut self.sock,
            &RustMessage::Prepare {
                rpc_id,
                request_id,
                request,
            },
        )
        .await?;
        self.pending_prepare = Some((request_id, rpc_id));
        Ok(())
    }

    /// Poll the outstanding preparation without holding up an active stream.
    pub fn poll_prepare_request(&mut self) -> Result<Option<Result<Vec<u32>>>> {
        let Some((_, rpc_id)) = self.pending_prepare else {
            return Ok(None);
        };
        let reply = if let Some(reply) = self.prepared_reply.take() {
            reply
        } else {
            match self.inbox.try_recv() {
                Ok(reply) => {
                    let reply = reply?;
                    let received = match &reply {
                        crate::protocol::PythonMessage::Prepared { rpc_id, .. } => *rpc_id,
                        crate::protocol::PythonMessage::ExecuteResult(result) => result.rpc_id,
                        crate::protocol::PythonMessage::Error(error) => error.rpc_id,
                        other => {
                            return Err(Error::UnexpectedMessageType(format!("{other:?}")));
                        }
                    };
                    if received < rpc_id {
                        if matches!(
                            reply,
                            crate::protocol::PythonMessage::Prepared { .. }
                                | crate::protocol::PythonMessage::Error(_)
                        ) && self.cancelled_prepares.remove(&received)
                        {
                            warn!(received, rpc_id, "discarding reply to cancelled RPC");
                            return Ok(None);
                        }
                        return Err(Error::UnexpectedMessageType(format!(
                            "unsolicited stale RPC reply {received} while waiting for {rpc_id}"
                        )));
                    }
                    if received != rpc_id {
                        return Err(Error::UnexpectedMessageType(format!(
                            "future RPC reply {received} while waiting for {rpc_id}"
                        )));
                    }
                    reply
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    if self._child.child().try_wait()?.is_some() {
                        return Err(Error::WorkerDied);
                    }
                    return Ok(None);
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return Err(Error::WorkerDied),
            }
        };
        self.pending_prepare = None;
        self.last_prepare = None;
        let result = match reply {
            crate::protocol::PythonMessage::Prepared {
                prompt_token_ids, ..
            } => Ok(prompt_token_ids),
            crate::protocol::PythonMessage::Error(e) => {
                if e.kind == crate::protocol::WorkerErrorKind::Validation {
                    Err(Error::WorkerValidation(e.message))
                } else {
                    Err(Error::WorkerError(e.message))
                }
            }
            other => {
                return Err(Error::UnexpectedMessageType(format!(
                    "wrong preparation reply type: {other:?}"
                )));
            }
        };
        Ok(Some(result))
    }

    /// Execute one inference step and return per-request next tokens.
    pub async fn execute_one_step(
        &mut self,
        scheduled: &[ScheduledRequest],
        finished_request_ids: &[u64],
        preempted_request_ids: &[u64],
        num_batched_tokens: usize,
    ) -> Result<WorkerOutput> {
        if scheduled.is_empty()
            && finished_request_ids.is_empty()
            && preempted_request_ids.is_empty()
        {
            debug!("empty batch — skipping execute");
            return Ok(WorkerOutput {
                outputs: Vec::new(),
            });
        }

        self.step_id += 1;
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        let started = Instant::now();
        let msg = RustMessage::Execute(ExecuteMsg {
            rpc_id,
            step_id: self.step_id,
            scheduled: scheduled
                .iter()
                .map(|s| ScheduledRequestMsg {
                    request_id: s.request_id,
                    token_ids: s.token_ids.clone(),
                    prefill_token_ids: s.prefill_token_ids.clone(),
                    num_computed_tokens: s.num_computed_tokens as u32,
                    fa_block_table: s.fa_block_table.clone(),
                    mamba_block_table: s.mamba_block_table.clone(),
                })
                .collect(),
            finished_request_ids: finished_request_ids.to_vec(),
            preempted_request_ids: preempted_request_ids.to_vec(),
            num_batched_tokens: num_batched_tokens as u32,
        });
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
            crate::protocol::PythonMessage::ExecuteResult(r) => {
                if !self.serving_enabled
                    && let Some(error) = r.outputs.iter().find_map(|output| output.error.as_ref())
                {
                    return Err(Error::WorkerError(error.clone()));
                }
                if self.serving_enabled {
                    for output in &r.outputs {
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
                Ok(WorkerOutput {
                    outputs: r
                        .outputs
                        .into_iter()
                        .filter(|o| o.error.is_none())
                        .map(|o| RequestOutput {
                            request_id: o.request_id,
                            token_ids: o.token_ids,
                            num_accepted_draft_tokens: o.num_accepted_draft_tokens as usize,
                            new_draft_token_ids: o.new_draft_token_ids,
                        })
                        .collect(),
                })
            }
            crate::protocol::PythonMessage::Error(e) => Err(Error::WorkerError(e.message)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    /// Send a graceful shutdown to the Python worker.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("sending shutdown to Python worker");
        Self::send_raw(&mut self.sock, &RustMessage::Shutdown).await?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self._child.child().try_wait()? {
                return if status.success() {
                    Ok(())
                } else {
                    Err(Error::WorkerDied)
                };
            }
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    async fn recv_for(&mut self, expected: u64) -> Result<crate::protocol::PythonMessage> {
        if self.pending_prepare.is_some_and(|(_, id)| id == expected)
            && let Some(reply) = self.prepared_reply.take()
        {
            return Ok(reply);
        }
        loop {
            let reply = self.recv_reply().await?;
            let received = match &reply {
                crate::protocol::PythonMessage::Prepared { rpc_id, .. } => *rpc_id,
                crate::protocol::PythonMessage::ExecuteResult(result) => result.rpc_id,
                crate::protocol::PythonMessage::ModeChanged { rpc_id, .. } => *rpc_id,
                crate::protocol::PythonMessage::Error(error) => error.rpc_id,
                other => {
                    return Err(Error::UnexpectedMessageType(format!("{other:?}")));
                }
            };
            if received == expected {
                let valid = match &reply {
                    crate::protocol::PythonMessage::Prepared { .. } => {
                        self.pending_prepare.is_some_and(|(_, id)| id == expected)
                    }
                    crate::protocol::PythonMessage::ExecuteResult(_) => {
                        self.pending_prepare.is_none_or(|(_, id)| id != expected)
                    }
                    crate::protocol::PythonMessage::ModeChanged { .. } => {
                        self.comparison && !self.serving_enabled && self.pending_prepare.is_none()
                    }
                    crate::protocol::PythonMessage::Error(_) => true,
                    _ => false,
                };
                if !valid {
                    return Err(Error::UnexpectedMessageType(format!(
                        "wrong reply type for RPC {expected}: {reply:?}"
                    )));
                }
                return Ok(reply);
            }
            if received > expected {
                return Err(Error::UnexpectedMessageType(format!(
                    "future RPC reply {received} while waiting for {expected}"
                )));
            }
            if self.pending_prepare.is_some_and(|(_, id)| id == received) {
                if self.prepared_reply.is_some() {
                    return Err(Error::UnexpectedMessageType(format!(
                        "duplicate preparation reply for RPC {received}"
                    )));
                }
                if !matches!(
                    reply,
                    crate::protocol::PythonMessage::Prepared { .. }
                        | crate::protocol::PythonMessage::Error(_)
                ) {
                    return Err(Error::UnexpectedMessageType(format!(
                        "wrong preparation reply for RPC {received}: {reply:?}"
                    )));
                }
                self.prepared_reply = Some(reply);
                debug!(
                    received,
                    expected, "prepare reply buffered before execute reply"
                );
            } else if matches!(
                reply,
                crate::protocol::PythonMessage::Prepared { .. }
                    | crate::protocol::PythonMessage::Error(_)
            ) && self.cancelled_prepares.remove(&received)
            {
                warn!(received, expected, "discarding reply to cancelled RPC");
            } else {
                return Err(Error::UnexpectedMessageType(format!(
                    "unsolicited stale RPC reply {received} while waiting for {expected}"
                )));
            }
        }
    }

    async fn send_raw<S: SocketSend + Send>(sock: &mut S, msg: &RustMessage) -> Result<()> {
        let bytes = encode(msg)?;
        let zmq_msg = ZmqMessage::from(bytes);
        sock.send(zmq_msg).await?;
        Ok(())
    }

    async fn recv_with_child(
        sock: &mut DealerSocket,
        child: &mut ManagedChild,
    ) -> Result<crate::protocol::PythonMessage> {
        let receive = Self::recv_raw(sock);
        tokio::pin!(receive);
        let mut heartbeat = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                reply = &mut receive => return reply,
                _ = heartbeat.tick() => {
                    if child.child().try_wait()?.is_some() { return Err(Error::WorkerDied); }
                }
            }
        }
    }

    async fn recv_reply(&mut self) -> Result<crate::protocol::PythonMessage> {
        let mut heartbeat = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                reply = self.inbox.recv() => return reply.ok_or(Error::WorkerDied)?,
                _ = heartbeat.tick() => {
                    if self._child.child().try_wait()?.is_some() {
                        return Err(Error::WorkerDied);
                    }
                }
            }
        }
    }

    async fn recv_raw<S: SocketRecv + Send>(
        sock: &mut S,
    ) -> Result<crate::protocol::PythonMessage> {
        let raw: ZmqMessage = sock.recv().await?;
        let bytes = raw.into_vec().into_iter().next().unwrap_or_default();
        Ok(decode(&bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "current_thread")]
    async fn child_drop_reaps_without_blocking_the_executor() {
        let child = Command::new("/bin/sleep").arg("10").spawn().unwrap();
        let pid = child.id();
        let start = Instant::now();
        drop(ManagedChild(Some(child)));
        assert!(start.elapsed() < Duration::from_millis(250));
        timeout(Duration::from_secs(2), async {
            while PathBuf::from(format!("/proc/{pid}")).exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("killed child must be reaped by the background waiter");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn ready_wait_uses_the_original_init_deadline() {
        use std::os::unix::fs::PermissionsExt;

        let base = std::env::temp_dir().join(format!("oh-my-vllm-silent-{}", std::process::id()));
        let script = base.with_extension("py");
        let socket = base.with_extension("ipc");
        std::fs::write(
            &script,
            concat!(
                "#!/usr/bin/env python3\n",
                "import sys, time, zmq\n",
                "sock = zmq.Context.instance().socket(zmq.DEALER)\n",
                "sock.connect(sys.argv[sys.argv.index('--socket') + 1])\n",
                "sock.recv()\n",
                "time.sleep(10)\n",
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let config = WorkerConfig {
            model_path: PathBuf::from("fixture-model"),
            python_executable: script.clone(),
            socket_path: socket.clone(),
            init_timeout: Duration::from_millis(250),
            ..WorkerConfig::default()
        };
        let start = Instant::now();
        let result = WorkerClient::launch(config).await;
        let error = result.err().expect("silent worker must not become ready");
        assert!(matches!(error, Error::Timeout), "{error:?}");
        assert!(start.elapsed() < Duration::from_secs(1));
        let _ = std::fs::remove_file(script);
        let _ = std::fs::remove_file(socket);
    }

    #[tokio::test]
    async fn empty_model_is_rejected_before_worker_launch() {
        let config = WorkerConfig {
            model_path: PathBuf::new(),
            python_executable: PathBuf::from("/bin/false"),
            ..WorkerConfig::default()
        };
        let result = WorkerClient::launch(config).await;
        assert!(matches!(result, Err(Error::Io(error))
            if error.kind() == std::io::ErrorKind::InvalidInput));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn comparison_mode_rpc_accepts_correlated_replies_and_keeps_guards() {
        use std::os::unix::fs::PermissionsExt;

        let base = std::env::temp_dir().join(format!("oh-my-vllm-modes-{}", std::process::id()));
        let script = base.with_extension("py");
        let socket = base.with_extension("ipc");
        std::fs::write(
            &script,
            concat!(
                "#!/usr/bin/env python3\n",
                "import sys, zmq, msgpack\n",
                "sock = zmq.Context.instance().socket(zmq.DEALER)\n",
                "sock.connect(sys.argv[sys.argv.index('--socket') + 1])\n",
                "init = msgpack.unpackb(sock.recv(), raw=False)\n",
                "sock.send(msgpack.packb(dict(type='ready', ",
                "logical_num_blocks=42, mamba_blocks=42)))\n",
                "while True:\n",
                "    msg = msgpack.unpackb(sock.recv(), raw=False)\n",
                "    if msg['type'] == 'shutdown': break\n",
                "    assert msg['type'] == 'set_speculative_mode'\n",
                "    sock.send(msgpack.packb(dict(type='mode_changed', ",
                "rpc_id=msg['rpc_id'], mode=msg['mode'])))\n",
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let config = WorkerConfig {
            model_path: PathBuf::from("fixture-model"),
            python_executable: script.clone(),
            socket_path: socket.clone(),
            comparison: true,
            num_speculative_tokens: 4,
            draft_model_path: Some(PathBuf::from("fixture-draft")),
            ..WorkerConfig::default()
        };
        let mut client = WorkerClient::launch(config).await.unwrap();
        for mode in ["dspark", "mtp"] {
            client.set_speculative_mode(mode).await.unwrap();
        }
        assert_eq!(client.rpc_id, 2);
        let last_rpc = client.rpc_id;
        assert!(client.set_speculative_mode("invalid").await.is_err());
        client.serving_enabled = true;
        assert!(client.set_speculative_mode("dspark").await.is_err());
        client.serving_enabled = false;
        client.comparison = false;
        assert!(client.set_speculative_mode("dspark").await.is_err());
        assert_eq!(client.rpc_id, last_rpc);
        client.shutdown().await.unwrap();
        drop(client);
        let _ = std::fs::remove_file(script);
        let _ = std::fs::remove_file(socket);
    }

    #[tokio::test]
    async fn reports_child_exit_before_connection_without_waiting_for_init_timeout() {
        let socket_path =
            std::env::temp_dir().join(format!("oh-my-vllm-dead-child-{}.ipc", std::process::id()));
        let config = WorkerConfig {
            model_path: PathBuf::from("fixture-model"),
            python_executable: PathBuf::from("/bin/false"),
            socket_path: socket_path.clone(),
            init_timeout: Duration::from_secs(60),
            ..WorkerConfig::default()
        };
        let result = timeout(Duration::from_secs(2), WorkerClient::launch(config))
            .await
            .expect("dead worker must be detected before the initialization deadline");
        assert!(matches!(result, Err(Error::WorkerDied)));
        let _ = std::fs::remove_file(socket_path);
    }
}
