//! Async client handle for the Python model worker.
//!
//! `WorkerClient` owns the ZMQ DEALER socket and the Python child process.
//! The scheduler calls `execute_one_step` once per inference step.

#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use oh_my_vllm_scheduler::output::{RequestOutput, ScheduledRequest, WorkerOutput};
use tokio::time::timeout;
use tracing::{debug, info, warn};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSocket, ZmqMessage};

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
    /// Python executable to use.
    pub python_executable: PathBuf,
    /// Timeout waiting for the worker to become ready after launch.
    pub init_timeout: Duration,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("/data0/shared/Qwen3.8-27B-FP8"),
            socket_path: PathBuf::from("/tmp/oh-my-vllm.ipc"),
            num_gpu_blocks: 1024,
            mamba_blocks: None,
            block_size: 784,
            tensor_parallel_size: 1,
            max_model_len: 65536,
            num_speculative_tokens: 0,
            python_executable: std::env::var_os("OH_MY_VLLM_WORKER_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from("/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python")
                }),
            init_timeout: Duration::from_secs(300),
        }
    }
}

/// Handle to the running Python model worker.
pub struct WorkerClient {
    sock: DealerSocket,
    _child: ManagedChild,
    step_id: u64,
    serving_enabled: bool,
    pub logical_num_blocks: u32,
    pub serving_outputs: std::collections::BTreeMap<u64, (String, Option<String>, usize)>,
}

/// Reap the owned worker on errors as well as normal shutdown.
struct ManagedChild(Child);

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl WorkerClient {
    /// Launch the Python worker and wait for the `ready` handshake.
    pub async fn launch(config: WorkerConfig) -> Result<Self> {
        let addr = format!("ipc://{}", config.socket_path.display());

        // Bind on the Rust side; the Python side connects.
        let mut sock = DealerSocket::new();
        sock.bind(&addr).await?;
        info!("ZMQ DEALER socket bound at {addr}");

        // Fork the Python worker process.
        let mut child = ManagedChild({
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
                cmd.pre_exec(|| {
                    libc::prctl(
                        libc::PR_SET_PDEATHSIG,
                        libc::SIGKILL as libc::c_ulong,
                        0,
                        0,
                        0,
                    );
                    Ok(())
                });
            }
            cmd.spawn()?
        });
        info!("Python worker launched (pid {})", child.0.id());

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
        });
        let mut delay_ms = 100u64;
        let deadline = std::time::Instant::now() + config.init_timeout;
        loop {
            if child.0.try_wait()?.is_some() {
                return Err(Error::WorkerDied);
            }
            match Self::send_raw(&mut sock, &init).await {
                Ok(()) => break,
                Err(e) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(e);
                    }
                    warn!("Init send failed ({e}), retrying in {delay_ms}ms…");
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 2).min(2000);
                }
            }
        }

        // Wait for `ready`.
        let ready_reply = timeout(
            config.init_timeout,
            Self::recv_with_child(&mut sock, &mut child),
        )
        .await
        .map_err(|_| Error::Timeout)??;

        let logical_num_blocks = match ready_reply {
            crate::protocol::PythonMessage::Ready { logical_num_blocks } => {
                info!(logical_num_blocks, "Python worker is ready");
                logical_num_blocks
            }
            crate::protocol::PythonMessage::Error(e) => {
                return Err(Error::WorkerError(e.message));
            }
            other => {
                return Err(Error::UnexpectedMessageType(format!("{other:?}")));
            }
        };

        Ok(Self {
            sock,
            _child: child,
            step_id: 0,
            serving_enabled: false,
            logical_num_blocks,
            serving_outputs: Default::default(),
        })
    }

    /// Tell the Python worker about a new request before it is first scheduled.
    /// Send failures are non-fatal: the worker may have already exited, and the
    /// next `execute_one_step` will detect the dead worker via `recv_with_child`.
    pub async fn register_request(&mut self, request_id: u64, prompt_token_ids: Vec<u32>) {
        debug!(
            request_id,
            input_tokens = prompt_token_ids.len(),
            "register_request"
        );
        let msg = RustMessage::Register(RegisterMsg {
            request_id,
            prompt_token_ids,
        });
        if let Err(e) = Self::send_raw(&mut self.sock, &msg).await {
            warn!(request_id, "register send failed: {e}");
        }
    }

    /// Prepare model input and sampling before Rust admission. The reply is correlated
    /// by the single-owner, sequential RPC stream.
    pub async fn prepare_request(
        &mut self,
        request_id: u64,
        request: serde_json::Value,
    ) -> Result<Vec<u32>> {
        self.serving_enabled = true;
        Self::send_raw(
            &mut self.sock,
            &RustMessage::Prepare {
                request_id,
                request,
            },
        )
        .await?;
        match Self::recv_with_child(&mut self.sock, &mut self._child).await? {
            crate::protocol::PythonMessage::Prepared { prompt_token_ids } => Ok(prompt_token_ids),
            crate::protocol::PythonMessage::Error(e) => Err(Error::WorkerError(e.message)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    /// Tell the Python worker to drop a request (aborted or evicted permanently).
    /// Send failures are non-fatal: the worker may have already exited.
    pub async fn abort_request(&mut self, request_id: u64) {
        let msg = RustMessage::Abort(AbortMsg { request_id });
        if let Err(e) = Self::send_raw(&mut self.sock, &msg).await {
            warn!(request_id, "abort send failed: {e}");
        }
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
        let started = Instant::now();
        let msg = RustMessage::Execute(ExecuteMsg {
            step_id: self.step_id,
            scheduled: scheduled
                .iter()
                .map(|s| ScheduledRequestMsg {
                    request_id: s.request_id,
                    token_ids: s.token_ids.clone(),
                    prefill_token_ids: s.prefill_token_ids.clone(),
                    new_block_ids_to_zero: s.new_block_ids_to_zero.clone(),
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

        let reply = Self::recv_with_child(&mut self.sock, &mut self._child).await?;
        debug!(
            step_id = self.step_id,
            requests = scheduled.len(),
            num_batched_tokens,
            elapsed_us = started.elapsed().as_micros() as u64,
            "execute_round_trip"
        );
        match reply {
            crate::protocol::PythonMessage::ExecuteResult(r) => {
                if self.serving_enabled {
                    self.serving_outputs = r
                        .outputs
                        .iter()
                        .map(|o| {
                            (
                                o.request_id,
                                (o.text.clone(), o.finish_reason.clone(), o.reasoning_tokens),
                            )
                        })
                        .collect();
                }
                Ok(WorkerOutput {
                    outputs: r
                        .outputs
                        .into_iter()
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
            if let Some(status) = self._child.0.try_wait()? {
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

    async fn send_raw(sock: &mut DealerSocket, msg: &RustMessage) -> Result<()> {
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
                    if child.0.try_wait()?.is_some() { return Err(Error::WorkerDied); }
                }
            }
        }
    }

    async fn recv_raw(sock: &mut DealerSocket) -> Result<crate::protocol::PythonMessage> {
        let raw: ZmqMessage = sock.recv().await?;
        let bytes = raw.into_vec().into_iter().next().unwrap_or_default();
        Ok(decode(&bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reports_child_exit_before_connection_without_waiting_for_init_timeout() {
        let socket_path =
            std::env::temp_dir().join(format!("oh-my-vllm-dead-child-{}.ipc", std::process::id()));
        let config = WorkerConfig {
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
