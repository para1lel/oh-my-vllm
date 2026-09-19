//! Async client handle for the Python model worker.
//!
//! `WorkerClient` owns the ZMQ PAIR socket and the Python child process.
//! The scheduler calls `execute_one_step` once per inference step.

use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

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
    /// Block size in tokens (Qwen3.5: 784).
    pub block_size: u32,
    /// Tensor parallel size (1 for single-GPU B200).
    pub tensor_parallel_size: u32,
    /// Maximum sequence length in tokens.
    pub max_model_len: u32,
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
            block_size: 784,
            tensor_parallel_size: 1,
            max_model_len: 65536,
            python_executable: PathBuf::from("python"),
            init_timeout: Duration::from_secs(300),
        }
    }
}

/// Handle to the running Python model worker.
pub struct WorkerClient {
    sock: DealerSocket,
    _child: Child,
}

impl WorkerClient {
    /// Launch the Python worker and wait for the `ready` handshake.
    pub async fn launch(config: WorkerConfig) -> Result<Self> {
        let addr = format!("ipc://{}", config.socket_path.display());

        // Bind on the Rust side; the Python side connects.
        let mut sock = DealerSocket::new();
        sock.bind(&addr).await?;
        info!("ZMQ PAIR socket bound at {addr}");

        // Fork the Python worker process.
        let child = Command::new(&config.python_executable)
            .args(["-m", "oh_my_vllm.worker.zmq_bridge", "--socket", &addr])
            .spawn()?;
        info!("Python worker launched (pid {})", child.id());

        // Send the init message.
        let init = RustMessage::Init(InitMsg {
            model_path: config.model_path.to_string_lossy().into_owned(),
            num_gpu_blocks: config.num_gpu_blocks,
            block_size: config.block_size,
            tensor_parallel_size: config.tensor_parallel_size,
            max_model_len: config.max_model_len,
        });
        Self::send_raw(&mut sock, &init).await?;

        // Wait for `ready`.
        let ready_reply = timeout(config.init_timeout, Self::recv_raw(&mut sock))
            .await
            .map_err(|_| Error::Timeout)??;

        match ready_reply {
            crate::protocol::PythonMessage::Ready => {
                info!("Python worker is ready");
            }
            crate::protocol::PythonMessage::Error(e) => {
                return Err(Error::WorkerError(e.message));
            }
            other => {
                return Err(Error::UnexpectedMessageType(format!("{other:?}")));
            }
        }

        Ok(Self {
            sock,
            _child: child,
        })
    }

    /// Tell the Python worker about a new request before it is first scheduled.
    pub async fn register_request(
        &mut self,
        request_id: u64,
        prompt_token_ids: Vec<u32>,
    ) -> Result<()> {
        let msg = RustMessage::Register(RegisterMsg {
            request_id,
            prompt_token_ids,
        });
        Self::send_raw(&mut self.sock, &msg).await
    }

    /// Tell the Python worker to drop a request (aborted or evicted permanently).
    pub async fn abort_request(&mut self, request_id: u64) -> Result<()> {
        let msg = RustMessage::Abort(AbortMsg { request_id });
        Self::send_raw(&mut self.sock, &msg).await
    }

    /// Execute one inference step and return per-request next tokens.
    pub async fn execute_one_step(
        &mut self,
        scheduled: &[ScheduledRequest],
        finished_request_ids: &[u64],
        preempted_request_ids: &[u64],
        num_batched_tokens: usize,
    ) -> Result<WorkerOutput> {
        if scheduled.is_empty() {
            debug!("empty batch — skipping execute");
            return Ok(WorkerOutput {
                outputs: Vec::new(),
            });
        }

        let msg = RustMessage::Execute(ExecuteMsg {
            scheduled: scheduled
                .iter()
                .map(|s| ScheduledRequestMsg {
                    request_id: s.request_id,
                    token_ids: s.token_ids.clone(),
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

        match Self::recv_raw(&mut self.sock).await? {
            crate::protocol::PythonMessage::ExecuteResult(r) => Ok(WorkerOutput {
                outputs: r
                    .outputs
                    .into_iter()
                    .map(|o| RequestOutput {
                        request_id: o.request_id,
                        next_token_id: o.next_token_id,
                        num_accepted_draft_tokens: o.num_accepted_draft_tokens as usize,
                        new_draft_token_ids: o.new_draft_token_ids,
                    })
                    .collect(),
            }),
            crate::protocol::PythonMessage::Error(e) => Err(Error::WorkerError(e.message)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    /// Send a graceful shutdown to the Python worker.
    pub async fn shutdown(&mut self) -> Result<()> {
        warn!("sending shutdown to Python worker");
        Self::send_raw(&mut self.sock, &RustMessage::Shutdown).await
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    async fn send_raw(sock: &mut DealerSocket, msg: &RustMessage) -> Result<()> {
        let bytes = encode(msg)?;
        let zmq_msg = ZmqMessage::from(bytes);
        sock.send(zmq_msg).await?;
        Ok(())
    }

    async fn recv_raw(sock: &mut DealerSocket) -> Result<crate::protocol::PythonMessage> {
        let raw: ZmqMessage = sock.recv().await?;
        let bytes = raw.into_vec().into_iter().next().unwrap_or_default();
        Ok(decode(&bytes)?)
    }
}
