use std::collections::BTreeSet;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicUsize, Ordering};
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

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub model_path: PathBuf,
    pub socket_path: PathBuf,
    pub num_gpu_blocks: Option<u32>,
    pub mamba_blocks: Option<u32>,
    pub host_cache_bytes: Option<u64>,
    pub block_size: u32,
    pub tensor_parallel_size: u32,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub max_num_seqs: u32,
    pub num_speculative_tokens: usize,
    pub speculative_mode: Option<String>,
    pub python_executable: PathBuf,
    pub init_timeout: Duration,
    pub torch_profile: Option<crate::protocol::BenchmarkProfile>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model_path: std::env::var_os("OH_MY_VLLM_MODEL")
                .map(PathBuf::from)
                .unwrap_or_default(),
            socket_path: PathBuf::from("/tmp/oh-my-vllm.ipc"),
            num_gpu_blocks: None,
            mamba_blocks: Some(128),
            host_cache_bytes: None,
            block_size: 784,
            tensor_parallel_size: 1,
            max_model_len: 262144,
            max_num_batched_tokens: 32768,
            max_num_seqs: 4,
            num_speculative_tokens: 0,
            speculative_mode: None,
            python_executable: std::env::var_os("OH_MY_VLLM_WORKER_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("python3")),
            init_timeout: Duration::from_secs(300),
            torch_profile: None,
        }
    }
}

pub struct WorkerClient {
    config: WorkerConfig,
    sock: DealerSendHalf,
    inbox: mpsc::Receiver<Result<crate::protocol::WorkerMessage>>,
    reader: JoinHandle<()>,
    _child: ManagedChild,
    step_id: u64,
    rpc_id: u64,
    pending_prepare: Option<(u64, u64)>,
    last_prepare: Option<(u64, u64)>,
    cancelled_prepares: BTreeSet<u64>,
    prepared_reply: Option<crate::protocol::WorkerMessage>,
    serving_enabled: bool,
    pub logical_num_blocks: u32,
    pub mamba_blocks: u32,
    pub host_cache_bytes: u64,
    pub serving_outputs: std::collections::BTreeMap<u64, (String, Option<String>, usize)>,
    pub serving_errors: std::collections::BTreeMap<u64, String>,
    binding: Binding,
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

struct ManagedChild {
    child: Option<Child>,
    group_killed: bool,
}

static PENDING_REAPS: AtomicUsize = AtomicUsize::new(0);

struct IpcPath {
    path: PathBuf,
    identity: std::fs::Metadata,
}

struct Binding {
    socket: Option<DealerSocket>,
    ipc: Option<IpcPath>,
    runtime: tokio::runtime::Handle,
}

impl Binding {
    fn stop(&mut self) -> Option<JoinHandle<()>> {
        let mut socket = self.socket.take()?;
        let ipc = self.ipc.take();
        Some(self.runtime.spawn(async move {
            for error in socket.unbind_all().await {
                warn!(%error, "IPC unbind failed");
            }
            drop(ipc);
        }))
    }

    async fn close(&mut self) {
        if let Some(cleanup) = self.stop() {
            let _ = cleanup.await;
        }
    }
}

impl Drop for Binding {
    fn drop(&mut self) {
        drop(self.stop());
    }
}

impl Drop for IpcPath {
    fn drop(&mut self) {
        if let Ok(current) = std::fs::symlink_metadata(&self.path) {
            #[cfg(unix)]
            if (current.dev(), current.ino()) == (self.identity.dev(), self.identity.ino()) {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }
}

impl ManagedChild {
    fn new(child: Child) -> Self {
        Self {
            child: Some(child),
            group_killed: false,
        }
    }

    fn child(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("worker child remains owned until drop")
    }

    fn kill_group(&mut self) -> std::io::Result<()> {
        if !self.group_killed {
            kill_worker_group(self.child())?;
            self.group_killed = true;
        }
        Ok(())
    }

    fn has_exited(&mut self) -> std::io::Result<bool> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: waitid writes to initialized storage for the owned, unreaped child.
            unsafe {
                let mut info: libc::siginfo_t = std::mem::zeroed();
                if libc::waitid(
                    libc::P_PID,
                    self.child().id(),
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                ) == -1
                {
                    return Err(std::io::Error::last_os_error());
                }
                let exited = info.si_pid() != 0;
                if exited {
                    warn!(
                        worker_pid = self.child().id(),
                        wait_code = info.si_code,
                        wait_status = info.si_status(),
                        "worker exit observed before process-group cleanup"
                    );
                }
                Ok(exited)
            }
        }
        #[cfg(not(target_os = "linux"))]
        Ok(self.child().try_wait()?.is_some())
    }
}

fn kill_worker_group(child: &mut Child) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let group = i32::try_from(child.id()).map_err(std::io::Error::other)?;
        // SAFETY: the unreaped child owns this process group; kill does not access Rust memory.
        if unsafe { libc::kill(-group, libc::SIGKILL) } == -1 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        Ok(())
    }
    #[cfg(not(unix))]
    child.kill()
}

fn worker_group_alive(pid: u32) -> std::io::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        let group = pid.to_string();
        for entry in std::fs::read_dir("/proc")? {
            let entry = entry?;
            if !entry
                .file_name()
                .to_string_lossy()
                .bytes()
                .all(|b| b.is_ascii_digit())
            {
                continue;
            }
            let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
                continue;
            };
            let Some((_, fields)) = stat.rsplit_once(") ") else {
                continue;
            };
            let mut fields = fields.split_whitespace();
            let state = fields.next();
            let process_group = fields.nth(1);
            if process_group == Some(group.as_str()) && !matches!(state, Some("Z" | "X")) {
                return Ok(true);
            }
        }
        Ok(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        Ok(false)
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.kill_group();
        if let Some(child) = self.child.take() {
            PENDING_REAPS.fetch_add(1, Ordering::SeqCst);
            let child = Arc::new(Mutex::new(child));
            let waiter = Arc::clone(&child);
            if std::thread::Builder::new()
                .name("oh-my-vllm-worker-reap".to_owned())
                .spawn(move || {
                    reap_worker_group(&mut waiter.lock().unwrap());
                    PENDING_REAPS.fetch_sub(1, Ordering::SeqCst);
                })
                .is_err()
            {
                reap_worker_group(&mut child.lock().unwrap());
                PENDING_REAPS.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

fn reap_worker_group(child: &mut Child) {
    let pid = child.id();
    let _ = child.wait();
    while worker_group_alive(pid).unwrap_or(true) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

impl WorkerClient {
    pub async fn launch(mut config: WorkerConfig) -> Result<Self> {
        if config.model_path.as_os_str().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "set OH_MY_VLLM_MODEL or WorkerConfig.model_path",
            )
            .into());
        }
        let addr = format!("ipc://{}", config.socket_path.display());

        let mut sock = DealerSocket::new();
        sock.bind(&addr).await?;
        let ipc = IpcPath {
            path: config.socket_path.clone(),
            identity: std::fs::symlink_metadata(&config.socket_path)?,
        };
        let mut binding_socket = DealerSocket::new();
        *binding_socket.binds() = std::mem::take(sock.binds());
        let binding = Binding {
            socket: Some(binding_socket),
            ipc: Some(ipc),
            runtime: tokio::runtime::Handle::current(),
        };
        info!("ZMQ DEALER socket bound at {addr}");

        let mut child = ManagedChild::new({
            let mut cmd = Command::new(&config.python_executable);
            cmd.args(["-m", "oh_my_vllm.worker.zmq_bridge", "--socket", &addr])
                .env(
                    "OH_MY_VLLM_RUN_ID",
                    std::env::var("OH_MY_VLLM_RUN_ID")
                        .unwrap_or_else(|_| format!("pid-{}", std::process::id())),
                );
            #[cfg(unix)]
            cmd.process_group(0);
            // SAFETY: prctl is async-signal-safe; no Rust objects are touched.
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
                    if libc::getppid() != parent_pid {
                        return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                    }
                    Ok(())
                });
            }
            cmd.spawn()?
        });
        info!("Worker launched (pid {})", child.child().id());

        let init = RustMessage::Init(InitMsg {
            model_path: config.model_path.to_string_lossy().into_owned(),
            num_gpu_blocks: config.num_gpu_blocks,
            mamba_blocks: config.mamba_blocks,
            host_cache_bytes: config.host_cache_bytes,
            block_size: config.block_size,
            tensor_parallel_size: config.tensor_parallel_size,
            max_model_len: config.max_model_len,
            max_num_batched_tokens: config.max_num_batched_tokens,
            max_num_seqs: config.max_num_seqs,
            num_speculative_tokens: config.num_speculative_tokens,
            speculative_mode: config.speculative_mode.clone(),
            torch_profile: config.torch_profile.clone(),
        });
        let mut delay_ms = 100u64;
        let deadline = std::time::Instant::now() + config.init_timeout;
        loop {
            if child.has_exited()? {
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

        let ready_reply = timeout(
            deadline.saturating_duration_since(Instant::now()),
            Self::recv_with_child(&mut sock, &mut child),
        )
        .await
        .map_err(|_| Error::Timeout)??;

        let (logical_num_blocks, mamba_blocks, host_cache_bytes) = match ready_reply {
            crate::protocol::WorkerMessage::Ready {
                logical_num_blocks,
                mamba_blocks,
                host_cache_bytes,
                cache_memory,
            } => {
                info!(
                    logical_num_blocks,
                    mamba_blocks,
                    host_cache_bytes,
                    ?cache_memory,
                    "Worker is ready"
                );
                (logical_num_blocks, mamba_blocks, host_cache_bytes)
            }
            crate::protocol::WorkerMessage::Error(e) => {
                return Err(Error::WorkerError(e.message));
            }
            other => {
                return Err(Error::UnexpectedMessageType(format!("{other:?}")));
            }
        };
        if logical_num_blocks < 2
            || mamba_blocks < 2
            || config
                .num_gpu_blocks
                .is_some_and(|blocks| blocks != logical_num_blocks)
            || config
                .mamba_blocks
                .is_some_and(|blocks| blocks != mamba_blocks)
        {
            return Err(Error::WorkerError(
                "invalid worker cache capacities".to_owned(),
            ));
        }
        config.num_gpu_blocks = Some(logical_num_blocks);
        config.mamba_blocks = Some(mamba_blocks);
        if config
            .host_cache_bytes
            .is_some_and(|limit| host_cache_bytes > limit)
            || (config.host_cache_bytes.is_some_and(|limit| limit > 0) && host_cache_bytes == 0)
        {
            return Err(Error::UnexpectedMessageType(
                "host cache exceeds requested limit".into(),
            ));
        }
        config.host_cache_bytes = Some(host_cache_bytes);

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
            config,
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
            logical_num_blocks,
            mamba_blocks,
            host_cache_bytes,
            serving_outputs: Default::default(),
            serving_errors: Default::default(),
            binding,
        })
    }

    pub async fn register_request(&mut self, request_id: u64, prompt_token_ids: Vec<u32>) {
        self.send_registration(request_id, prompt_token_ids, None)
            .await;
    }

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
                        crate::protocol::WorkerMessage::Prepared { rpc_id, .. } => *rpc_id,
                        crate::protocol::WorkerMessage::ExecuteResult(result) => result.rpc_id,
                        crate::protocol::WorkerMessage::Error(error) => error.rpc_id,
                        other => {
                            return Err(Error::UnexpectedMessageType(format!("{other:?}")));
                        }
                    };
                    if received < rpc_id {
                        if matches!(
                            reply,
                            crate::protocol::WorkerMessage::Prepared { .. }
                                | crate::protocol::WorkerMessage::Error(_)
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
                    if self._child.has_exited()? {
                        return Err(Error::WorkerDied);
                    }
                    return Ok(None);
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return Err(Error::WorkerDied),
            }
        };
        self.finish_prepare_reply(reply).map(Some)
    }

    pub async fn wait_prepare_request(&mut self) -> Result<Result<Vec<u32>>> {
        let (_, rpc_id) = self.pending_prepare.ok_or_else(|| {
            Error::UnexpectedMessageType("no preparation is in flight".to_owned())
        })?;
        let reply = self.recv_for(rpc_id).await?;
        self.finish_prepare_reply(reply)
    }

    fn finish_prepare_reply(
        &mut self,
        reply: crate::protocol::WorkerMessage,
    ) -> Result<Result<Vec<u32>>> {
        self.pending_prepare = None;
        self.last_prepare = None;
        let result = match reply {
            crate::protocol::WorkerMessage::Prepared {
                prompt_token_ids, ..
            } => Ok(prompt_token_ids),
            crate::protocol::WorkerMessage::Error(e) => Err(worker_error(e)),
            other => {
                return Err(Error::UnexpectedMessageType(format!(
                    "wrong preparation reply type: {other:?}"
                )));
            }
        };
        Ok(result)
    }

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
                    fa_block_copies: s.fa_block_copies.clone(),
                    prefix_state_slot: s.prefix_state_slot,
                    cache_checkpoint: s.cache_checkpoint,
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
            crate::protocol::WorkerMessage::ExecuteResult(r) => {
                validate_reply_ids(scheduled, &r.outputs)?;
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
            crate::protocol::WorkerMessage::Error(e) => Err(worker_error(e)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        info!("sending worker shutdown");
        Self::send_raw(&mut self.sock, &RustMessage::Shutdown).await?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if self._child.has_exited()? {
                self._child.kill_group()?;
                let status = self._child.child().try_wait()?.ok_or(Error::WorkerDied)?;
                self.wait_for_worker_group().await?;
                self.binding.close().await;
                self.reader.abort();
                let _ = (&mut self.reader).await;
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

    pub fn configuration(&self) -> WorkerConfig {
        self.config.clone()
    }

    pub async fn cache_step(
        &mut self,
        scheduler: &mut oh_my_vllm_scheduler::Scheduler,
    ) -> Result<()> {
        if !scheduler.cache_pending() {
            return Ok(());
        }
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        let operations = scheduler.cache_operations();
        Self::send_raw(&mut self.sock, &RustMessage::Cache { rpc_id, operations }).await?;
        match self.recv_for(rpc_id).await? {
            crate::protocol::WorkerMessage::CacheResult {
                completed,
                host_cache_bytes,
                ..
            } => {
                if host_cache_bytes != self.host_cache_bytes {
                    return Err(Error::UnexpectedMessageType(
                        "host cache capacity changed".into(),
                    ));
                }
                scheduler
                    .complete_cache(&completed)
                    .map_err(|error| Error::UnexpectedMessageType(error.to_string()))
            }
            crate::protocol::WorkerMessage::Error(error) => Err(Error::WorkerError(error.message)),
            reply => Err(Error::UnexpectedMessageType(format!("{reply:?}"))),
        }
    }

    pub fn process_id(&mut self) -> u32 {
        self._child.child().id()
    }

    pub async fn terminate(&mut self) -> Result<()> {
        self.reader.abort();
        self._child.kill_group()?;
        let deadline = Instant::now() + Duration::from_secs(30);
        while self._child.child().try_wait()?.is_none() {
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        self.wait_for_worker_group().await?;
        self.binding.close().await;
        let _ = (&mut self.reader).await;
        Ok(())
    }

    async fn wait_for_worker_group(&mut self) -> Result<()> {
        let pid = self._child.child().id();
        let deadline = Instant::now() + Duration::from_secs(30);
        while worker_group_alive(pid)? {
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok(())
    }

    pub async fn wait_for_reapers() -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(30);
        while PENDING_REAPS.load(Ordering::SeqCst) != 0 {
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok(())
    }

    pub async fn query_stats(&mut self) -> Result<serde_json::Value> {
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        Self::send_raw(&mut self.sock, &RustMessage::Stats { rpc_id }).await?;
        match self.recv_for(rpc_id).await? {
            crate::protocol::WorkerMessage::StatsResult { stats, .. } => {
                if serde_json::to_vec(&stats)
                    .map_err(|error| Error::UnexpectedMessageType(error.to_string()))?
                    .len()
                    > 65536
                {
                    return Err(Error::UnexpectedMessageType(
                        "oversized worker stats".to_owned(),
                    ));
                }
                Ok(stats)
            }
            crate::protocol::WorkerMessage::Error(error) => Err(worker_error(error)),
            other => Err(Error::UnexpectedMessageType(format!(
                "wrong stats reply: {other:?}"
            ))),
        }
    }

    async fn recv_for(&mut self, expected: u64) -> Result<crate::protocol::WorkerMessage> {
        if self.pending_prepare.is_some_and(|(_, id)| id == expected)
            && let Some(reply) = self.prepared_reply.take()
        {
            return Ok(reply);
        }
        loop {
            let reply = self.recv_reply().await?;
            let received = match &reply {
                crate::protocol::WorkerMessage::Prepared { rpc_id, .. } => *rpc_id,
                crate::protocol::WorkerMessage::ExecuteResult(result) => result.rpc_id,
                crate::protocol::WorkerMessage::Error(error) => error.rpc_id,
                crate::protocol::WorkerMessage::StatsResult { rpc_id, .. } => *rpc_id,
                crate::protocol::WorkerMessage::CacheResult { rpc_id, .. } => *rpc_id,
                other => {
                    return Err(Error::UnexpectedMessageType(format!("{other:?}")));
                }
            };
            if received == expected {
                let valid = match &reply {
                    crate::protocol::WorkerMessage::Prepared { .. } => {
                        self.pending_prepare.is_some_and(|(_, id)| id == expected)
                    }
                    crate::protocol::WorkerMessage::ExecuteResult(_) => {
                        self.pending_prepare.is_none_or(|(_, id)| id != expected)
                    }
                    crate::protocol::WorkerMessage::Error(_) => true,
                    crate::protocol::WorkerMessage::StatsResult { .. } => true,
                    crate::protocol::WorkerMessage::CacheResult { .. } => true,
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
                    crate::protocol::WorkerMessage::Prepared { .. }
                        | crate::protocol::WorkerMessage::Error(_)
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
                crate::protocol::WorkerMessage::Prepared { .. }
                    | crate::protocol::WorkerMessage::Error(_)
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
    ) -> Result<crate::protocol::WorkerMessage> {
        let receive = Self::recv_raw(sock);
        tokio::pin!(receive);
        let mut heartbeat = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                reply = &mut receive => return reply,
                _ = heartbeat.tick() => {
                    if child.has_exited()? { return Err(Error::WorkerDied); }
                }
            }
        }
    }

    async fn recv_reply(&mut self) -> Result<crate::protocol::WorkerMessage> {
        let mut heartbeat = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                reply = self.inbox.recv() => return reply.ok_or(Error::WorkerDied)?,
                _ = heartbeat.tick() => {
                    if self._child.has_exited()? {
                        return Err(Error::WorkerDied);
                    }
                }
            }
        }
    }

    async fn recv_raw<S: SocketRecv + Send>(
        sock: &mut S,
    ) -> Result<crate::protocol::WorkerMessage> {
        let raw: ZmqMessage = sock.recv().await?;
        if raw.len() != 1 {
            return Err(Error::UnexpectedMessageType(
                "expected one msgpack frame".to_owned(),
            ));
        }
        let bytes = raw.into_vec().into_iter().next().unwrap_or_default();
        Ok(decode(&bytes)?)
    }
}

fn worker_error(error: crate::protocol::ErrorMsg) -> Error {
    use crate::protocol::WorkerErrorKind;
    match error.kind {
        WorkerErrorKind::Validation => Error::WorkerValidation(error.message),
        WorkerErrorKind::Internal => Error::WorkerError(error.message),
        WorkerErrorKind::Oom => Error::WorkerOom(error.message),
        WorkerErrorKind::Device => Error::WorkerDevice(error.message),
    }
}

fn validate_reply_ids(
    scheduled: &[ScheduledRequest],
    outputs: &[crate::protocol::RequestResultMsg],
) -> Result<()> {
    let expected: BTreeSet<_> = scheduled.iter().map(|request| request.request_id).collect();
    let mut received = BTreeSet::new();
    for output in outputs {
        if !expected.contains(&output.request_id) || !received.insert(output.request_id) {
            return Err(Error::UnexpectedMessageType(
                "unknown or duplicate execution request ID".to_owned(),
            ));
        }
        if output.error.is_some()
            && (!output.token_ids.is_empty()
                || output.num_accepted_draft_tokens != 0
                || !output.new_draft_token_ids.is_empty())
        {
            return Err(Error::UnexpectedMessageType(
                "error result contains successful token state".to_owned(),
            ));
        }
    }
    if received != expected {
        return Err(Error::UnexpectedMessageType(
            "missing execution request ID".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(all(test, unix))]
#[path = "test_worker.rs"]
pub(crate) mod fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::RequestResultMsg;

    #[test]
    fn execution_reply_ids_must_be_complete_unique_and_known() {
        let scheduled = [ScheduledRequest {
            request_id: 1,
            token_ids: vec![1],
            prefill_token_ids: None,
            num_computed_tokens: 0,
            fa_block_table: vec![],
            mamba_block_table: vec![],
            fa_block_copies: vec![],
            prefix_state_slot: None,
            cache_checkpoint: None,
        }];
        let output = |id| RequestResultMsg {
            request_id: id,
            error: None,
            token_ids: vec![2],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![],
            text: "a".to_owned(),
            finish_reason: None,
            reasoning_tokens: 0,
        };
        assert!(validate_reply_ids(&scheduled, &[output(1)]).is_ok());
        for invalid in [vec![], vec![output(2)], vec![output(1), output(1)]] {
            assert!(validate_reply_ids(&scheduled, &invalid).is_err());
        }
        let mut mixed = output(1);
        mixed.error = Some("failed".to_owned());
        assert!(validate_reply_ids(&scheduled, &[mixed]).is_err());
    }
}
