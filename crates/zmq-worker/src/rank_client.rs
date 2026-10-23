use std::ffi::OsString;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use oh_my_vllm_kv_cache::state::StateTransaction;
use oh_my_vllm_model_contract::ModelDescriptor;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSendHalf, DealerSocket, SocketOptions, ZmqMessage};

use crate::client::lifecycle::{Binding, IpcPath, ManagedChild, reap_worker_group, spawn};
use crate::group::protocol::{GROUP_PROTOCOL_VERSION, RankHeader, StepPlan};
use crate::group::{CompletedStep, GroupCapacities, GroupGate, GroupLimits};

mod encoding;
mod model;
mod protocol;
mod shutdown;

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupTransport {
    None,
    Nccl,
}

pub struct GroupConfig {
    pub epoch: u64,
    pub descriptor: ModelDescriptor,
    pub device_uuids: Vec<String>,
    pub transport: GroupTransport,
    pub limits: GroupLimits,
    pub socket_directory: PathBuf,
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
    pub startup_timeout: Duration,
    pub step_timeout: Duration,
}

pub struct CompletedStorage {
    completion: CompletedStep,
    transactions: Vec<StateTransaction>,
}

impl CompletedStorage {
    pub fn completion(&self) -> &CompletedStep {
        &self.completion
    }

    pub fn into_parts(self) -> (CompletedStep, Vec<StateTransaction>) {
        (self.completion, self.transactions)
    }
}

pub struct GroupClient {
    gate: GroupGate,
    senders: Vec<DealerSendHalf>,
    readers: Vec<JoinHandle<()>>,
    children: Vec<ManagedChild>,
    bindings: Vec<Binding>,
    inbox: mpsc::Receiver<(u32, Result<protocol::Reply>)>,
    pending: Vec<StateTransaction>,
    step_timeout: Duration,
    epoch: u64,
    closed: bool,
    cleanup: Option<CleanupWait>,
}

struct CleanupWait {
    receiver: Option<oneshot::Receiver<Result<()>>>,
    failure: Option<String>,
}

impl CleanupWait {
    fn new(receiver: oneshot::Receiver<Result<()>>) -> Self {
        Self {
            receiver: Some(receiver),
            failure: None,
        }
    }

    async fn wait(&mut self) -> Result<()> {
        if let Some(receiver) = self.receiver.as_mut() {
            let result = receiver
                .await
                .context("rank cleanup stopped without completion")
                .and_then(|result| result);
            self.receiver = None;
            self.failure = result.err().map(|error| error.to_string());
        }
        if let Some(failure) = &self.failure {
            bail!("rank cleanup failed: {failure}");
        }
        Ok(())
    }
}

struct ReapBundle {
    children: Vec<ManagedChild>,
    pending: Vec<StateTransaction>,
}

fn reap(mut bundle: ReapBundle) -> Result<()> {
    let mut failure = None;
    for child in &mut bundle.children {
        if let Err(error) = child.kill_group() {
            failure.get_or_insert(error);
        }
    }
    for child in &mut bundle.children {
        reap_worker_group(child.child());
    }
    drop(bundle.pending);
    match failure {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}

fn start_reaper(bundle: ReapBundle) -> oneshot::Receiver<Result<()>> {
    let (sender, receiver) = oneshot::channel();
    let work = Arc::new(Mutex::new(Some((bundle, sender))));
    let thread_work = Arc::clone(&work);
    if std::thread::Builder::new()
        .name("oh-my-vllm-rank-reap".into())
        .spawn(move || {
            let (bundle, sender) = thread_work.lock().unwrap().take().unwrap();
            let _ = sender.send(reap(bundle));
        })
        .is_err()
    {
        let (bundle, sender) = work.lock().unwrap().take().unwrap();
        let _ = sender.send(reap(bundle));
    }
    receiver
}

impl Drop for GroupClient {
    fn drop(&mut self) {
        for reader in &self.readers {
            reader.abort();
        }
        if !self.children.is_empty() || !self.pending.is_empty() {
            drop(start_reaper(ReapBundle {
                children: std::mem::take(&mut self.children),
                pending: std::mem::take(&mut self.pending),
            }));
        }
    }
}

fn deadline(duration: Duration) -> Result<Instant> {
    ensure!(!duration.is_zero(), "group timeout must be positive");
    Instant::now()
        .checked_add(duration)
        .context("group timeout exceeds the monotonic clock range")
}

impl GroupClient {
    pub async fn launch(config: GroupConfig) -> Result<Self> {
        let startup_deadline = deadline(config.startup_timeout)?;
        deadline(config.step_timeout)?;
        ensure!(
            config.socket_directory.is_dir(),
            "group IPC directory must exist"
        );
        let gate = GroupGate::new(
            config.epoch,
            config.descriptor.clone(),
            config.device_uuids.clone(),
            config.limits,
            startup_deadline,
        )?;
        let (sender, inbox) = mpsc::channel(16);
        let mut client = Self {
            gate,
            senders: Vec::new(),
            readers: Vec::new(),
            children: Vec::new(),
            bindings: Vec::new(),
            inbox,
            pending: Vec::new(),
            step_timeout: config.step_timeout,
            epoch: config.epoch,
            closed: false,
            cleanup: None,
        };
        let result = client
            .start(
                &config.program,
                &config.arguments,
                &config.socket_directory,
                &config.descriptor,
                &config.device_uuids,
                config.transport,
                sender,
                startup_deadline,
            )
            .await;
        if let Err(error) = result {
            client
                .close()
                .await
                .context("rank startup cleanup failed")?;
            return Err(error);
        }
        Ok(client)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "startup borrows config after its limits move"
    )]
    async fn start(
        &mut self,
        program: &std::path::Path,
        arguments: &[OsString],
        socket_directory: &std::path::Path,
        descriptor: &ModelDescriptor,
        device_uuids: &[String],
        transport: GroupTransport,
        inbox: mpsc::Sender<(u32, Result<protocol::Reply>)>,
        startup_deadline: Instant,
    ) -> Result<()> {
        for rank in 0..descriptor.parallel_plan.world_size {
            let path = socket_directory.join(format!("group-{}-{rank}.ipc", self.epoch));
            ensure!(
                std::fs::symlink_metadata(&path)
                    .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
                "group IPC path must not exist"
            );
            let address = format!("ipc://{}", path.display());
            let mut options = SocketOptions::default();
            options.peer_identity(format!("controller-{}-{rank}", self.epoch).parse()?);
            let mut socket = DealerSocket::with_options(options);
            socket.bind(&address).await?;
            let ipc = IpcPath {
                path: path.clone(),
                identity: std::fs::symlink_metadata(&path)?,
            };
            let mut binding_socket = DealerSocket::new();
            *binding_socket.binds() = std::mem::take(socket.binds());
            self.bindings.push(Binding {
                socket: Some(binding_socket),
                ipc: Some(ipc),
                runtime: tokio::runtime::Handle::current(),
            });
            let (send, mut receive) = socket.split();
            self.senders.push(send);
            let inbox = inbox.clone();
            self.readers.push(tokio::spawn(async move {
                loop {
                    let reply = receive_reply(&mut receive).await;
                    let failed = reply.is_err();
                    if inbox.send((rank, reply)).await.is_err() || failed {
                        break;
                    }
                }
            }));
            let mut command = Command::new(program);
            command
                .args(arguments)
                .args(["--socket", &address, "--rank", &rank.to_string()])
                .env("CUDA_VISIBLE_DEVICES", &device_uuids[rank as usize]);
            self.children.push(spawn(&mut command)?);
        }
        for rank in 0..descriptor.parallel_plan.world_size {
            let init = protocol::Command::Init {
                header: RankHeader {
                    version: GROUP_PROTOCOL_VERSION,
                    epoch: self.epoch,
                    rank,
                },
                descriptor,
                device_uuids,
                transport,
            };
            self.send(rank, &init, startup_deadline, true).await?;
        }
        let bootstrap_nonce = if transport == GroupTransport::Nccl {
            Some(self.bootstrap(device_uuids, startup_deadline).await?)
        } else {
            None
        };
        while self.gate.capacities().is_none() {
            let (rank, reply) = self.receive().await?;
            match reply {
                protocol::Reply::Ready(ready) => {
                    ensure!(
                        ready.bootstrap_nonce == bootstrap_nonce,
                        "rank readiness does not acknowledge its connect exchange"
                    );
                    self.gate.accept_ready(rank, ready, Instant::now())?;
                }
                protocol::Reply::Error { header, message } => {
                    bail!("rank {rank} ({header:?}): {message}")
                }
                protocol::Reply::Result(_) | protocol::Reply::ModelResult(_) => {
                    bail!("rank result before group readiness")
                }
                protocol::Reply::Closed { .. } => bail!("rank closed before group readiness"),
                protocol::Reply::Bootstrap(_) => {
                    bail!("unexpected rank bootstrap during readiness")
                }
            }
        }
        Ok(())
    }

    async fn bootstrap(&mut self, devices: &[String], end: Instant) -> Result<String> {
        let mut observed = vec![false; devices.len()];
        let mut unique_id = None;
        for _ in devices {
            let (rank, reply) = self.receive().await?;
            let bootstrap = match reply {
                protocol::Reply::Bootstrap(bootstrap) => bootstrap,
                protocol::Reply::Error { header, message } => {
                    bail!("rank {rank} ({header:?}): {message}")
                }
                _ => bail!("rank must complete bootstrap before readiness"),
            };
            ensure!(
                bootstrap.header == self.header(rank)
                    && devices.get(rank as usize) == Some(&bootstrap.device_uuid),
                "bootstrap identity differs from its bound connection"
            );
            ensure!(!observed[rank as usize], "duplicate rank bootstrap");
            observed[rank as usize] = true;
            if rank == 0 {
                ensure!(
                    bootstrap
                        .unique_id
                        .as_ref()
                        .is_some_and(|id| id.len() == 128),
                    "rank zero must supply a 128-byte NCCL unique ID"
                );
                unique_id = bootstrap.unique_id;
            } else {
                ensure!(
                    bootstrap.unique_id.is_none(),
                    "only rank zero can supply the NCCL unique ID"
                );
            }
        }
        let unique_id = unique_id.context("missing rank zero NCCL unique ID")?;
        let mut nonce_bytes = [0_u8; 32];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut nonce_bytes)?;
        let nonce: String = nonce_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        for rank in 0..devices.len() as u32 {
            let connect = protocol::Command::Connect {
                header: self.header(rank),
                unique_id: &unique_id,
                bootstrap_nonce: &nonce,
            };
            self.send(rank, &connect, end, false).await?;
        }
        Ok(nonce)
    }

    fn header(&self, rank: u32) -> RankHeader {
        RankHeader {
            version: GROUP_PROTOCOL_VERSION,
            epoch: self.epoch,
            rank,
        }
    }

    pub fn capacities(&self) -> Option<&GroupCapacities> {
        self.gate.capacities()
    }

    fn check_children(&mut self) -> Result<()> {
        for (rank, child) in self.children.iter_mut().enumerate() {
            ensure!(!child.has_exited()?, "rank {rank} exited");
        }
        Ok(())
    }

    async fn send(
        &mut self,
        rank: u32,
        message: &protocol::Command<'_>,
        deadline: Instant,
        retry: bool,
    ) -> Result<()> {
        let bytes = rmp_serde::to_vec_named(message)?;
        self.send_bytes(rank, &bytes, deadline, retry).await
    }

    async fn send_bytes(
        &mut self,
        rank: u32,
        bytes: &[u8],
        deadline: Instant,
        retry: bool,
    ) -> Result<()> {
        loop {
            self.check_children()?;
            self.gate.check_deadline(Instant::now())?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            let result = tokio::time::timeout(
                remaining,
                self.senders[rank as usize].send(ZmqMessage::from(bytes.to_vec())),
            )
            .await?;
            match result {
                Ok(()) => return Ok(()),
                Err(_) if retry => {
                    tokio::time::sleep(Duration::from_millis(20).min(remaining)).await
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    async fn receive(&mut self) -> Result<(u32, protocol::Reply)> {
        loop {
            self.check_children()?;
            self.gate.check_deadline(Instant::now())?;
            tokio::select! {
                reply = self.inbox.recv() => {
                    let (rank, reply) = reply.context("rank inbox closed")?;
                    return Ok((rank, reply?));
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }
    }

    pub async fn execute(
        &mut self,
        plan: StepPlan,
        transactions: Vec<StateTransaction>,
    ) -> Result<CompletedStorage> {
        ensure!(
            !self.closed && self.pending.is_empty(),
            "group is closed or a step is pending"
        );
        self.validate_transactions(&plan, &transactions)?;
        let end = deadline(self.step_timeout)?;
        let dispatch = self.gate.begin(plan, end, Instant::now())?;
        self.pending = transactions;
        let result = self.execute_dispatch(&dispatch, end).await;
        match result {
            Ok(completion) => Ok(CompletedStorage {
                completion,
                transactions: std::mem::take(&mut self.pending),
            }),
            Err(error) => {
                self.close()
                    .await
                    .context("rank execution cleanup failed")?;
                Err(error)
            }
        }
    }

    fn validate_transactions(
        &self,
        plan: &StepPlan,
        transactions: &[StateTransaction],
    ) -> Result<()> {
        ensure!(
            plan.requests.len() == transactions.len(),
            "transaction count differs from schedule"
        );
        let keys: std::collections::BTreeSet<_> = transactions
            .iter()
            .map(|transaction| {
                let key = transaction.key();
                (key.epoch, key.step_id, key.request_id, key.transaction_id)
            })
            .collect();
        ensure!(
            keys.len() == transactions.len()
                && plan.requests.iter().all(|request| {
                    keys.contains(&(
                        self.epoch,
                        plan.step_id,
                        request.request_id,
                        request.transaction_id,
                    ))
                }),
            "transaction ownership differs from schedule"
        );
        Ok(())
    }

    async fn execute_dispatch(
        &mut self,
        dispatch: &crate::group::protocol::StepDispatch,
        end: Instant,
    ) -> Result<CompletedStep> {
        for rank in 0..self.senders.len() as u32 {
            self.send(rank, &protocol::Command::Execute(dispatch), end, false)
                .await?;
        }
        loop {
            let (rank, reply) = self.receive().await?;
            match reply {
                protocol::Reply::Result(result) => {
                    if let Some(complete) = self.gate.accept_result(rank, result, Instant::now())? {
                        return Ok(complete);
                    }
                }
                protocol::Reply::ModelResult(_) => {
                    bail!("model result for metadata-only execution")
                }
                protocol::Reply::Error { header, message } => {
                    bail!("rank {rank} ({header:?}): {message}")
                }
                protocol::Reply::Ready(_) => bail!("unexpected rank readiness during execution"),
                protocol::Reply::Closed { .. } => bail!("rank closed during execution"),
                protocol::Reply::Bootstrap(_) => {
                    bail!("unexpected rank bootstrap during execution")
                }
            }
        }
    }

    pub async fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.closed = true;
            self.gate.fail_rank(0);
            self.senders.clear();
            let reaper = start_reaper(ReapBundle {
                children: std::mem::take(&mut self.children),
                pending: std::mem::take(&mut self.pending),
            });
            let readers = std::mem::take(&mut self.readers);
            let mut bindings = std::mem::take(&mut self.bindings);
            let (sender, receiver) = oneshot::channel();
            for reader in &readers {
                reader.abort();
            }
            tokio::spawn(async move {
                for reader in readers {
                    let _ = reader.await;
                }
                for binding in &mut bindings {
                    binding.close().await;
                }
                let result = match reaper.await {
                    Ok(result) => result,
                    Err(error) => Err(error.into()),
                };
                let _ = sender.send(result);
            });
            self.cleanup = Some(CleanupWait::new(receiver));
        }
        self.cleanup
            .as_mut()
            .expect("closed group owns cleanup")
            .wait()
            .await
    }
}

async fn receive_reply<S: SocketRecv + Send>(socket: &mut S) -> Result<protocol::Reply> {
    let message = socket.recv().await?;
    ensure!(message.len() == 1, "rank reply must have one msgpack frame");
    let bytes = message
        .into_vec()
        .into_iter()
        .next()
        .context("missing rank frame")?;
    ensure!(bytes.len() <= 1 << 20, "rank reply exceeds byte limit");
    Ok(rmp_serde::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_cleanup_wait_retains_its_completion_source() {
        let (sender, receiver) = oneshot::channel();
        let mut cleanup = CleanupWait::new(receiver);
        for _ in 0..2 {
            assert!(
                tokio::time::timeout(Duration::from_millis(1), cleanup.wait())
                    .await
                    .is_err()
            );
        }
        assert!(sender.send(Ok(())).is_ok());
        cleanup.wait().await.unwrap();
        cleanup.wait().await.unwrap();
        let (sender, receiver) = oneshot::channel();
        let mut cleanup = CleanupWait::new(receiver);
        assert!(sender.send(Err(anyhow::anyhow!("cleanup failure"))).is_ok());
        assert!(cleanup.wait().await.is_err());
        assert!(cleanup.wait().await.is_err());
    }
}
