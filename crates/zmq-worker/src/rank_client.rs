use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use oh_my_vllm_kv_cache::state::StateTransaction;
use oh_my_vllm_model_contract::ModelDescriptor;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSendHalf, ZmqMessage};

use crate::client::lifecycle::{Binding, ManagedChild, reap_worker_group};
use crate::group::protocol::{GROUP_PROTOCOL_VERSION, RankHeader, StepPlan};
use crate::group::{CompletedStep, GroupCapacities, GroupGate, GroupLimits};

mod encoding;
mod model;
mod protocol;
mod shutdown;
mod startup;

pub use model::BorrowedModelResult;

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
    borrowed_in_flight: bool,
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
            !self.closed && self.pending.is_empty() && !self.borrowed_in_flight,
            "group is closed or a step is pending"
        );
        self.validate_transactions(&plan, transactions.iter())?;
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

    fn validate_transactions<'a>(
        &self,
        plan: &StepPlan,
        transactions: impl ExactSizeIterator<Item = &'a StateTransaction>,
    ) -> Result<()> {
        ensure!(
            plan.requests.len() == transactions.len(),
            "transaction count differs from schedule"
        );
        let count = transactions.len();
        let keys: std::collections::BTreeSet<_> = transactions
            .map(|transaction| {
                let key = transaction.key();
                (key.epoch, key.step_id, key.request_id, key.transaction_id)
            })
            .collect();
        ensure!(
            keys.len() == count
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
