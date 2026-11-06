use crate::client::lifecycle::{Binding, IpcPath, spawn};
use crate::group::GroupGate;
use crate::group::protocol::{GROUP_PROTOCOL_VERSION, RankHeader};
use crate::rank_client::{
    GroupClient, GroupConfig, GroupTransport, deadline, protocol, receive_reply,
};
use anyhow::{Context, Result, bail, ensure};
use oh_my_vllm_model_contract::ModelDescriptor;
use std::ffi::OsString;
use std::io::Read;
use std::process::Command;
use std::time::Instant;
use tokio::sync::mpsc;
use zeromq::prelude::Socket;
use zeromq::{DealerSocket, SocketOptions};

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
            borrowed_in_flight: false,
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
            let address = self
                .bind_rank(rank, socket_directory, inbox.clone())
                .await?;
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

    async fn bind_rank(
        &mut self,
        rank: u32,
        socket_directory: &std::path::Path,
        inbox: mpsc::Sender<(u32, Result<protocol::Reply>)>,
    ) -> Result<String> {
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
        Ok(address)
    }
}
