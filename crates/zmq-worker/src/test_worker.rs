use super::*;
use std::process::Stdio;
use std::sync::atomic::AtomicU64;

pub(crate) struct TestWorker {
    pub client: WorkerClient,
    pub socket: DealerSocket,
}

impl TestWorker {
    pub async fn launch(mut config: WorkerConfig) -> Result<Self> {
        static NEXT_SOCKET: AtomicU64 = AtomicU64::new(0);
        let logical_num_blocks = config.num_gpu_blocks.expect("explicit test cache capacity");
        config.socket_path = std::env::temp_dir().join(format!(
            "omv-admit-{}-{}.ipc",
            std::process::id(),
            NEXT_SOCKET.fetch_add(1, Ordering::Relaxed),
        ));
        let address = format!("ipc://{}", config.socket_path.display());
        let mut socket = DealerSocket::new();
        socket.bind(&address).await?;
        let ipc = IpcPath {
            path: config.socket_path.clone(),
            identity: std::fs::symlink_metadata(&config.socket_path)?,
        };
        let mut binding_socket = DealerSocket::new();
        *binding_socket.binds() = std::mem::take(socket.binds());
        let binding = Binding {
            socket: Some(binding_socket),
            ipc: Some(ipc),
            runtime: tokio::runtime::Handle::current(),
        };
        let mut peer = DealerSocket::new();
        peer.connect(&address).await?;
        peer.send(ZmqMessage::from(encode(
            &crate::protocol::WorkerMessage::Ready {
                logical_num_blocks,
                mamba_blocks: config.mamba_blocks.unwrap(),
                host_cache_bytes: config.host_cache_bytes.unwrap_or(0),
                cache_memory: None,
            },
        )?))
        .await?;
        WorkerClient::recv_raw(&mut socket).await?;
        let (send, mut receive) = socket.split();
        let (inbox_tx, inbox) = mpsc::channel(16);
        let reader = tokio::spawn(async move {
            loop {
                let reply = WorkerClient::recv_raw(&mut receive).await;
                let failed = reply.is_err();
                if inbox_tx.send(reply).await.is_err() || failed {
                    break;
                }
            }
        });
        let child = Command::new("/bin/sh")
            .args(["-c", "read -r value"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()?;
        let client = WorkerClient {
            logical_num_blocks,
            mamba_blocks: config.mamba_blocks.unwrap(),
            host_cache_bytes: config.host_cache_bytes.unwrap_or(0),
            config,
            sock: send,
            inbox,
            reader,
            _child: ManagedChild::new(child),
            step_id: 0,
            rpc_id: 0,
            pending_prepare: None,
            last_prepare: None,
            cancelled_prepares: BTreeSet::new(),
            prepared_reply: None,
            serving_enabled: false,
            serving_outputs: Default::default(),
            serving_errors: Default::default(),
            binding,
        };
        Ok(Self {
            client,
            socket: peer,
        })
    }

    pub async fn close(mut self) {
        let path = self.client.configuration().socket_path;
        let pid = self.client.process_id();
        let cleanup = self.client.binding.stop();
        drop(self);
        if let Some(cleanup) = cleanup {
            cleanup.await.unwrap();
        }
        WorkerClient::wait_for_reapers().await.unwrap();
        assert!(!path.exists());
        assert!(!worker_group_alive(pid).unwrap());
    }
}
