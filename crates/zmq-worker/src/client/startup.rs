use crate::client::lifecycle::{Binding, IpcPath, ManagedChild};
use crate::client::{WorkerClient, WorkerConfig, lifecycle};
use crate::error::{Error, Result};
use crate::model_ready::validate_ready;
use crate::models::{ModelFamily, ModelPolicy, select_model};
use crate::protocol::{InitMsg, MODEL_DESCRIPTOR_VERSION, ModelDescriptor, RustMessage};
use std::collections::BTreeSet;
use std::process::Command;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{info, warn};
use zeromq::DealerSocket;
use zeromq::prelude::Socket;

impl WorkerClient {
    pub async fn launch(config: WorkerConfig) -> Result<Self> {
        if config.model_path.as_os_str().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "set OH_MY_VLLM_MODEL or WorkerConfig.model_path",
            )
            .into());
        }
        let model = select_model(&config.model_path)?;
        Self::launch_selected(config, model).await
    }

    pub async fn launch_with_model(config: WorkerConfig, model: ModelPolicy) -> Result<Self> {
        model.validate_options(&config)?;
        if select_model(&config.model_path)? != model {
            return Err(Error::ModelConfiguration(
                "model selection changed before worker startup".into(),
            ));
        }
        Self::launch_selected(config, model).await
    }

    async fn launch_selected(mut config: WorkerConfig, model: ModelPolicy) -> Result<Self> {
        model.validate_options(&config)?;
        let (mut sock, binding) = Self::bind_socket(&config).await?;
        let addr = format!("ipc://{}", config.socket_path.display());

        let mut cmd = Command::new(&config.python_executable);
        cmd.args(["-m", "oh_my_vllm.worker.zmq_bridge", "--socket", &addr])
            .env(
                "OH_MY_VLLM_RUN_ID",
                std::env::var("OH_MY_VLLM_RUN_ID")
                    .unwrap_or_else(|_| format!("pid-{}", std::process::id())),
            );
        let mut child = lifecycle::spawn(&mut cmd)?;
        info!("Worker launched (pid {})", child.child().id());

        let init = Self::init_message(&config);
        let deadline = Instant::now() + config.init_timeout;
        Self::send_init(&mut sock, &mut child, &init, deadline).await?;

        let ready_reply = timeout(
            deadline.saturating_duration_since(Instant::now()),
            Self::recv_with_child(&mut sock, &mut child),
        )
        .await
        .map_err(|_| Error::Timeout)??;

        let (model_family, descriptor, logical_num_blocks, mamba_blocks, host_cache_bytes) =
            Self::validate_startup(model, &config, ready_reply)?;
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

        Ok(Self::connected(
            config,
            model_family,
            descriptor,
            sock,
            child,
            binding,
        ))
    }

    async fn bind_socket(config: &WorkerConfig) -> Result<(DealerSocket, Binding)> {
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

        Ok((sock, binding))
    }

    fn init_message(config: &WorkerConfig) -> RustMessage {
        RustMessage::Init(InitMsg {
            descriptor_version: MODEL_DESCRIPTOR_VERSION,
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
        })
    }

    async fn send_init(
        sock: &mut DealerSocket,
        child: &mut ManagedChild,
        init: &RustMessage,
        deadline: Instant,
    ) -> Result<()> {
        let mut delay_ms = 100u64;
        loop {
            if child.has_exited()? {
                return Err(Error::WorkerDied);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let sent = timeout(remaining, Self::send_raw(sock, init))
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
        Ok(())
    }

    fn validate_startup(
        model: ModelPolicy,
        config: &WorkerConfig,
        ready_reply: crate::protocol::WorkerMessage,
    ) -> Result<(ModelFamily, ModelDescriptor, u32, u32, u64)> {
        match ready_reply {
            crate::protocol::WorkerMessage::Ready {
                descriptor,
                logical_num_blocks,
                mamba_blocks,
                host_cache_bytes,
                cache_memory,
            } => {
                let model_family =
                    validate_ready(model, config, &descriptor, logical_num_blocks, mamba_blocks)?;
                info!(
                    logical_num_blocks,
                    mamba_blocks,
                    host_cache_bytes,
                    ?cache_memory,
                    "Worker is ready"
                );
                Ok((
                    model_family,
                    descriptor,
                    logical_num_blocks,
                    mamba_blocks,
                    host_cache_bytes,
                ))
            }
            crate::protocol::WorkerMessage::Error(e) => Err(Error::WorkerError(e.message)),
            other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        }
    }

    fn connected(
        config: WorkerConfig,
        model_family: ModelFamily,
        descriptor: ModelDescriptor,
        sock: DealerSocket,
        child: ManagedChild,
        binding: Binding,
    ) -> Self {
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
        let logical_num_blocks = config.num_gpu_blocks.expect("validated blocks");
        let mamba_blocks = config.mamba_blocks.expect("validated Mamba blocks");
        let host_cache_bytes = config.host_cache_bytes.expect("validated host cache");
        Self {
            config,
            execution: crate::models::execution::ModelExecution::new(model_family, &descriptor),
            descriptor,
            model_family,
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
        }
    }
}
