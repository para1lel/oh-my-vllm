#[path = "test_worker/rejection.rs"]
mod rejection;
#[path = "test_worker/termination.rs"]
mod termination;

use super::{WorkerClient, WorkerConfig};
use crate::client::lifecycle::{Binding, IpcPath, ManagedChild, worker_group_alive};
use crate::model_ready::validate_ready;
use crate::models::{ModelFamily, ModelPolicy};
use crate::protocol::{ModelDescriptor, encode};
use crate::{Error, Result};
use std::collections::BTreeSet;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::process::Stdio;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use zeromq::prelude::{Socket, SocketSend};
use zeromq::{DealerSocket, ZmqMessage};

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
        let descriptor = qwen_descriptor(&config);
        peer.send(ZmqMessage::from(encode(
            &crate::protocol::WorkerMessage::Ready {
                descriptor: descriptor.clone(),
                logical_num_blocks,
                mamba_blocks: config.mamba_blocks.unwrap(),
                protocol_version: crate::protocol::EXECUTION_PROTOCOL_VERSION,
                worker_epoch: config.worker_epoch,
                cache_layout: qwen_layout(&config),
                cache_memory: None,
            },
        )?))
        .await?;
        let model_family = match WorkerClient::recv_raw(&mut socket).await? {
            crate::protocol::WorkerMessage::Ready {
                descriptor,
                logical_num_blocks,
                mamba_blocks,
                ..
            } => validate_ready(
                ModelPolicy::for_family(ModelFamily::Qwen38),
                &config,
                &descriptor,
                logical_num_blocks,
                mamba_blocks,
            )?,
            other => return Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        };
        Self::connected(config, model_family, descriptor, socket, binding, peer)
    }

    fn connected(
        config: WorkerConfig,
        model_family: ModelFamily,
        descriptor: ModelDescriptor,
        socket: DealerSocket,
        binding: Binding,
        peer: DealerSocket,
    ) -> Result<Self> {
        let logical_num_blocks = config.num_gpu_blocks.expect("explicit test cache capacity");
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
            cache_layout: qwen_layout(&config),
            execution: crate::models::execution::ModelExecution::new(model_family, &descriptor),
            descriptor,
            model_family,
            config,
            sock: send,
            inbox,
            reader,
            _child: ManagedChild::new(child),
            step_id: 0,
            execution_trace: None,
            preparation_trace: Default::default(),
            rpc_id: 0,
            pending_prepare: None,
            last_prepare: None,
            cancelled_prepares: BTreeSet::new(),
            prepared_reply: None,
            serving_enabled: false,
            verified_reasoning_tokens: Default::default(),
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

pub(crate) fn qwen_descriptor(config: &WorkerConfig) -> ModelDescriptor {
    let fa_layers: Vec<_> = (3..64)
        .step_by(4)
        .chain((config.num_speculative_tokens == 4).then_some(64))
        .collect();
    let gdn_layers: Vec<_> = (0..64).filter(|layer| layer % 4 != 3).collect();
    serde_json::from_value(serde_json::json!({
        "schema_version": 1,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "checkpoint_sha256": "a".repeat(64),
        "config_sha256": "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc",
        "tokenizer_sha256": "b".repeat(64),
        "vocabulary_size": 248320,
        "max_context_tokens": 262144,
        "capabilities": [
            "text", "images", "video", "tools", "reasoning",
            "structured_outputs", "prefix_cache",
        ],
        "weight_formats": ["bf16", "f32", "qwen_fp8_e4m3_block128x128_fp32_scales"],
        "cache_schema": {
            "schema_version": 1,
            "groups": [
                {
                    "name": "fa", "kind": "full_attention",
                    "block_tokens": 128, "capacity_blocks": config.num_gpu_blocks.unwrap() + 1,
                    "storage": {
                        "type": "layout", "id": "qwen38_radix_fa_fp8kv128", "version": 1,
                    },
                    "layers": fa_layers, "source": {"type": "owned"},
                },
                {
                    "name": "gdn", "kind": "recurrent",
                    "block_tokens": 8192, "capacity_blocks": config.mamba_blocks.unwrap() + 1,
                    "storage": {
                        "type": "layout",
                        "id": "qwen38_radix_gdn8192_f32_conv_bf16_hidden_bf16",
                        "version": 1,
                    },
                    "layers": gdn_layers, "source": {"type": "owned"},
                },
            ],
        },
        "parallel_plan": {
            "strategy": "tp1", "world_size": 1, "ranks": [0], "request_placement": "replicated",
        },
        "speculation": [{
            "method": "mtp", "predictor_stages": 1,
            "max_draft_tokens": 4, "max_verification_rows": 5,
        }],
    }))
    .unwrap()
}

pub(crate) fn qwen_layout(config: &WorkerConfig) -> crate::protocol::CacheLayout {
    crate::protocol::CacheLayout::QwenRadixV1 {
        layout_id: crate::protocol::QWEN_LAYOUT_ID.into(),
        gdn_period_tokens: 8192,
        kv_page_tokens: 128,
        usable_checkpoints: config.mamba_blocks.unwrap(),
        usable_pages: config.num_gpu_blocks.unwrap(),
        null_checkpoints: 1,
        null_pages: 1,
        private_checkpoints: 6 * config.max_num_seqs,
        scratch_checkpoints: config.max_num_batched_tokens.div_ceil(8192) + config.max_num_seqs,
        pool_bytes: std::collections::BTreeMap::from([("fixture".into(), 1)]),
        format_fingerprint: crate::protocol::QWEN_FORMAT_FINGERPRINT.into(),
    }
}
