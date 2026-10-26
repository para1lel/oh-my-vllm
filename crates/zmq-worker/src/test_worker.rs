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
        let descriptor = stub_descriptor(&config);
        peer.send(ZmqMessage::from(encode(
            &crate::protocol::WorkerMessage::Ready {
                descriptor: descriptor.clone(),
                logical_num_blocks,
                mamba_blocks: config.mamba_blocks.unwrap(),
                host_cache_bytes: config.host_cache_bytes.unwrap_or(0),
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
                ModelPolicy::for_family(ModelFamily::Stub),
                &config,
                &descriptor,
                logical_num_blocks,
                mamba_blocks,
            )?,
            other => return Err(Error::UnexpectedMessageType(format!("{other:?}"))),
        };
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
            descriptor,
            model_family,
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

pub(crate) fn stub_descriptor(config: &WorkerConfig) -> ModelDescriptor {
    serde_json::from_value(serde_json::json!({
        "schema_version": 1,
        "architecture": "OhMyVllmStubForTesting",
        "model_type": "oh_my_vllm_stub",
        "checkpoint_sha256": "6cafe580d6f80a355f3e30bf2c8a7559aef2b081845084d16e9eca733c09d7c2",
        "config_sha256": "3b7c7f7627a034cd753e4d4db30346a940e49b635952460c51b2ba5f4f332f18",
        "tokenizer_sha256": "243e380a5f7db0cea0f70d4ddc53663d47e9f2b191b928fa5ccbba201ecf99b7",
        "vocabulary_size": 248320,
        "max_context_tokens": 262144,
        "capabilities": [
            "text", "images", "video", "tools", "reasoning",
            "structured_outputs", "prefix_cache", "host_offload",
        ],
        "weight_formats": ["f32"],
        "cache_schema": {
            "schema_version": 1,
            "groups": [
                {
                    "name": "fa", "kind": "model_checkpoint",
                    "block_tokens": 784, "capacity_blocks": config.num_gpu_blocks.unwrap(),
                    "storage": {
                        "type": "layout", "id": "oh_my_vllm_stub_fa_metadata", "version": 1,
                    },
                    "layers": [0], "source": {"type": "owned"},
                },
                {
                    "name": "gdn", "kind": "model_checkpoint",
                    "block_tokens": 784, "capacity_blocks": config.mamba_blocks.unwrap(),
                    "storage": {
                        "type": "layout", "id": "oh_my_vllm_stub_gdn_metadata", "version": 1,
                    },
                    "layers": [0], "source": {"type": "owned"},
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

#[tokio::test]
async fn rejected_ready_reaps_the_owned_process_group_and_removes_the_socket() {
    use std::os::unix::fs::PermissionsExt;

    assert!(
        std::env::var_os("OH_MY_VLLM_WORKER_PYTHON").is_some(),
        "run CPU worker tests through scripts/with-env.sh"
    );
    for oversized in [false, true] {
        let directory = std::env::temp_dir().join(format!(
            "omv-ready-reject-{}-{oversized}",
            std::process::id(),
        ));
        std::fs::create_dir(&directory).unwrap();
        let socket = directory.join("worker.ipc");
        let launcher = directory.join("worker.sh");
        let source = r#"
import os
from pathlib import Path
import subprocess
import sys
import time
import msgpack
import zmq
from oh_my_vllm.models.stub import build_descriptor
address = sys.argv[sys.argv.index("--socket") + 1]
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt(zmq.LINGER, 0)
socket.connect(address)
init = msgpack.unpackb(socket.recv(), raw=False)
assert init["descriptor_version"] == 1
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
Path(address.removeprefix("ipc://") + ".pids").write_text(f"{os.getpid()} {child.pid}")
d = build_descriptor(1608, 128)
d["schema_version"] = 2
reply = dict(type="ready", descriptor=d, logical_num_blocks=1608, mamba_blocks=128)
payload = msgpack.packb(reply, use_bin_type=True)
if OVERSIZED:
    payload = b"x" * ((1 << 20) + 1)
socket.send(payload)
time.sleep(300)
"#;
        let source = source.replace(
            "OVERSIZED",
            if oversized {
                "True"
            } else {
                "False"
            },
        );
        assert!(!source.contains('\''));
        std::fs::write(
            &launcher,
            format!("#!/bin/sh\nexec \"$OH_MY_VLLM_WORKER_PYTHON\" -c '{source}' \"$@\"\n"),
        )
        .unwrap();
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o700)).unwrap();
        let config = WorkerConfig {
            socket_path: socket.clone(),
            num_gpu_blocks: Some(1608),
            mamba_blocks: Some(128),
            python_executable: launcher,
            init_timeout: Duration::from_secs(5),
            ..WorkerConfig::for_model("stub".into(), ModelPolicy::for_family(ModelFamily::Stub))
        };
        for selected in [false, true] {
            let invalid = WorkerConfig {
                tensor_parallel_size: 4,
                ..config.clone()
            };
            let result = if selected {
                WorkerClient::launch_with_model(invalid, ModelPolicy::for_family(ModelFamily::Stub))
                    .await
            } else {
                WorkerClient::launch(invalid).await
            };
            assert!(matches!(result, Err(Error::ModelConfiguration(_))));
            assert!(!socket.exists());
            assert!(!socket.with_extension("ipc.pids").exists());
        }
        assert!(matches!(
            WorkerClient::launch_with_model(
                config.clone(),
                ModelPolicy::for_family(ModelFamily::Qwen38),
            )
            .await,
            Err(Error::ModelConfiguration(_))
        ));
        assert!(!socket.exists());
        assert!(!socket.with_extension("ipc.pids").exists());
        std::fs::write(
            directory.join("config.json"),
            r#"{"architectures":["Unsupported"],"model_type":"unsupported"}"#,
        )
        .unwrap();
        let invalid = WorkerConfig {
            model_path: directory.clone(),
            ..config.clone()
        };
        assert!(matches!(
            WorkerClient::launch(invalid).await,
            Err(Error::ModelConfiguration(_))
        ));
        assert!(!socket.exists());
        assert!(!socket.with_extension("ipc.pids").exists());
        // Each synthetic worker and its child stay inside the launcher's owned group.
        let result = WorkerClient::launch(config).await;
        if oversized {
            assert!(matches!(
                result,
                Err(Error::UnexpectedMessageType(ref reason)) if reason.contains("oversized")
            ));
        } else {
            assert!(matches!(
                result,
                Err(Error::WorkerError(ref reason)) if reason.contains("descriptor")
            ));
        }
        WorkerClient::wait_for_reapers().await.unwrap();
        let pids: Vec<u32> = std::fs::read_to_string(socket.with_extension("ipc.pids"))
            .unwrap()
            .split_whitespace()
            .map(|pid| pid.parse().unwrap())
            .collect();
        assert_eq!(pids.len(), 2);
        assert!(!worker_group_alive(pids[0]).unwrap());
        let deadline = Instant::now() + Duration::from_secs(3);
        while socket.exists() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!socket.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
