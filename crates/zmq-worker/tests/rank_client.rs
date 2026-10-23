use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use oh_my_vllm_kv_cache::state::{AllocationRequest, StateArena, StateTransaction};
use oh_my_vllm_model_contract::ModelDescriptor;
use oh_my_vllm_zmq_worker::group::GroupLimits;
use oh_my_vllm_zmq_worker::group::protocol::{RequestWork, StepPlan};
use oh_my_vllm_zmq_worker::rank_client::{GroupClient, GroupConfig, GroupTransport};
use serde_json::json;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Directory(PathBuf);

struct UnrelatedProcess(std::process::Child);

impl Drop for UnrelatedProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "omv-ranks-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn assert_stopped(&self) {
        for entry in std::fs::read_dir(&self.0).unwrap() {
            let path = entry.unwrap().path();
            assert_ne!(path.extension().and_then(|ext| ext.to_str()), Some("ipc"));
            if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                let record: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                for key in ["pid", "child"] {
                    let pid = record[key].as_u64().unwrap();
                    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                        let state = stat
                            .rsplit_once(") ")
                            .unwrap()
                            .1
                            .split_whitespace()
                            .next()
                            .unwrap();
                        assert!(matches!(state, "Z" | "X"), "process {pid} still active");
                    }
                }
            }
        }
    }

    fn assert_devices(&self, expected: &[String]) {
        for (rank, device) in expected.iter().enumerate() {
            let record: serde_json::Value = serde_json::from_slice(
                &std::fs::read(self.0.join(format!("rank-{rank}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(record["visible_devices"], *device);
        }
    }

    fn assert_graceful(&self) {
        for rank in 0..4 {
            assert!(self.0.join(format!("closed-{rank}.sent")).exists());
        }
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn descriptor(strategy: &str) -> ModelDescriptor {
    serde_json::from_value(json!({
        "schema_version": 1, "architecture": "SyntheticForGeneration", "model_type": "synthetic",
        "checkpoint_sha256": "a".repeat(64), "config_sha256": "b".repeat(64),
        "tokenizer_sha256": "c".repeat(64), "vocabulary_size": 101, "max_context_tokens": 97,
        "capabilities": ["text"], "weight_formats": ["bf16"], "speculation": [],
        "cache_schema": {"schema_version": 1, "groups": [
            {"name": "state", "kind": "model_checkpoint", "block_tokens": 1, "capacity_blocks": 4,
             "storage": {"type": "primitive", "format": "f32"},
             "layers": [0], "source": {"type": "owned"}}
        ]},
        "parallel_plan": {"strategy": strategy, "world_size": 4, "ranks": [0, 1, 2, 3],
            "request_placement": if strategy == "dep4" {"rank_partitioned"} else {"replicated"}}
    }))
    .unwrap()
}

fn config(directory: &Directory, strategy: &str, failure: &str) -> GroupConfig {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    GroupConfig {
        epoch: 7,
        descriptor: descriptor(strategy),
        device_uuids: (0..4)
            .map(|rank| format!("GPU-{:08x}-{:04x}-{:04x}-{:04x}-{rank:012x}", 0, 0, 0, 0))
            .collect(),
        transport: GroupTransport::None,
        limits: GroupLimits {
            max_slots: 16,
            max_requests: 64,
            max_rows: 1024,
        },
        socket_directory: directory.0.clone(),
        program: std::env::var_os("OH_MY_VLLM_WORKER_PYTHON")
            .unwrap_or_else(|| "python3".into())
            .into(),
        arguments: vec![
            root.join("tests/rank_peer.py").into_os_string(),
            "--records".into(),
            directory.0.clone().into_os_string(),
            "--failure".into(),
            failure.into(),
        ],
        startup_timeout: Duration::from_secs(5),
        step_timeout: Duration::from_millis(300),
    }
}

fn state(strategy: &str) -> StateArena {
    let model = descriptor(strategy);
    StateArena::new(7, &model.cache_schema, &model.parallel_plan, 16).unwrap()
}

fn step(state: &StateArena) -> (StepPlan, Vec<StateTransaction>) {
    let transaction = state
        .reserve(
            1,
            10,
            &[AllocationRequest {
                group: "state".into(),
                rank: 0,
                blocks: 3,
            }],
            &[],
        )
        .unwrap();
    let plan = StepPlan {
        step_id: 1,
        collective_plan: serde_json::from_value(
            serde_json::from_str::<serde_json::Value>(include_str!(
                "../../../tests/fixtures/collectives.json"
            ))
            .unwrap()["plan"]
                .clone(),
        )
        .unwrap(),
        requests: vec![RequestWork {
            request_id: 10,
            transaction_id: transaction.key().transaction_id,
            owner_rank: 0,
            rows: 3,
            prefill: true,
        }],
    };
    (plan, vec![transaction])
}

#[tokio::test]
async fn four_process_quorum_returns_storage_only_after_all_ranks_finish() {
    for strategy in ["tp4", "tep4", "dep4"] {
        let directory = Directory::new();
        let mut client = GroupClient::launch(config(&directory, strategy, "none"))
            .await
            .unwrap();
        directory.assert_devices(&config(&directory, strategy, "none").device_uuids);
        assert_eq!(client.capacities().unwrap().owned_slots, 16);
        let arena = state(strategy);
        let (plan, transactions) = step(&arena);
        assert_eq!(arena.free_blocks("state", 0).unwrap(), 0);
        let completed = client.execute(plan, transactions).await.unwrap();
        assert_eq!(completed.completion().transactions(), &[(10, 1)]);
        assert_eq!(arena.free_blocks("state", 0).unwrap(), 0);
        drop(completed);
        assert_eq!(arena.free_blocks("state", 0).unwrap(), 3);
        client.shutdown(Duration::from_secs(2)).await.unwrap();
        directory.assert_graceful();
        directory.assert_stopped();
    }
}

#[tokio::test]
async fn bootstrap_binds_one_rank_zero_id_before_any_final_readiness() {
    for failure in [
        "none",
        "early_ready",
        "premature_ready",
        "wrong_nonce",
        "missing_nonce",
        "bootstrap_epoch",
        "bootstrap_rank",
        "bootstrap_version",
        "bootstrap_device",
        "bootstrap_missing_id",
        "bootstrap_extra_id",
        "bootstrap_short_id",
        "bootstrap_duplicate",
        "bootstrap_hang",
    ] {
        let directory = Directory::new();
        let mut settings = config(&directory, "dep4", failure);
        settings.transport = GroupTransport::Nccl;
        settings.startup_timeout = Duration::from_secs(1);
        let started = std::time::Instant::now();
        let launched = GroupClient::launch(settings).await;
        if failure == "none" {
            let mut client = launched.unwrap();
            let arena = state("dep4");
            let (plan, transactions) = step(&arena);
            drop(client.execute(plan, transactions).await.unwrap());
            client.close().await.unwrap();
        } else {
            assert!(launched.is_err(), "{failure}");
        }
        assert!(started.elapsed() < Duration::from_secs(3), "{failure}");
        directory.assert_stopped();
    }
}

#[tokio::test]
async fn graceful_shutdown_errors_and_cancel_poison_the_group_until_cleanup() {
    for failure in [
        "shutdown_stale",
        "shutdown_nonzero",
        "shutdown_hang",
        "shutdown_duplicate",
        "cancel",
    ] {
        let directory = Directory::new();
        let mode = if failure == "cancel" {
            "shutdown_hang"
        } else {
            failure
        };
        let mut client = GroupClient::launch(config(&directory, "dep4", mode))
            .await
            .unwrap();
        let started = std::time::Instant::now();
        if failure == "cancel" {
            assert!(
                tokio::time::timeout(
                    Duration::from_millis(50),
                    client.shutdown(Duration::from_secs(2))
                )
                .await
                .is_err()
            );
            let arena = state("dep4");
            let (plan, transactions) = step(&arena);
            assert!(client.execute(plan, transactions).await.is_err());
            assert_eq!(arena.free_blocks("state", 0).unwrap(), 3);
            assert!(client.shutdown(Duration::from_secs(1)).await.is_err());
        } else {
            assert!(
                client.shutdown(Duration::from_millis(300)).await.is_err(),
                "{failure}"
            );
        }
        client.close().await.unwrap();
        assert!(started.elapsed() < Duration::from_secs(3));
        directory.assert_stopped();
    }
}

#[tokio::test]
async fn group_failure_and_cancelled_wait_keep_leases_until_children_stop() {
    let mut unrelated = UnrelatedProcess(
        std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap(),
    );
    for failure in [
        "startup_exit",
        "bad_ready",
        "step_exit",
        "step_hang",
        "stale_result",
        "cancel",
        "empty_cancel",
        "drop",
    ] {
        let directory = Directory::new();
        let mode = if matches!(failure, "cancel" | "empty_cancel" | "drop") {
            "step_hang"
        } else {
            failure
        };
        let launched = GroupClient::launch(config(&directory, "dep4", mode)).await;
        if matches!(failure, "startup_exit" | "bad_ready") {
            assert!(launched.is_err(), "{failure}");
        } else {
            let mut client = launched.unwrap();
            let arena = state("dep4");
            let (mut plan, mut transactions) = step(&arena);
            let reserved_free = if failure == "empty_cancel" {
                plan.requests.clear();
                transactions.clear();
                3
            } else {
                0
            };
            if matches!(failure, "cancel" | "empty_cancel" | "drop") {
                assert!(
                    tokio::time::timeout(
                        Duration::from_millis(100),
                        client.execute(plan, transactions)
                    )
                    .await
                    .is_err()
                );
                assert_eq!(arena.free_blocks("state", 0).unwrap(), reserved_free);
                assert!(client.shutdown(Duration::from_millis(100)).await.is_err());
                assert!(
                    client.capacities().is_some(),
                    "shutdown changed a running group"
                );
                assert_eq!(arena.free_blocks("state", 0).unwrap(), reserved_free);
                if matches!(failure, "cancel" | "empty_cancel") {
                    client.close().await.unwrap();
                } else {
                    drop(client);
                    tokio::time::timeout(Duration::from_secs(3), async {
                        while arena.free_blocks("state", 0).unwrap() == 0
                            || std::fs::read_dir(&directory.0).unwrap().any(|entry| {
                                entry
                                    .unwrap()
                                    .path()
                                    .extension()
                                    .is_some_and(|ext| ext == "ipc")
                            })
                        {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }
                    })
                    .await
                    .unwrap();
                }
            } else {
                assert!(
                    client.execute(plan, transactions).await.is_err(),
                    "{failure}"
                );
            }
            assert_eq!(arena.free_blocks("state", 0).unwrap(), 3);
        }
        directory.assert_stopped();
        assert!(unrelated.0.try_wait().unwrap().is_none());
    }
}
