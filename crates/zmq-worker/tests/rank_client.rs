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

#[path = "support/model_policy.rs"]
mod model_policy;

fn model_settings(directory: &Directory, strategy: &str, failure: &str) -> GroupConfig {
    let mut settings = config(directory, strategy, failure);
    let group = settings.descriptor.cache_schema.groups[0].clone();
    settings.descriptor.cache_schema.groups = ["latent", "window", "history"]
        .map(|name| {
            let mut group = group.clone();
            group.name = name.into();
            group.capacity_blocks = 8;
            group
        })
        .to_vec();
    settings.limits.max_slots = 96;
    settings.step_timeout = Duration::from_secs(2);
    settings.descriptor.speculation = serde_json::from_value(json!([{
        "method": "dspark", "predictor_stages": 1,
        "max_draft_tokens": 5, "max_verification_rows": 6
    }]))
    .unwrap();
    settings
}

fn model_scheduler(
    settings: &GroupConfig,
) -> oh_my_vllm_scheduler::Scheduler<model_policy::ArenaPolicy> {
    use oh_my_vllm_scheduler::{Request, Scheduler, SchedulerConfig};

    let mut scheduler = Scheduler::new(
        SchedulerConfig::default(),
        model_policy::ArenaPolicy::new(settings.descriptor.clone(), settings.epoch),
    );
    assert!(scheduler.add_request(Request::new(4, vec![1, 2], 3, vec![])));
    assert!(scheduler.add_request(Request::new(5, vec![4], 3, vec![])));
    scheduler
}

fn model_plan(
    batch: &oh_my_vllm_scheduler::SchedulerOutput<
        oh_my_vllm_zmq_worker::group::model::ModelState<model_policy::Layout>,
    >,
    strategy: &str,
) -> StepPlan {
    StepPlan {
        step_id: batch.scheduled[0].state.transaction.step_id,
        collective_plan: serde_json::from_value(json!({
            "version":1,"world_size":4,"operations":[]
        }))
        .unwrap(),
        requests: batch
            .scheduled
            .iter()
            .map(|request| RequestWork {
                request_id: request.request_id,
                transaction_id: request.state.transaction.transaction_id,
                owner_rank: if strategy == "dep4" {
                    (request.request_id % 4) as u32
                } else {
                    0
                },
                rows: request.token_ids.len() as u32,
                prefill: request.num_computed_tokens == 0,
            })
            .collect(),
    }
}

#[tokio::test]
async fn typed_model_execution_commits_only_after_quorum_and_full_output_validation() {
    for strategy in ["tp4", "tep4", "dep4"] {
        let directory = Directory::new();
        let settings = model_settings(&directory, strategy, "model_slow");
        let policy = model_policy::ArenaPolicy::new(settings.descriptor.clone(), settings.epoch);
        let commits = policy.commits.clone();
        let mut scheduler = oh_my_vllm_scheduler::Scheduler::new(Default::default(), policy);
        for (id, prompt) in [(4, vec![1, 2]), (5, vec![4])] {
            assert!(scheduler.add_request(oh_my_vllm_scheduler::Request::new(
                id,
                prompt,
                3,
                vec![]
            )));
        }
        let mut client = GroupClient::launch(settings).await.unwrap();
        for step in [1, 2] {
            let batch = scheduler.schedule();
            let plan = model_plan(&batch, strategy);
            assert_eq!(plan.step_id, step);
            let mut execution = Box::pin(client.execute_scheduled(&mut scheduler, batch, plan));
            assert!(
                tokio::time::timeout(Duration::from_millis(30), &mut execution)
                    .await
                    .is_err()
            );
            assert_eq!(commits.load(Ordering::SeqCst), (step as usize - 1) * 2);
            let output = execution.await.unwrap();
            assert_eq!(
                output
                    .outputs
                    .iter()
                    .map(|row| row.token_ids.clone())
                    .collect::<Vec<_>>(),
                [vec![3], vec![4]]
            );
            assert_eq!(commits.load(Ordering::SeqCst), step as usize * 2);
            assert!(!scheduler.has_detached_state());
            for rank in 0..4 {
                let record: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(directory.0.join(format!("model-{rank}-{step}.json"))).unwrap(),
                )
                .unwrap();
                assert_eq!(record["payload"]["requests"].as_array().unwrap().len(), 2);
            }
        }
        scheduler.abort(4);
        scheduler.abort(5);
        assert_eq!(scheduler.cache_resource_counts(), 84);
        client.shutdown(Duration::from_secs(2)).await.unwrap();
        directory.assert_graceful();
        directory.assert_stopped();
    }
}

#[tokio::test]
async fn typed_model_input_and_result_errors_cannot_publish_state() {
    for failure in [
        "model_digest",
        "model_missing",
        "model_duplicate",
        "model_foreign",
        "model_token",
        "model_accept",
        "tokens",
        "generation",
        "duplicate_write",
        "unowned_read",
        "transaction",
        "prefill",
    ] {
        let directory = Directory::new();
        let settings = model_settings(&directory, "tp4", failure);
        let policy = model_policy::ArenaPolicy::new(settings.descriptor.clone(), settings.epoch);
        let commits = policy.commits.clone();
        let mut scheduler = oh_my_vllm_scheduler::Scheduler::new(Default::default(), policy);
        assert!(scheduler.add_request(oh_my_vllm_scheduler::Request::new(
            4,
            vec![1, 2],
            3,
            vec![]
        )));
        assert!(scheduler.add_request(oh_my_vllm_scheduler::Request::new(5, vec![4], 3, vec![])));
        let mut client = GroupClient::launch(settings).await.unwrap();
        let mut batch = scheduler.schedule();
        let mut plan = model_plan(&batch, "tp4");
        let first = &mut batch.scheduled[0];
        match failure {
            "tokens" => first.token_ids[0] += 1,
            "generation" => first.state.writes[0].generation += 1,
            "duplicate_write" => first.state.writes.push(first.state.writes[0].clone()),
            "unowned_read" => {
                let mut unowned = first.state.writes[0].clone();
                unowned.slot = 0;
                first.state.reads.push(unowned);
            }
            "transaction" => first.state.transaction.transaction_id += 1,
            "prefill" => plan.requests[0].prefill = false,
            _ => {}
        }
        let error = client
            .execute_scheduled(&mut scheduler, batch, plan)
            .await
            .expect_err(failure);
        if failure == "model_accept" {
            assert!(
                error.to_string().contains("accepted unscheduled drafts"),
                "{error}"
            );
        }
        assert_eq!(commits.load(Ordering::SeqCst), 0, "{failure}");
        assert!(scheduler.has_detached_state());
        assert!(scheduler.schedule().scheduled.is_empty());
        assert!(
            scheduler
                .update(oh_my_vllm_scheduler::WorkerOutput { outputs: vec![] })
                .is_err()
        );
        scheduler.abort(4);
        scheduler.abort(5);
        assert_eq!(scheduler.cache_resource_counts(), 84, "{failure}");
        directory.assert_stopped();
    }
}

#[tokio::test]
async fn cancelled_typed_execution_keeps_detached_owners_until_process_reaping() {
    let directory = Directory::new();
    let settings = model_settings(&directory, "dep4", "model_hang");
    let mut scheduler = model_scheduler(&settings);
    let mut client = GroupClient::launch(settings).await.unwrap();
    let batch = scheduler.schedule();
    let plan = model_plan(&batch, "dep4");
    assert_eq!(scheduler.cache_resource_counts(), 78);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(30),
            client.execute_scheduled(&mut scheduler, batch, plan)
        )
        .await
        .is_err()
    );
    assert!(scheduler.has_detached_state());
    scheduler.abort(4);
    scheduler.abort(5);
    assert_eq!(scheduler.cache_resource_counts(), 78);
    client.close().await.unwrap();
    assert_eq!(scheduler.cache_resource_counts(), 84);
    assert!(!scheduler.empty_like().has_detached_state());
    directory.assert_stopped();
}
