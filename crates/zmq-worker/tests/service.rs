use std::path::Path;
use std::process::Command;

use oh_my_vllm_zmq_worker::client::{WorkerClient, WorkerConfig};
use oh_my_vllm_zmq_worker::models::select_model;

struct WorkerFixture(std::path::PathBuf);

impl WorkerFixture {
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "omv-worker-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/qwen38/config.json",
            ))
            .strip_suffix(b"\n")
            .unwrap(),
        )
        .unwrap();
        let launcher = directory.join("worker.sh");
        std::fs::write(
            &launcher,
            "#!/bin/sh\nexec \"$OH_MY_VLLM_WORKER_PYTHON\" -m tests.worker_fixture \"$@\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(directory)
    }

    fn config(&self) -> WorkerConfig {
        WorkerConfig {
            python_executable: self.0.join("worker.sh"),
            ..WorkerConfig::for_model(self.0.clone(), select_model(&self.0).unwrap())
        }
    }
}

impl Drop for WorkerFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[tokio::test]
async fn request_swaps_with_fixed_worker_replies_preserve_all_output() {
    use oh_my_vllm_kv_cache::HybridCoordinator;
    use oh_my_vllm_scheduler::{Request, Scheduler, SchedulerConfig};

    let fixture = WorkerFixture::new();
    let config = WorkerConfig {
        socket_path: std::env::temp_dir().join(format!("omv-swap-{}.ipc", std::process::id())),
        num_gpu_blocks: Some(5),
        mamba_blocks: Some(16),
        host_cache_bytes: Some(4 << 30),
        max_num_seqs: 2,
        max_num_batched_tokens: 4096,
        init_timeout: std::time::Duration::from_secs(3),
        ..fixture.config()
    };
    let mut client = WorkerClient::launch(config).await.unwrap();
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_seqs: 2,
            max_num_batched_tokens: 4096,
            ..Default::default()
        },
        HybridCoordinator::new(5, 784, true, 0).with_mamba_capacity(16),
    );
    scheduler.enable_host_cache(client.host_cache_bytes);
    for id in 1..=2 {
        let tokens = (0..1567)
            .map(|token| token + id as u32 * 100)
            .collect::<Vec<_>>();
        let mut request = Request::new(id, tokens.clone(), 8, vec![]);
        request.record_partial_prefix = false;
        assert!(scheduler.add_request(request));
        client.register_request_with_limit(id, tokens, 8).await;
    }
    let mut counts = [0; 2];
    for _ in 0..100 {
        client.cache_step(&mut scheduler).await.unwrap();
        let step = scheduler.schedule();
        client.cache_step(&mut scheduler).await.unwrap();
        assert!(scheduler.take_cache_failures().is_empty());
        let result = client
            .execute_one_step(
                &step.scheduled,
                &step.finished_request_ids,
                &step.preempted_request_ids,
                step.num_batched_tokens,
            )
            .await
            .unwrap();
        if !step.scheduled.is_empty() {
            let result = scheduler.update(result).unwrap();
            for output in result.outputs {
                counts[output.request_id as usize - 1] += output.token_ids.len();
            }
        }
        if scheduler.num_running() + scheduler.num_waiting() == 0 && !scheduler.cache_pending() {
            break;
        }
    }
    assert_eq!(counts, [8, 8]);
    let stats = client.query_stats().await.unwrap();
    assert!(stats["host_cache_store_bytes"].as_u64().unwrap() > 0);
    assert!(stats["host_cache_load_bytes"].as_u64().unwrap() > 0);
    client.shutdown().await.unwrap();
}

#[test]
fn service_contracts_with_fixed_worker_replies() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let python = std::env::var("OH_MY_VLLM_WORKER_PYTHON").unwrap_or("python3".to_owned());
    let status = Command::new(python)
        .args([
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_service.py",
        ])
        .current_dir(root)
        .env(
            "OH_MY_VLLM_TEST_BINARY",
            env!("CARGO_BIN_EXE_oh-my-vllm-zmq-worker"),
        )
        .status()
        .unwrap();
    assert!(status.success());
}

#[tokio::test]
async fn worker_shutdown_releases_its_socket_before_return() {
    let name = format!(
        "oh-my-vllm-lifecycle-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    );
    let directory = std::env::temp_dir().join(name);
    std::fs::create_dir(&directory).unwrap();
    let socket = directory.join("worker.ipc");
    let fixture = WorkerFixture::new();
    let config = WorkerConfig {
        socket_path: socket.clone(),
        init_timeout: std::time::Duration::from_secs(3),
        ..fixture.config()
    };
    for _ in 0..2 {
        let mut worker = WorkerClient::launch(config.clone()).await.unwrap();
        assert!(
            worker.query_stats().await.unwrap()["process_id"]
                .as_u64()
                .unwrap()
                > 0
        );
        worker.shutdown().await.unwrap();
        drop(worker);
        assert!(!socket.exists());
    }
    std::fs::remove_dir(directory).unwrap();
}
