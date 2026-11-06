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
