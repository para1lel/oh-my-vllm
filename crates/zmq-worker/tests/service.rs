use std::path::Path;
use std::process::Command;

use oh_my_vllm_zmq_worker::client::{WorkerClient, WorkerConfig};

#[test]
fn service_contracts_with_the_python_placeholder() {
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
    let config = WorkerConfig {
        model_path: "stub".into(),
        socket_path: socket.clone(),
        init_timeout: std::time::Duration::from_secs(3),
        ..Default::default()
    };
    for _ in 0..2 {
        let mut worker = WorkerClient::launch(config.clone()).await.unwrap();
        assert_eq!(worker.query_stats().await.unwrap()["placeholder"], true);
        worker.shutdown().await.unwrap();
        drop(worker);
        assert!(!socket.exists());
    }
    std::fs::remove_dir(directory).unwrap();
}
