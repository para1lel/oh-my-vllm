use super::*;

#[tokio::test]
async fn rejected_ready_reaps_the_owned_process_group_and_removes_the_socket() {
    assert!(
        std::env::var_os("OH_MY_VLLM_WORKER_PYTHON").is_some(),
        "run CPU worker tests through scripts/with-env.sh"
    );
    for oversized in [false, true] {
        let config = rejection_config(oversized);
        let directory = config.model_path.clone();
        let socket = config.socket_path.clone();
        reject_configuration_before_spawn(&config, &directory, &socket).await;
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

const INVALID_READY_SOURCE: &str = r#"
import os
from pathlib import Path
import subprocess
import sys
import time
import msgpack
import zmq
from tests.test_models import sample_descriptor
address = sys.argv[sys.argv.index("--socket") + 1]
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt(zmq.LINGER, 0)
socket.connect(address)
init = msgpack.unpackb(socket.recv(), raw=False)
assert init["descriptor_version"] == 1
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
Path(address.removeprefix("ipc://") + ".pids").write_text(f"{os.getpid()} {child.pid}")
d = sample_descriptor(num_speculative_tokens=0)
d["schema_version"] = 2
reply = dict(type="ready", descriptor=d, logical_num_blocks=1608, mamba_blocks=128)
payload = msgpack.packb(reply, use_bin_type=True)
if OVERSIZED:
    payload = b"x" * ((1 << 20) + 1)
socket.send(payload)
time.sleep(300)
"#;

fn rejection_config(oversized: bool) -> WorkerConfig {
    use std::os::unix::fs::PermissionsExt;
    let directory = std::env::temp_dir().join(format!(
        "omv-ready-reject-{}-{oversized}",
        std::process::id(),
    ));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/qwen38/config.json"
        ))
        .strip_suffix(b"\n")
        .unwrap(),
    )
    .unwrap();
    let socket = directory.join("worker.ipc");
    let launcher = directory.join("worker.sh");
    let source = INVALID_READY_SOURCE.replace(
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
    WorkerConfig {
        socket_path: socket.clone(),
        num_gpu_blocks: Some(1608),
        mamba_blocks: Some(128),
        python_executable: launcher,
        init_timeout: Duration::from_secs(5),
        ..WorkerConfig::for_model(
            directory.clone(),
            ModelPolicy::for_family(ModelFamily::Qwen38),
        )
    }
}

async fn reject_configuration_before_spawn(
    config: &WorkerConfig,
    directory: &std::path::Path,
    socket: &std::path::Path,
) {
    for selected in [false, true] {
        let invalid = WorkerConfig {
            tensor_parallel_size: 4,
            ..config.clone()
        };
        let result = if selected {
            WorkerClient::launch_with_model(invalid, ModelPolicy::for_family(ModelFamily::Qwen38))
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
            ModelPolicy::for_family(ModelFamily::DeepSeekV41),
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
        model_path: directory.to_owned(),
        ..config.clone()
    };
    assert!(matches!(
        WorkerClient::launch(invalid).await,
        Err(Error::ModelConfiguration(_))
    ));
    assert!(!socket.exists());
    assert!(!socket.with_extension("ipc.pids").exists());
    std::fs::write(
        directory.join("config.json"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/qwen38/config.json"
        ))
        .strip_suffix(b"\n")
        .unwrap(),
    )
    .unwrap();
}
