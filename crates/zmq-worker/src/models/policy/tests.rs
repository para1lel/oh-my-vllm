use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use super::{ModelFamily, ModelPolicy, select_model};
use crate::client::WorkerConfig;
use crate::error::Error;

struct ModelDirectory(PathBuf);

impl ModelDirectory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oh-my-vllm-policy-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, bytes: &[u8]) {
        std::fs::write(self.0.join("config.json"), bytes).unwrap();
    }
}

impl Drop for ModelDirectory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn selection_rejects_unknown_malformed_oversized_and_changed_configs() {
    let directory = ModelDirectory::new();
    assert!(select_model(&directory.0).is_err());
    for bytes in [b"{".as_slice(), b"", b"\xff", b"null", b"[]"] {
        directory.write(bytes);
        assert!(select_model(&directory.0).is_err(), "{bytes:?}");
    }
    let mut config = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {
            "quant_method": "fp8", "activation_scheme": "dynamic",
            "fmt": "e4m3", "weight_block_size": [128, 128],
        },
    });
    for (architectures, model_type) in [
        (json!(["OtherForCausalLM"]), "qwen3_5"),
        (json!(["Qwen3_5ForConditionalGeneration"]), "other_model"),
        (json!(["OtherForCausalLM"]), "other_model"),
        (
            json!(["Qwen3_5ForConditionalGeneration", "other"]),
            "qwen3_5",
        ),
        (json!("Qwen3_5ForConditionalGeneration"), "qwen3_5"),
    ] {
        config["architectures"] = architectures;
        config["model_type"] = json!(model_type);
        directory.write(&serde_json::to_vec(&config).unwrap());
        let error = select_model(&directory.0).unwrap_err();
        assert!(error.to_string().contains("architecture"), "{error}");
    }
    config["architectures"] = json!(["Qwen3_5ForConditionalGeneration"]);
    config["model_type"] = json!("qwen3_5");
    directory.write(&serde_json::to_vec(&config).unwrap());
    let error = select_model(&directory.0).unwrap_err();
    assert!(error.to_string().contains("identity"), "{error}");
    for (field, value) in [
        ("quant_method", json!("int8")),
        ("activation_scheme", json!("static")),
        ("fmt", json!("e5m2")),
        ("weight_block_size", json!([32, 32])),
    ] {
        let mut changed = config.clone();
        changed["quantization_config"][field] = value;
        directory.write(&serde_json::to_vec(&changed).unwrap());
        let error = select_model(&directory.0).unwrap_err();
        assert!(error.to_string().contains("format"), "{error}");
    }
    for bytes in [vec![b' '; 1 << 20], vec![b' '; (1 << 20) + 1]] {
        directory.write(&bytes);
        let error = select_model(&directory.0).unwrap_err();
        if bytes.len() > 1 << 20 {
            assert!(error.to_string().contains("byte limit"), "{error}");
        } else {
            assert!(!error.to_string().contains("byte limit"), "{error}");
        }
    }
    #[cfg(unix)]
    reject_fifo_configuration(&directory);
}

#[test]
fn selection_requires_a_model_configuration() {
    for path in ["", "stub", "unknown"] {
        assert!(select_model(Path::new(path)).is_err(), "{path}");
    }
    let qwen = ModelPolicy::for_family(ModelFamily::Qwen38);
    assert!(
        qwen.validate_options(&WorkerConfig::for_model("".into(), qwen))
            .is_err()
    );
}

#[test]
fn selects_deepseek_and_exposes_dspark5_defaults() {
    let directory = ModelDirectory::new();
    directory.write(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/deepseek_v41/config.json"
    )));
    let policy = select_model(&directory.0).unwrap();
    assert_eq!(policy.vocabulary_size(), 129280);
    assert_eq!(policy.max_offline_batch_size(), 64);
    assert_eq!(policy.served_model_name(), "deepseek-v4.1-flash");
    let defaults = policy.defaults();
    assert_eq!(defaults.init_timeout_seconds, 1800);
    assert_eq!(
        (
            defaults.block_size,
            defaults.tensor_parallel_size,
            defaults.max_model_len,
            defaults.max_num_batched_tokens,
            defaults.max_num_seqs,
            defaults.mamba_blocks,
        ),
        (128, 4, 1048576, 32768, 64, 128)
    );
    for (mode, drafts, expected) in [
        (None, 0, ("none", 0)),
        (Some("none"), 0, ("none", 0)),
        (Some("dspark5"), 5, ("dspark5", 5)),
        (None, 5, ("dspark5", 5)),
    ] {
        let resolved = policy.resolve_speculation(mode, drafts).unwrap();
        assert_eq!((resolved.name, resolved.num_tokens), expected);
    }
}

#[test]
fn selected_configuration_preserves_defaults_modes_and_rejections() {
    let path = "/synthetic-qwen";
    let policy = ModelPolicy::for_family(ModelFamily::Qwen38);
    let config = WorkerConfig::for_model(path.into(), policy);
    assert_eq!(config.model_path, Path::new(path));
    assert_eq!(
        (
            config.block_size,
            config.tensor_parallel_size,
            config.max_model_len,
            config.max_num_batched_tokens,
            config.max_num_seqs,
            config.mamba_blocks,
        ),
        (128, 1, 262144, 32768, 4, Some(128))
    );
    assert_eq!((config.num_gpu_blocks, config.worker_epoch), (None, 1));
    assert_eq!(
        (
            config.num_speculative_tokens,
            config.speculative_mode.as_deref()
        ),
        (0, None)
    );
    assert!(policy.validate_options(&config).is_ok());
    for (mode, drafts, expected_name, expected_count) in [
        (None, 0, "none", 0),
        (None, 4, "mtp", 4),
        (Some("none"), 0, "none", 0),
        (Some("mtp"), 0, "mtp", 4),
        (Some("mtp"), 4, "mtp", 4),
    ] {
        let resolved = policy.resolve_speculation(mode, drafts).unwrap();
        assert_eq!(
            (resolved.name, resolved.num_tokens),
            (expected_name, expected_count)
        );
    }
    for (mode, drafts) in [(None, 1), (None, 5), (Some("none"), 4), (Some("mtp"), 1)] {
        assert!(
            policy.resolve_speculation(mode, drafts).is_err(),
            "{mode:?}/{drafts}"
        );
    }
    reject_invalid_options(path, policy);
}

#[cfg(unix)]
fn reject_fifo_configuration(directory: &ModelDirectory) {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::sync::mpsc::{RecvTimeoutError, channel};
        use std::time::Duration;

        let fifo_path = directory.0.join("config.json");
        std::fs::remove_file(&fifo_path).unwrap();
        let fifo_name = CString::new(fifo_path.as_os_str().as_bytes()).unwrap();
        // SAFETY: fifo_name is a live, NUL-terminated path in the owned test directory.
        let created = unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) };
        assert_eq!(created, 0, "{}", std::io::Error::last_os_error());
        let model_path = directory.0.clone();
        let (sender, receiver) = channel();
        let selection = std::thread::spawn(move || {
            let _ = sender.send(select_model(&model_path));
        });
        let response = receiver.recv_timeout(Duration::from_millis(500));
        let _unblock = if matches!(response.as_ref(), Err(RecvTimeoutError::Timeout)) {
            Some(
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&fifo_path)
                    .unwrap(),
            )
        } else {
            None
        };
        selection.join().unwrap();
        let result = response.expect("FIFO selection must finish before the CPU deadline");
        assert!(
            matches!(result, Err(Error::ModelConfiguration(_))),
            "{result:?}"
        );
    }
}

fn reject_invalid_options(path: &str, policy: ModelPolicy) {
    let config = WorkerConfig::for_model(path.into(), policy);
    for invalid in [
        WorkerConfig {
            block_size: 785,
            ..config.clone()
        },
        WorkerConfig {
            tensor_parallel_size: 4,
            ..config.clone()
        },
        WorkerConfig {
            max_model_len: 0,
            ..config.clone()
        },
        WorkerConfig {
            max_model_len: 262145,
            ..config.clone()
        },
        WorkerConfig {
            max_num_batched_tokens: 0,
            ..config.clone()
        },
        WorkerConfig {
            max_num_batched_tokens: 32769,
            ..config.clone()
        },
        WorkerConfig {
            max_num_seqs: 0,
            ..config.clone()
        },
        WorkerConfig {
            num_gpu_blocks: Some(1),
            ..config.clone()
        },
        WorkerConfig {
            mamba_blocks: Some(1),
            ..config.clone()
        },
        WorkerConfig {
            mamba_blocks: Some(65537),
            ..config.clone()
        },
        WorkerConfig {
            num_speculative_tokens: 1,
            ..config.clone()
        },
        WorkerConfig {
            speculative_mode: Some("mtp".into()),
            ..config.clone()
        },
        WorkerConfig {
            speculative_mode: Some("unknown".into()),
            ..config.clone()
        },
    ] {
        assert!(
            matches!(
                policy.validate_options(&invalid),
                Err(Error::ModelConfiguration(_))
            ),
            "{invalid:?}"
        );
    }
    assert!(
        policy
            .validate_options(&WorkerConfig {
                num_gpu_blocks: None,
                mamba_blocks: None,
                max_num_seqs: 32,
                max_num_batched_tokens: 32768,
                num_speculative_tokens: 4,
                speculative_mode: Some("mtp".into()),
                ..config
            })
            .is_ok()
    );
}
