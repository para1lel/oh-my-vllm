use std::fs::OpenOptions;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use oh_my_vllm_model_contract::ModelDescriptor;

use super::{ModelFamily, deepseek_v41, qwen38, stub};
use crate::client::WorkerConfig;
use crate::error::{Error, Result};

const MAX_CONFIG_BYTES: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDefaults {
    pub block_size: u32,
    pub tensor_parallel_size: u32,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub max_num_seqs: u32,
    pub mamba_blocks: u32,
    pub init_timeout_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeMode {
    pub name: &'static str,
    pub num_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPolicy {
    family: ModelFamily,
}

impl ModelPolicy {
    pub fn defaults(self) -> ModelDefaults {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::defaults(),
            ModelFamily::Qwen38 => qwen38::policy::defaults(),
            ModelFamily::Stub => stub::defaults(),
        }
    }

    pub fn vocabulary_size(self) -> u32 {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::vocabulary_size(),
            ModelFamily::Qwen38 => qwen38::policy::vocabulary_size(),
            ModelFamily::Stub => stub::vocabulary_size(),
        }
    }

    pub fn max_offline_batch_size(self) -> usize {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::max_offline_batch_size(),
            ModelFamily::Qwen38 => qwen38::policy::max_offline_batch_size(),
            ModelFamily::Stub => stub::max_offline_batch_size(),
        }
    }

    pub fn served_model_name(self) -> &'static str {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::served_model_name(),
            ModelFamily::Qwen38 => qwen38::policy::served_model_name(),
            ModelFamily::Stub => stub::served_model_name(),
        }
    }

    pub fn resolve_speculation(self, mode: Option<&str>, drafts: usize) -> Result<SpeculativeMode> {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::resolve_speculation(mode, drafts),
            ModelFamily::Qwen38 => qwen38::policy::resolve_speculation(mode, drafts),
            ModelFamily::Stub => stub::resolve_speculation(mode, drafts),
        }
    }

    pub fn validate_options(self, config: &WorkerConfig) -> Result<()> {
        require_options(
            !config.model_path.as_os_str().is_empty(),
            "set OH_MY_VLLM_MODEL or WorkerConfig.model_path",
        )?;
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::validate_options(config),
            ModelFamily::Qwen38 => qwen38::policy::validate_options(config),
            ModelFamily::Stub => stub::validate_options(config),
        }
    }

    pub(crate) fn for_family(family: ModelFamily) -> Self {
        Self { family }
    }

    pub(crate) fn validate_ready(
        self,
        config: &WorkerConfig,
        descriptor: &ModelDescriptor,
        logical_num_blocks: u32,
        mamba_blocks: u32,
    ) -> Result<ModelFamily> {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::validate_ready(
                config,
                descriptor,
                logical_num_blocks,
                mamba_blocks,
            )?,
            ModelFamily::Qwen38 => qwen38::policy::validate_ready(
                config,
                descriptor,
                logical_num_blocks,
                mamba_blocks,
            )?,
            ModelFamily::Stub => {
                stub::validate_ready(config, descriptor, logical_num_blocks, mamba_blocks)?
            }
        }
        Ok(self.family)
    }
}

pub fn select_model(path: &Path) -> Result<ModelPolicy> {
    if path.as_os_str() == "stub" {
        return Ok(ModelPolicy::for_family(ModelFamily::Stub));
    }
    require_options(!path.as_os_str().is_empty(), "model path must not be empty")?;
    let raw = read_config(path)?;
    let config: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|error| Error::ModelConfiguration(format!("cannot parse config.json: {error}")))?;
    require_options(config.is_object(), "model configuration must be an object")?;
    if qwen38::policy::matches_config(&config) {
        qwen38::policy::validate_config(&config, &raw)?;
        return Ok(ModelPolicy::for_family(ModelFamily::Qwen38));
    }
    if deepseek_v41::policy::matches_config(&config) {
        deepseek_v41::policy::validate_config(&config, &raw)?;
        return Ok(ModelPolicy::for_family(ModelFamily::DeepSeekV41));
    }
    Err(Error::ModelConfiguration(
        "unsupported model architecture or model type".into(),
    ))
}

fn read_config(path: &Path) -> Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    // A FIFO must not block before the regular-file check.
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK);
    let mut source = options.open(path.join("config.json")).map_err(|error| {
        Error::ModelConfiguration(format!("cannot open model config.json: {error}"))
    })?;
    let metadata = source.metadata().map_err(|error| {
        Error::ModelConfiguration(format!("cannot inspect model config.json: {error}"))
    })?;
    require_options(
        metadata.is_file(),
        "model config.json must be a regular file",
    )?;
    require_options(
        metadata.len() <= MAX_CONFIG_BYTES,
        "model configuration exceeds its byte limit",
    )?;
    let mut raw = Vec::new();
    (&mut source)
        .take(MAX_CONFIG_BYTES)
        .read_to_end(&mut raw)
        .map_err(|error| {
            Error::ModelConfiguration(format!("cannot read model config.json: {error}"))
        })?;
    let final_size = source
        .metadata()
        .map_err(|error| {
            Error::ModelConfiguration(format!("cannot inspect model config.json: {error}"))
        })?
        .len();
    require_options(
        final_size <= MAX_CONFIG_BYTES,
        "model configuration exceeds its byte limit",
    )?;
    require_options(
        metadata.len() == raw.len() as u64 && final_size == raw.len() as u64,
        "model configuration size changed during read",
    )?;
    Ok(raw)
}

pub(super) fn require_options(valid: bool, reason: &str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(Error::ModelConfiguration(reason.into()))
    }
}

pub(super) fn require_ready(valid: bool, reason: &str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(Error::WorkerError(format!(
            "invalid model readiness: {reason}"
        )))
    }
}

#[cfg(test)]
mod tests {
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
            (json!(["OhMyVllmStubForTesting"]), "oh_my_vllm_stub"),
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

    #[test]
    fn only_literal_stub_selects_the_metadata_simulation() {
        let policy = select_model(Path::new("stub")).unwrap();
        assert_eq!(policy.vocabulary_size(), 248320);
        assert_eq!(policy.max_offline_batch_size(), 32);
        assert_eq!(policy.served_model_name(), "qwen3.8-27b-fp8");
        for path in ["", "stub/", "stub/.", "./stub", "/stub", "unknown"] {
            assert!(select_model(Path::new(path)).is_err(), "{path}");
            assert!(
                policy
                    .validate_options(&WorkerConfig::for_model(path.into(), policy))
                    .is_err(),
                "{path}"
            );
        }
        let qwen = ModelPolicy::for_family(ModelFamily::Qwen38);
        for path in ["stub", ""] {
            assert!(
                qwen.validate_options(&WorkerConfig::for_model(path.into(), qwen))
                    .is_err(),
                "{path}"
            );
        }
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
        for (path, policy) in [
            (
                "/synthetic-qwen",
                ModelPolicy::for_family(ModelFamily::Qwen38),
            ),
            ("stub", select_model(Path::new("stub")).unwrap()),
        ] {
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
                (784, 1, 262144, 32768, 4, Some(128))
            );
            assert_eq!(
                (config.num_gpu_blocks, config.host_cache_bytes),
                (None, None)
            );
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
        }
        let policy = select_model(Path::new("stub")).unwrap();
        let config = WorkerConfig::for_model("stub".into(), policy);
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
                    max_num_batched_tokens: 65536,
                    num_speculative_tokens: 4,
                    speculative_mode: Some("mtp".into()),
                    ..config
                })
                .is_ok()
        );
    }
}
