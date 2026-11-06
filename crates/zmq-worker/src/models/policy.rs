use std::fs::OpenOptions;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use oh_my_vllm_model_contract::ModelDescriptor;

use super::{ModelFamily, deepseek_v41, qwen38};
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
        }
    }

    pub fn vocabulary_size(self) -> u32 {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::vocabulary_size(),
            ModelFamily::Qwen38 => qwen38::policy::vocabulary_size(),
        }
    }

    pub fn max_offline_batch_size(self) -> usize {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::max_offline_batch_size(),
            ModelFamily::Qwen38 => qwen38::policy::max_offline_batch_size(),
        }
    }

    pub fn served_model_name(self) -> &'static str {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::served_model_name(),
            ModelFamily::Qwen38 => qwen38::policy::served_model_name(),
        }
    }

    pub fn resolve_speculation(self, mode: Option<&str>, drafts: usize) -> Result<SpeculativeMode> {
        match self.family {
            ModelFamily::DeepSeekV41 => deepseek_v41::policy::resolve_speculation(mode, drafts),
            ModelFamily::Qwen38 => qwen38::policy::resolve_speculation(mode, drafts),
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
        }
        Ok(self.family)
    }
}

pub fn select_model(path: &Path) -> Result<ModelPolicy> {
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
mod tests;
