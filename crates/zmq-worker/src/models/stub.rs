use oh_my_vllm_model_contract::{CacheKind, ModelDescriptor, WeightFormat};

use super::policy::{require_options, require_ready};
use super::qwen38::policy::{
    cache_group, validate_group, validate_runtime_options, validate_runtime_ready,
};
use crate::client::WorkerConfig;
use crate::error::Result;

// The literal test backend simulates the Qwen runtime and response contract.
pub(super) use super::qwen38::policy::{
    defaults, max_offline_batch_size, resolve_speculation, served_model_name, vocabulary_size,
};

const CONFIG_SHA256: &str = "3b7c7f7627a034cd753e4d4db30346a940e49b635952460c51b2ba5f4f332f18";
const CHECKPOINT_SHA256: &str = "6cafe580d6f80a355f3e30bf2c8a7559aef2b081845084d16e9eca733c09d7c2";
const TOKENIZER_SHA256: &str = "243e380a5f7db0cea0f70d4ddc53663d47e9f2b191b928fa5ccbba201ecf99b7";

pub(super) fn validate_options(config: &WorkerConfig) -> Result<()> {
    require_options(
        config.model_path.as_os_str() == "stub",
        "selected test stub policy requires the literal stub path",
    )?;
    validate_runtime_options(config)
}

pub(super) fn validate_ready(
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    logical_num_blocks: u32,
    mamba_blocks: u32,
) -> Result<()> {
    require_ready(
        config.model_path.as_os_str() == "stub",
        "selected test stub policy requires the literal stub path",
    )?;
    require_ready(
        descriptor.architecture == "OhMyVllmStubForTesting"
            && descriptor.model_type == "oh_my_vllm_stub"
            && descriptor.config_sha256 == CONFIG_SHA256
            && descriptor.checkpoint_sha256 == CHECKPOINT_SHA256
            && descriptor.tokenizer_sha256 == TOKENIZER_SHA256
            && descriptor.weight_formats == [WeightFormat::F32],
        "unsupported test stub identity or sentinel",
    )?;
    validate_runtime_ready(config, descriptor, logical_num_blocks, mamba_blocks)?;
    validate_group(
        cache_group(descriptor, "fa")?,
        CacheKind::ModelCheckpoint,
        "oh_my_vllm_stub_fa_metadata",
        &[0],
    )?;
    validate_group(
        cache_group(descriptor, "gdn")?,
        CacheKind::ModelCheckpoint,
        "oh_my_vllm_stub_gdn_metadata",
        &[0],
    )
}
