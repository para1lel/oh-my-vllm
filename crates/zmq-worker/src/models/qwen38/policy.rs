use std::collections::BTreeSet;

use oh_my_vllm_model_contract::{
    CacheGroup, CacheKind, CacheSource, CacheStorage, Capability, ModelDescriptor,
    ParallelStrategy, SpeculationMethod, SpeculationSpec, WeightFormat,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::super::policy::{ModelDefaults, SpeculativeMode, require_options, require_ready};
use crate::client::WorkerConfig;
use crate::error::{Error, Result};

const CONFIG_SHA256: &str = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc";
const ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const MODEL_TYPE: &str = "qwen3_5";
const CAPABILITIES: [Capability; 8] = [
    Capability::Text,
    Capability::Images,
    Capability::Video,
    Capability::Tools,
    Capability::Reasoning,
    Capability::StructuredOutputs,
    Capability::PrefixCache,
    Capability::HostOffload,
];

pub(in crate::models) fn defaults() -> ModelDefaults {
    ModelDefaults {
        block_size: 784,
        tensor_parallel_size: 1,
        max_model_len: 262144,
        max_num_batched_tokens: 32768,
        max_num_seqs: 4,
        mamba_blocks: 128,
    }
}

pub(in crate::models) fn vocabulary_size() -> u32 {
    248320
}

pub(in crate::models) fn max_offline_batch_size() -> usize {
    32
}

pub(in crate::models) fn served_model_name() -> &'static str {
    "qwen3.8-27b-fp8"
}

pub(in crate::models) fn matches_config(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some(MODEL_TYPE)
        && config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|values| values.len() == 1 && values[0].as_str() == Some(ARCHITECTURE))
}

pub(in crate::models) fn validate_config(config: &Value, raw: &[u8]) -> Result<()> {
    let quantization = &config["quantization_config"];
    require_options(
        quantization.is_object()
            && quantization["quant_method"] == "fp8"
            && quantization["activation_scheme"] == "dynamic"
            && quantization["fmt"] == "e4m3"
            && quantization["weight_block_size"] == serde_json::json!([128, 128]),
        "unsupported Qwen weight or activation format",
    )?;
    require_options(
        format!("{:x}", Sha256::digest(raw)) == CONFIG_SHA256,
        "Qwen configuration identity differs from the model contract",
    )
}

pub(in crate::models) fn resolve_speculation(
    mode: Option<&str>,
    drafts: usize,
) -> Result<SpeculativeMode> {
    require_options(
        matches!(drafts, 0 | 4),
        "only ordinary decode and four-draft MTP are implemented",
    )?;
    let name = mode.unwrap_or(if drafts == 4 {
        "mtp"
    } else {
        "none"
    });
    let num_tokens = match name {
        "none" => 0,
        "mtp" => 4,
        _ => {
            return Err(Error::ModelConfiguration(
                "unsupported Qwen speculative mode".into(),
            ));
        }
    };
    require_options(
        drafts == 0 || drafts == num_tokens,
        "speculative mode disagrees with the requested draft count",
    )?;
    Ok(SpeculativeMode {
        name: if num_tokens == 4 {
            "mtp"
        } else {
            "none"
        },
        num_tokens,
    })
}

pub(in crate::models) fn validate_options(config: &WorkerConfig) -> Result<()> {
    require_options(
        config.model_path.as_os_str() != "stub",
        "selected Qwen policy requires a checkpoint path",
    )?;
    validate_runtime_options(config)
}

pub(in crate::models) fn validate_runtime_options(config: &WorkerConfig) -> Result<()> {
    require_options(config.tensor_parallel_size == 1, "only TP1 is implemented")?;
    require_options(
        config.block_size == defaults().block_size,
        "only 784-token cache blocks are implemented",
    )?;
    require_options(
        (1..=defaults().max_model_len).contains(&config.max_model_len),
        "requested context exceeds supported limits",
    )?;
    require_options(
        config.max_num_batched_tokens > 0 && config.max_num_seqs > 0,
        "token and sequence budgets must be positive",
    )?;
    require_options(
        config.num_gpu_blocks.is_none_or(|count| count > 1)
            && config
                .mamba_blocks
                .is_none_or(|count| (2..=65536).contains(&count)),
        "invalid worker cache capacities",
    )?;
    let mode = resolve_speculation(
        config.speculative_mode.as_deref(),
        config.num_speculative_tokens,
    )?;
    require_options(
        mode.num_tokens == config.num_speculative_tokens,
        "speculative mode disagrees with the requested draft count",
    )
}

pub(in crate::models) fn validate_ready(
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    logical_num_blocks: u32,
    mamba_blocks: u32,
) -> Result<()> {
    require_ready(
        config.model_path.as_os_str() != "stub",
        "selected Qwen policy requires a checkpoint path",
    )?;
    require_ready(
        descriptor.architecture == ARCHITECTURE
            && descriptor.model_type == MODEL_TYPE
            && descriptor.config_sha256 == CONFIG_SHA256,
        "unsupported Qwen model identity",
    )?;
    require_ready(
        same_members(
            &descriptor.weight_formats,
            &[
                WeightFormat::Bf16,
                WeightFormat::F32,
                WeightFormat::QwenFp8E4m3Block128x128Fp32Scales,
            ],
        ),
        "unsupported Qwen weight formats",
    )?;
    validate_runtime_ready(config, descriptor, logical_num_blocks, mamba_blocks)?;
    let mut fa_layers: Vec<_> = (3..64).step_by(4).collect();
    if config.num_speculative_tokens == 4 {
        fa_layers.push(64);
    }
    validate_group(
        cache_group(descriptor, "fa")?,
        CacheKind::FullAttention,
        "qwen38_paired_fa_kv_bf16",
        &fa_layers,
    )?;
    validate_group(
        cache_group(descriptor, "gdn")?,
        CacheKind::Recurrent,
        "qwen38_paired_gdn_f32_conv_bf16_hidden_bf16",
        &(0..64).filter(|layer| layer % 4 != 3).collect::<Vec<_>>(),
    )
}

pub(in crate::models) fn validate_runtime_ready(
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    logical_num_blocks: u32,
    mamba_blocks: u32,
) -> Result<()> {
    validate_runtime_options(config)
        .map_err(|error| Error::WorkerError(format!("invalid model readiness: {error}")))?;
    require_ready(
        descriptor.parallel_plan.strategy == ParallelStrategy::Tp1,
        "only TP1 is implemented",
    )?;
    require_ready(
        descriptor.vocabulary_size == vocabulary_size()
            && descriptor.max_context_tokens == u64::from(defaults().max_model_len),
        "unsupported vocabulary or context ceiling",
    )?;
    require_ready(
        logical_num_blocks > 1
            && (2..=65536).contains(&mamba_blocks)
            && config
                .num_gpu_blocks
                .is_none_or(|count| count == logical_num_blocks)
            && config
                .mamba_blocks
                .is_none_or(|count| count == mamba_blocks),
        "invalid worker cache capacities",
    )?;
    require_ready(
        same_members(&descriptor.capabilities, &CAPABILITIES),
        "unsupported capability set",
    )?;
    require_ready(
        descriptor.speculation
            == [SpeculationSpec {
                method: SpeculationMethod::Mtp,
                predictor_stages: 1,
                max_draft_tokens: 4,
                max_verification_rows: 5,
            }],
        "unsupported speculation limits",
    )?;
    require_ready(
        descriptor.cache_schema.groups.len() == 2,
        "expected paired FA and GDN cache groups",
    )?;
    let fa = cache_group(descriptor, "fa")?;
    let gdn = cache_group(descriptor, "gdn")?;
    require_ready(
        fa.capacity_blocks == u64::from(logical_num_blocks)
            && gdn.capacity_blocks == u64::from(mamba_blocks)
            && fa.block_tokens == config.block_size
            && gdn.block_tokens == config.block_size,
        "cache groups disagree with Ready capacities or Init block size",
    )
}

pub(in crate::models) fn cache_group<'a>(
    descriptor: &'a ModelDescriptor,
    name: &str,
) -> Result<&'a CacheGroup> {
    descriptor
        .cache_schema
        .groups
        .iter()
        .find(|group| group.name == name)
        .ok_or_else(|| Error::WorkerError(format!("missing {name} cache group")))
}

pub(in crate::models) fn validate_group(
    group: &CacheGroup,
    kind: CacheKind,
    layout_id: &str,
    layers: &[u32],
) -> Result<()> {
    require_ready(
        group.kind == kind
            && group.layers == layers
            && group.source == (CacheSource::Owned {})
            && matches!(
                &group.storage,
                CacheStorage::Layout { id, version: 1 } if id == layout_id
            ),
        "unsupported cache kind, layout, layer membership, or producer",
    )
}

fn same_members<T: Ord>(actual: &[T], expected: &[T]) -> bool {
    actual.iter().collect::<BTreeSet<_>>() == expected.iter().collect::<BTreeSet<_>>()
}
