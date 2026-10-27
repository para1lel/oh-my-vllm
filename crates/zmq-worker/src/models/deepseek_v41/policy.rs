use std::collections::BTreeSet;

use oh_my_vllm_model_contract::{
    CacheGroup, CacheKind, CacheSource, CacheStorage, Capability, ModelDescriptor,
    ParallelStrategy, RequestPlacement, SpeculationMethod, SpeculationSpec, WeightFormat,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::super::policy::{ModelDefaults, SpeculativeMode, require_options, require_ready};
use crate::client::WorkerConfig;
use crate::error::{Error, Result};

const CONFIG_SHA256: &str = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879";
const ARCHITECTURE: &str = "DeepseekV41ForCausalLM";
const MODEL_TYPE: &str = "deepseek_v41";
const CAPABILITIES: [Capability; 1] = [Capability::Text];
const WEIGHT_FORMATS: [WeightFormat; 4] = [
    WeightFormat::Bf16,
    WeightFormat::F32,
    WeightFormat::DeepseekFp4,
    WeightFormat::DeepseekFp8E4m3Block32Ue8m0,
];

pub(in crate::models) fn defaults() -> ModelDefaults {
    ModelDefaults {
        block_size: 128,
        tensor_parallel_size: 4,
        max_model_len: 1048576,
        max_num_batched_tokens: 32768,
        max_num_seqs: 64,
        mamba_blocks: 128,
    }
}

pub(in crate::models) fn vocabulary_size() -> u32 {
    129280
}

pub(in crate::models) fn max_offline_batch_size() -> usize {
    64
}

pub(in crate::models) fn served_model_name() -> &'static str {
    "deepseek-v4.1-flash"
}

pub(in crate::models) fn matches_config(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some(MODEL_TYPE)
        && config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|values| values.len() == 1 && values[0].as_str() == Some(ARCHITECTURE))
}

pub(in crate::models) fn validate_config(config: &Value, raw: &[u8]) -> Result<()> {
    require_options(
        matches_config(config),
        "unsupported DeepSeek architecture or model type",
    )?;
    let quantization = &config["quantization_config"];
    require_options(
        quantization.is_object()
            && quantization["quant_method"] == "fp8"
            && quantization["activation_scheme"] == "dynamic"
            && quantization["weight_block_size"] == serde_json::json!([32, 32])
            && quantization["scale_fmt"] == "ue8m0"
            && quantization["expert_dtype"] == "fp4",
        "unsupported DeepSeek weight or activation format",
    )?;
    let text = &config["text_config"];
    require_options(
        text.is_object()
            && text["vocab_size"] == vocabulary_size()
            && text["hidden_size"] == 5120
            && text["num_hidden_layers"] == 40
            && text["num_attention_heads"] == 64
            && text["num_key_value_heads"] == 1
            && text["head_dim"] == 512
            && text["max_position_embeddings"] == defaults().max_model_len
            && text["n_routed_experts"] == 384
            && text["num_experts_per_tok"] == 6
            && text["sliding_window"] == 128
            && text["num_nextn_predict_layers"] == 3
            && text["dspark_block_size"] == 5,
        "unsupported DeepSeek text configuration",
    )?;
    require_options(
        format!("{:x}", Sha256::digest(raw)) == CONFIG_SHA256,
        "DeepSeek configuration identity differs from the model contract",
    )
}

pub(in crate::models) fn resolve_speculation(
    mode: Option<&str>,
    drafts: usize,
) -> Result<SpeculativeMode> {
    require_options(
        matches!(drafts, 0 | 5),
        "only ordinary decode and five-draft DSpark5 are implemented",
    )?;
    let name = mode.unwrap_or(if drafts == 5 {
        "dspark5"
    } else {
        "none"
    });
    let num_tokens = match name {
        "none" => 0,
        "dspark5" => 5,
        _ => {
            return Err(Error::ModelConfiguration(
                "unsupported DeepSeek speculative mode".into(),
            ));
        }
    };
    require_options(
        drafts == 0 || drafts == num_tokens,
        "speculative mode disagrees with the requested draft count",
    )?;
    Ok(SpeculativeMode {
        name: if num_tokens == 5 {
            "dspark5"
        } else {
            "none"
        },
        num_tokens,
    })
}

pub(in crate::models) fn validate_options(config: &WorkerConfig) -> Result<()> {
    require_options(
        config.model_path.as_os_str() != "stub",
        "selected DeepSeek policy requires a checkpoint path",
    )?;
    require_options(
        config.tensor_parallel_size == 4,
        "DeepSeek requires four worker ranks",
    )?;
    require_options(
        config.block_size == defaults().block_size,
        "DeepSeek requires 128-token cache blocks",
    )?;
    require_options(
        (1..=defaults().max_model_len).contains(&config.max_model_len),
        "requested context exceeds DeepSeek limits",
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
        "selected DeepSeek policy requires a checkpoint path",
    )?;
    require_ready(
        descriptor.architecture == ARCHITECTURE
            && descriptor.model_type == MODEL_TYPE
            && descriptor.config_sha256 == CONFIG_SHA256,
        "unsupported DeepSeek model identity",
    )?;
    require_ready(
        same_members(&descriptor.weight_formats, &WEIGHT_FORMATS),
        "unsupported DeepSeek weight formats",
    )?;
    validate_runtime_ready(config, descriptor, logical_num_blocks, mamba_blocks)?;
    validate_group(
        cache_group(descriptor, "window")?,
        CacheKind::SlidingWindow,
        "deepseek_v41_sliding_window_kv_bf16",
        &(0..40).collect::<Vec<_>>(),
    )?;
    validate_group(
        cache_group(descriptor, "compressed_kv")?,
        CacheKind::CompressedKv,
        "deepseek_v41_compressed_kv_source_bf16",
        &[2, 8, 14, 20],
    )?;
    validate_group(
        cache_group(descriptor, "index_keys")?,
        CacheKind::IndexKeys,
        "deepseek_v41_index_keys_bf16",
        &[2, 8, 14, 20, 24, 28, 32, 36],
    )?;
    validate_group(
        cache_group(descriptor, "accepted_state")?,
        CacheKind::ModelCheckpoint,
        "deepseek_v41_accepted_state_v1",
        &(0..40).collect::<Vec<_>>(),
    )
}

pub(in crate::models) fn validate_runtime_ready(
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    logical_num_blocks: u32,
    mamba_blocks: u32,
) -> Result<()> {
    validate_options(config)
        .map_err(|error| Error::WorkerError(format!("invalid model readiness: {error}")))?;
    require_ready(
        matches!(
            descriptor.parallel_plan.strategy,
            ParallelStrategy::Tp4 | ParallelStrategy::Tep4 | ParallelStrategy::Dep4
        ) && descriptor.parallel_plan.world_size == 4
            && descriptor.parallel_plan.ranks == [0, 1, 2, 3]
            && descriptor.parallel_plan.request_placement
                == if descriptor.parallel_plan.strategy == ParallelStrategy::Dep4 {
                    RequestPlacement::RankPartitioned
                } else {
                    RequestPlacement::Replicated
                },
        "unsupported DeepSeek parallel plan",
    )?;
    require_ready(
        descriptor.vocabulary_size == vocabulary_size()
            && descriptor.max_context_tokens == u64::from(defaults().max_model_len),
        "unsupported DeepSeek vocabulary or context ceiling",
    )?;
    require_ready(
        logical_num_blocks > 1
            && mamba_blocks > 1
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
        "unsupported DeepSeek capability set",
    )?;
    require_ready(
        descriptor.speculation
            == [SpeculationSpec {
                method: SpeculationMethod::DsPark5,
                predictor_stages: 3,
                max_draft_tokens: 5,
                max_verification_rows: 6,
            }],
        "unsupported DSpark5 limits",
    )?;
    require_ready(
        descriptor.cache_schema.groups.len() == 4,
        "expected DeepSeek cache groups",
    )?;
    require_ready(
        cache_group(descriptor, "window")?.capacity_blocks == u64::from(logical_num_blocks)
            && cache_group(descriptor, "accepted_state")?.capacity_blocks
                == u64::from(mamba_blocks)
            && cache_group(descriptor, "compressed_kv")?.capacity_blocks > 0
            && cache_group(descriptor, "index_keys")?.capacity_blocks > 0
            && descriptor
                .cache_schema
                .groups
                .iter()
                .all(|group| group.block_tokens == config.block_size),
        "cache groups disagree with Ready capacities or Init block size",
    )
}

fn cache_group<'a>(descriptor: &'a ModelDescriptor, name: &str) -> Result<&'a CacheGroup> {
    descriptor
        .cache_schema
        .groups
        .iter()
        .find(|group| group.name == name)
        .ok_or_else(|| Error::WorkerError(format!("missing {name} cache group")))
}

fn validate_group(
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
        "unsupported DeepSeek cache kind, layout, layer membership, or producer",
    )
}

fn same_members<T: Ord>(actual: &[T], expected: &[T]) -> bool {
    actual.iter().collect::<BTreeSet<_>>() == expected.iter().collect::<BTreeSet<_>>()
}
