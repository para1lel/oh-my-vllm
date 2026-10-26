use serde::{Deserialize, Serialize};

mod validation;

pub const MODEL_DESCRIPTOR_VERSION: u32 = 1;
pub const CACHE_SCHEMA_VERSION: u32 = 1;
pub const MAX_IDENTIFIER_BYTES: usize = 128;
pub const MAX_CACHE_GROUPS: usize = 256;
pub const MAX_LAYER_REFERENCES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDescriptor {
    pub schema_version: u32,
    pub architecture: String,
    pub model_type: String,
    pub checkpoint_sha256: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub vocabulary_size: u32,
    pub max_context_tokens: u64,
    pub capabilities: Vec<Capability>,
    pub weight_formats: Vec<WeightFormat>,
    pub cache_schema: CacheSchema,
    pub parallel_plan: ParallelPlan,
    pub speculation: Vec<SpeculationSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Text,
    Images,
    Video,
    Tools,
    Reasoning,
    StructuredOutputs,
    PrefixCache,
    HostOffload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    Bf16,
    F32,
    #[serde(rename = "qwen_fp8_e4m3_block128x128_fp32_scales")]
    QwenFp8E4m3Block128x128Fp32Scales,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheSchema {
    pub schema_version: u32,
    pub groups: Vec<CacheGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheGroup {
    pub name: String,
    pub kind: CacheKind,
    pub block_tokens: u32,
    pub capacity_blocks: u64,
    pub storage: CacheStorage,
    pub layers: Vec<u32>,
    pub source: CacheSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    FullAttention,
    Recurrent,
    SlidingWindow,
    CompressedKv,
    IndexKeys,
    ModelCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CacheStorage {
    Primitive { format: ScalarFormat },
    Layout { id: String, version: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarFormat {
    Bf16,
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CacheSource {
    Owned {},
    Shared { producer: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelPlan {
    pub strategy: ParallelStrategy,
    pub world_size: u32,
    pub ranks: Vec<u32>,
    pub request_placement: RequestPlacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelStrategy {
    Tp1,
    Tp4,
    Tep4,
    Dep4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestPlacement {
    Replicated,
    RankPartitioned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationSpec {
    pub method: SpeculationMethod,
    pub predictor_stages: u32,
    pub max_draft_tokens: u32,
    pub max_verification_rows: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculationMethod {
    Mtp,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field}: {reason}")]
pub struct ContractError {
    pub field: &'static str,
    pub reason: &'static str,
}
