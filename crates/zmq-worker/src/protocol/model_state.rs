use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelStateWire {
    QwenRadixV1(QwenStateWire),
    DeepseekResidentV1(ResidentStateWire),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentStateWire {
    pub fa_block_table: Vec<u32>,
    pub mamba_block_table: Vec<u32>,
    pub fa_block_copies: Vec<(u32, u32, u32)>,
    pub prefix_state_slot: Option<u32>,
    pub cache_checkpoint: Option<(usize, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotRef {
    pub slot: u32,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointRef {
    pub end: u32,
    pub slot: u32,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenStateWire {
    pub layout_fingerprint: String,
    pub checkpoint_hit_end: u32,
    pub kv_contiguous_end: u32,
    pub resume: EndpointRef,
    pub private_checkpoints: Vec<SlotRef>,
    pub page_table: Vec<PageBinding>,
    pub kv_spans: Vec<KvSpan>,
    pub page_copies: Vec<PageCopy>,
    pub exports: Vec<CheckpointExport>,
    pub endpoint_source: EndpointSource,
    pub mtp_endpoint: Option<EndpointRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageBinding {
    pub page_index: u32,
    pub slot: u32,
    pub version: u64,
    pub valid_tokens: u16,
    pub access: PageAccess,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageAccess {
    Read,
    Write,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSpan {
    pub start: u32,
    pub end: u32,
    pub mode: KvMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvMode {
    Reuse,
    Write,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageCopy {
    pub source_slot: u32,
    pub source_version: u64,
    pub destination_slot: u32,
    pub destination_version: u64,
    pub valid_tokens: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointExport {
    pub end: u32,
    pub slot: u32,
    pub version: u64,
    pub role: ExportRole,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportRole {
    Periodic,
    CandidateTail,
    PrivateEndpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EndpointSource {
    Forward,
    SavedHidden { slot: u32, version: u64, end: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageValidEnd {
    pub page_index: u32,
    pub slot: u32,
    pub version: u64,
    pub valid_end: u32,
}
