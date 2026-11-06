mod slots;
mod tree;

pub use slots::{PoolReservation, SlotKey, SlotPool};
pub use tree::{NodeId, PrefixPath, TokenRadix};

use crate::hash::BlockHash;

pub const KV_PAGE_TOKENS: usize = 128;
pub const GDN_PERIOD_TOKENS: usize = 8192;
pub const QWEN_LAYOUT: &str = "qwen_radix_gdn8192_fp8kv128_v1";

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct CacheDomain {
    pub model: String,
    pub template: String,
    pub layout: String,
    pub namespace: String,
    pub salt: Option<BlockHash>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointId(pub SlotKey);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub SlotKey);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixEvidence {
    pub checkpoint_hit_end: usize,
    pub kv_contiguous_end: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CheckpointRead {
    #[default]
    Zero,
    Cached {
        id: CheckpointId,
        end: usize,
    },
}

impl CheckpointRead {
    pub fn end(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::Cached { end, .. } => end,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageAccess {
    ReadOnly,
    UniqueWrite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageBinding {
    pub page_index: usize,
    pub id: PageId,
    pub valid_end: usize,
    pub access: PageAccess,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageCopy {
    pub source: PageId,
    pub destination: PageId,
    pub valid_tokens: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportRole {
    Periodic,
    CandidateTail,
    PrivateEndpoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointExport {
    pub end: usize,
    pub destination: CheckpointId,
    pub role: ExportRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointSource {
    Forward,
    SavedHidden { id: CheckpointId, end: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSpanMode {
    Reuse,
    Write,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSpan {
    pub start: usize,
    pub end: usize,
    pub mode: KvSpanMode,
}
