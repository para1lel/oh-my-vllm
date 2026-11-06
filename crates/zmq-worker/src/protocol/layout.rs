use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const QWEN_LAYOUT_ID: &str = "qwen_radix_gdn8192_fp8kv128_v1";
pub const QWEN_FORMAT_FINGERPRINT: &str = "qwen_fp8_e4m3fn_rne_fp32scale_v1";
pub const DEEPSEEK_LAYOUT_ID: &str = "deepseek_v41_paired128_resident_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CacheLayout {
    QwenRadixV1 {
        layout_id: String,
        gdn_period_tokens: u32,
        kv_page_tokens: u32,
        usable_checkpoints: u32,
        usable_pages: u32,
        null_checkpoints: u32,
        null_pages: u32,
        private_checkpoints: u32,
        scratch_checkpoints: u32,
        pool_bytes: BTreeMap<String, u64>,
        format_fingerprint: String,
    },
    DeepseekResidentV1 {
        layout_id: String,
    },
}

impl CacheLayout {
    pub fn validate(&self, pages: u32, checkpoints: u32, sequences: u32, tokens: u32) -> bool {
        match self {
            Self::QwenRadixV1 {
                layout_id,
                gdn_period_tokens,
                kv_page_tokens,
                usable_checkpoints,
                usable_pages,
                null_checkpoints,
                null_pages,
                private_checkpoints,
                scratch_checkpoints,
                pool_bytes,
                format_fingerprint,
            } => {
                layout_id == QWEN_LAYOUT_ID
                    && *gdn_period_tokens == 8192
                    && *kv_page_tokens == 128
                    && *usable_checkpoints == checkpoints
                    && *usable_pages == pages
                    && checkpoints.checked_mul(64) == Some(pages)
                    && *null_checkpoints == 1
                    && *null_pages == 1
                    && *private_checkpoints == sequences.saturating_mul(6)
                    && *scratch_checkpoints == tokens.div_ceil(8192).saturating_add(sequences)
                    && !pool_bytes.is_empty()
                    && pool_bytes.values().all(|bytes| *bytes > 0)
                    && format_fingerprint == QWEN_FORMAT_FINGERPRINT
            }
            Self::DeepseekResidentV1 { layout_id } => layout_id == DEEPSEEK_LAYOUT_ID,
        }
    }
}
