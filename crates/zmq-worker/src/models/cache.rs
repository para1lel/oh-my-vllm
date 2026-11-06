use oh_my_vllm_kv_cache::{HybridCoordinator, radix as domain};
use oh_my_vllm_scheduler::policy::{
    CachePolicy,
    qwen::{QwenAssignment, QwenResidentCache},
    resident::ResidentState,
};

use crate::{client::WorkerClient, protocol as wire};

pub trait ServiceCache: CachePolicy<Hit: Send, State: Send + Sync> + Send + 'static {
    fn encode_assignment(state: &Self::State) -> anyhow::Result<wire::ModelStateWire>;

    fn resource_snapshot(&self) -> serde_json::Value;

    fn attach_client(&self, _client: &mut WorkerClient) {}
}

fn resident_wire(state: &ResidentState) -> wire::ModelStateWire {
    wire::ModelStateWire::DeepseekResidentV1(wire::ResidentStateWire {
        fa_block_table: state.fa_block_table.clone(),
        mamba_block_table: state.mamba_block_table.clone(),
        fa_block_copies: state.fa_block_copies.clone(),
        prefix_state_slot: state.prefix_state_slot,
        cache_checkpoint: state.cache_checkpoint,
    })
}

fn resident_resources(counts: (usize, usize, usize, usize, usize, usize)) -> serde_json::Value {
    let (fa_blocks, fa_free, fa_cached, gdn_blocks, gdn_free, gdn_cached) = counts;
    serde_json::json!({ "fa_blocks": fa_blocks, "fa_free": fa_free, "fa_cached": fa_cached,
        "gdn_blocks": gdn_blocks, "gdn_free": gdn_free, "gdn_cached": gdn_cached })
}

impl ServiceCache for HybridCoordinator {
    fn encode_assignment(state: &Self::State) -> anyhow::Result<wire::ModelStateWire> {
        Ok(resident_wire(state))
    }

    fn resource_snapshot(&self) -> serde_json::Value {
        resident_resources(self.resource_counts())
    }
}

impl ServiceCache for super::deepseek_v41::cache::Dep4Cache {
    fn encode_assignment(state: &Self::State) -> anyhow::Result<wire::ModelStateWire> {
        Ok(resident_wire(state))
    }

    fn resource_snapshot(&self) -> serde_json::Value {
        resident_resources(self.resource_counts())
    }

    fn attach_client(&self, client: &mut WorkerClient) {
        client.attach_request_owners(self.owners.clone());
    }
}

impl ServiceCache for QwenResidentCache {
    fn encode_assignment(state: &QwenAssignment) -> anyhow::Result<wire::ModelStateWire> {
        let resume = match state.resume {
            domain::CheckpointRead::Zero => wire::EndpointRef {
                end: 0,
                slot: 0,
                version: 0,
            },
            domain::CheckpointRead::Cached { id, end } => endpoint(end, id)?,
        };
        let endpoint_source = match state.endpoint_source {
            domain::EndpointSource::Forward => wire::EndpointSource::Forward,
            domain::EndpointSource::SavedHidden { id, end } => wire::EndpointSource::SavedHidden {
                slot: id.0.slot,
                version: id.0.version,
                end: end.try_into()?,
            },
        };
        let state = wire::QwenStateWire {
            layout_fingerprint: domain::QWEN_LAYOUT.into(),
            checkpoint_hit_end: state.evidence.checkpoint_hit_end.try_into()?,
            kv_contiguous_end: state.evidence.kv_contiguous_end.try_into()?,
            resume,
            endpoint_source,
            page_table: state
                .page_table
                .iter()
                .map(page)
                .collect::<anyhow::Result<_>>()?,
            kv_spans: state
                .kv_spans
                .iter()
                .map(|span| {
                    Ok(wire::KvSpan {
                        start: span.start.try_into()?,
                        end: span.end.try_into()?,
                        mode: match span.mode {
                            domain::KvSpanMode::Reuse => wire::KvMode::Reuse,
                            domain::KvSpanMode::Write => wire::KvMode::Write,
                        },
                    })
                })
                .collect::<anyhow::Result<_>>()?,
            page_copies: state
                .page_copies
                .iter()
                .map(|copy| wire::PageCopy {
                    source_slot: copy.source.0.slot,
                    source_version: copy.source.0.version,
                    destination_slot: copy.destination.0.slot,
                    destination_version: copy.destination.0.version,
                    valid_tokens: copy.valid_tokens,
                })
                .collect(),
            exports: state
                .exports
                .iter()
                .map(export)
                .collect::<anyhow::Result<_>>()?,
            private_checkpoints: state
                .private_checkpoints
                .iter()
                .map(|id| wire::SlotRef {
                    slot: id.0.slot,
                    version: id.0.version,
                })
                .collect(),
            mtp_endpoint: state
                .mtp_endpoint
                .map(|(end, id)| endpoint(end, id))
                .transpose()?,
        };
        Ok(wire::ModelStateWire::QwenRadixV1(state))
    }

    fn resource_snapshot(&self) -> serde_json::Value {
        let count = self.resource_counts();
        serde_json::json!({ "fa_blocks": count.usable_pages, "fa_free": count.available_pages,
            "fa_cached": count.cached_pages, "gdn_blocks": count.usable_checkpoints,
            "gdn_free": count.available_checkpoints, "gdn_cached": count.cached_checkpoints,
            "private_checkpoints": count.private_checkpoints,
            "available_private_checkpoints": count.available_private_checkpoints,
            "scratch_checkpoints": count.scratch_checkpoints })
    }
}

fn endpoint(end: usize, id: domain::CheckpointId) -> anyhow::Result<wire::EndpointRef> {
    Ok(wire::EndpointRef {
        end: end.try_into()?,
        slot: id.0.slot,
        version: id.0.version,
    })
}

fn page(page: &domain::PageBinding) -> anyhow::Result<wire::PageBinding> {
    Ok(wire::PageBinding {
        page_index: page.page_index.try_into()?,
        slot: page.id.0.slot,
        version: page.id.0.version,
        valid_tokens: page
            .valid_end
            .saturating_sub(page.page_index * domain::KV_PAGE_TOKENS)
            .try_into()?,
        access: match page.access {
            domain::PageAccess::ReadOnly => wire::PageAccess::Read,
            domain::PageAccess::UniqueWrite => wire::PageAccess::Write,
        },
    })
}

fn export(export: &domain::CheckpointExport) -> anyhow::Result<wire::CheckpointExport> {
    Ok(wire::CheckpointExport {
        end: export.end.try_into()?,
        slot: export.destination.0.slot,
        version: export.destination.0.version,
        role: match export.role {
            domain::ExportRole::Periodic => wire::ExportRole::Periodic,
            domain::ExportRole::CandidateTail => wire::ExportRole::CandidateTail,
            domain::ExportRole::PrivateEndpoint => wire::ExportRole::PrivateEndpoint,
        },
    })
}
