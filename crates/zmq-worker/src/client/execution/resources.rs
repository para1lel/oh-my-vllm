use crate::error::{Error, Result};
use crate::protocol::{ModelStateWire, RequestResultMsg, ScheduledRequestMsg};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(
    rows: &[ScheduledRequestMsg],
    outputs: &[RequestResultMsg],
    mtp: bool,
) -> Result<()> {
    let rows: BTreeMap<_, _> = rows.iter().map(|row| (row.request_id, row)).collect();
    for output in outputs.iter().filter(|output| output.error.is_none()) {
        match &rows[&output.request_id].model_state {
            ModelStateWire::DeepseekResidentV1(_) => {
                if !output.validated_exports.is_empty()
                    || !output.committed_page_valid_ends.is_empty()
                    || output.mtp_accepted_endpoint.is_some()
                {
                    return Err(invalid("resident result contains Qwen resource evidence"));
                }
            }
            ModelStateWire::QwenRadixV1(state) => {
                let expected: BTreeSet<_> = state
                    .page_table
                    .iter()
                    .map(|page| (page.page_index, page.slot, page.version))
                    .collect();
                let actual: BTreeSet<_> = output
                    .committed_page_valid_ends
                    .iter()
                    .map(|page| (page.page_index, page.slot, page.version))
                    .collect();
                if expected != actual || actual.len() != output.committed_page_valid_ends.len() {
                    return Err(invalid("page reply changed page index, slot or version"));
                }
                validate_mtp(state, output, mtp)?;
            }
        }
    }
    Ok(())
}

fn validate_mtp(
    state: &crate::protocol::QwenStateWire,
    output: &RequestResultMsg,
    enabled: bool,
) -> Result<()> {
    let Some(endpoint) = &output.mtp_accepted_endpoint else {
        if enabled && output.purpose == oh_my_vllm_scheduler::execution::WorkPurpose::Finalize {
            return Err(invalid(
                "finalize result is missing the accepted MTP context",
            ));
        }
        return Ok(());
    };
    let exported = state.exports.iter().any(|export| {
        export.end == output.computed_end
            && export.slot == endpoint.slot
            && export.version == endpoint.version
    });
    let saved = state.exports.is_empty()
        && output.action == crate::protocol::Action::HeadOnly
        && matches!(
            state.endpoint_source,
            crate::protocol::EndpointSource::SavedHidden { slot, version, end }
                if slot == endpoint.slot && version == endpoint.version
                    && end == output.computed_end
        );
    if !enabled || endpoint.end != output.computed_end.saturating_sub(1) || !(exported || saved) {
        return Err(invalid(
            "MTP result differs from the accepted target bundle",
        ));
    }
    Ok(())
}

fn invalid(reason: &str) -> Error {
    Error::UnexpectedMessageType(reason.to_owned())
}
