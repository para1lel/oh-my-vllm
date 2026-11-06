use crate::error::{Error, Result};
use crate::models::ServiceCache;
use crate::protocol::{Action, RequestResultMsg, Sampling, ScheduledRequestMsg};
use oh_my_vllm_scheduler::execution::{CacheStepResult, ExecutionKey, SamplingMode, WorkAction};
use oh_my_vllm_scheduler::output::ScheduledRequest;

pub(super) fn action(action: WorkAction) -> Action {
    match action {
        WorkAction::Forward { .. } => Action::Forward,
        WorkAction::HeadOnly { .. } => Action::HeadOnly,
        WorkAction::ExportOnly { .. } => Action::ExportOnly,
    }
}

pub(super) fn encode_row<P: ServiceCache>(
    row: &ScheduledRequest<P::State>,
) -> Result<ScheduledRequestMsg> {
    row.work
        .validate()
        .map_err(|reason| Error::UnexpectedMessageType(reason.to_owned()))?;
    let (start, end) = row.work.range();
    if end.checked_sub(start) != Some(row.token_ids.len()) {
        return Err(Error::UnexpectedMessageType(
            "scheduled forward range differs from token rows".into(),
        ));
    }
    let narrow = |value| {
        u32::try_from(value)
            .map_err(|_| Error::UnexpectedMessageType("token position exceeds wire bound".into()))
    };
    Ok(ScheduledRequestMsg {
        request_id: row.request_id,
        lease_id: row.execution.lease_id,
        purpose: row.work.purpose,
        action: action(row.work.action),
        sampling: match row.work.sampling {
            SamplingMode::None => Sampling::None,
            SamplingMode::Target => Sampling::Target,
            SamplingMode::Verify { drafts } => Sampling::Verify {
                drafts: narrow(drafts)?,
            },
        },
        token_ids: row.token_ids.clone(),
        prefill_token_ids: row.prefill_token_ids.clone(),
        forward_start: narrow(start)?,
        forward_end: narrow(end)?,
        accepted_end: narrow(row.accepted_end)?,
        endpoint: narrow(end)?,
        model_state: P::encode_assignment(&row.state)
            .map_err(|error| Error::UnexpectedMessageType(error.to_string()))?,
    })
}

pub(super) fn cache_result(output: &RequestResultMsg, execution: ExecutionKey) -> CacheStepResult {
    CacheStepResult {
        execution,
        computed_end: output.computed_end as usize,
        accepted_end: output.accepted_end as usize,
        exported_checkpoints: output
            .validated_exports
            .iter()
            .map(|value| (value.end as usize, value.slot, value.version))
            .collect(),
        page_valid_ends: output
            .committed_page_valid_ends
            .iter()
            .map(|value| (value.slot, value.version, value.valid_end as usize))
            .collect(),
        mtp_accepted_end: output
            .mtp_accepted_endpoint
            .as_ref()
            .map(|value| value.end as usize),
    }
}
