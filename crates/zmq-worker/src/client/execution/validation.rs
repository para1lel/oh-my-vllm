use crate::error::{Error, Result};
use crate::protocol::{ExecuteResultMsg, RequestResultMsg};
use oh_my_vllm_scheduler::execution::{SamplingMode, WorkPurpose};
use oh_my_vllm_scheduler::output::ScheduledRequest;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests;

fn invalid(reason: &str) -> Error {
    Error::UnexpectedMessageType(reason.to_owned())
}

pub(super) fn execution_step<S>(
    rows: &[ScheduledRequest<S>],
    epoch: u64,
    previous: u64,
) -> Result<u64> {
    let step = rows
        .first()
        .map_or(previous + 1, |row| row.execution.step_id);
    if step <= previous
        || rows.iter().any(|row| {
            row.execution.worker_epoch != epoch
                || row.execution.step_id != step
                || row.execution.request_id != row.request_id
        })
    {
        return Err(invalid(
            "scheduled execution identity is stale or inconsistent",
        ));
    }
    Ok(step)
}

pub(super) fn validate_reply<S>(
    rows: &[ScheduledRequest<S>],
    reply: &ExecuteResultMsg,
    epoch: u64,
    step: u64,
    vocabulary_size: u32,
    max_draft_tokens: usize,
    verified_reasoning_tokens: &BTreeMap<u64, usize>,
) -> Result<()> {
    if reply.worker_epoch != epoch || reply.step_id != step {
        return Err(invalid("execution reply has stale epoch or step"));
    }
    if reply
        .rank_completions
        .iter()
        .any(|rank| rank.worker_epoch != epoch || rank.step_id != step)
    {
        return Err(invalid("rank reply has stale epoch or step"));
    }
    let expected: BTreeMap<_, _> = rows.iter().map(|row| (row.request_id, row)).collect();
    let mut received = BTreeSet::new();
    if expected.len() != rows.len() {
        return Err(invalid("duplicate scheduled request ID"));
    }
    for output in &reply.outputs {
        let Some(row) = expected.get(&output.request_id) else {
            return Err(invalid("unknown execution request ID"));
        };
        if !received.insert(output.request_id) {
            return Err(invalid("duplicate execution request ID"));
        }
        validate_row(
            row,
            output,
            vocabulary_size,
            max_draft_tokens,
            verified_reasoning_tokens
                .get(&output.request_id)
                .copied()
                .unwrap_or(0),
        )?;
    }
    if received.len() != expected.len() {
        return Err(invalid("missing execution request ID"));
    }
    Ok(())
}

fn validate_tokens(
    sampling: SamplingMode,
    output: &RequestResultMsg,
    vocabulary_size: u32,
    max_draft_tokens: usize,
) -> Result<()> {
    if output
        .token_ids
        .iter()
        .chain(&output.new_draft_token_ids)
        .any(|&token| token >= vocabulary_size)
    {
        return Err(invalid(
            "execution reply token is outside the negotiated vocabulary",
        ));
    }
    if output.new_draft_token_ids.len() > max_draft_tokens
        || matches!(sampling, SamplingMode::Verify { drafts } if drafts > max_draft_tokens)
    {
        return Err(invalid(
            "execution reply exceeds the configured draft limit",
        ));
    }
    if sampling == SamplingMode::None
        && (!output.new_draft_token_ids.is_empty()
            || !output.text.is_empty()
            || output.finish_reason.is_some())
    {
        return Err(invalid("non-sampling result advances generation state"));
    }
    Ok(())
}

fn validate_row<S>(
    row: &ScheduledRequest<S>,
    output: &RequestResultMsg,
    vocabulary_size: u32,
    max_draft_tokens: usize,
    verified_reasoning_tokens: usize,
) -> Result<()> {
    if output.lease_id != row.execution.lease_id
        || output.purpose != row.work.purpose
        || output.action != super::codec::action(row.work.action)
    {
        return Err(invalid("execution reply changed lease, purpose or action"));
    }
    if output.error.is_some() {
        if !output.token_ids.is_empty()
            || output.num_accepted_draft_tokens != 0
            || !output.new_draft_token_ids.is_empty()
            || !output.validated_exports.is_empty()
            || !output.committed_page_valid_ends.is_empty()
            || output.mtp_accepted_endpoint.is_some()
        {
            return Err(invalid("error result contains successful state"));
        }
        return Ok(());
    }
    validate_tokens(row.work.sampling, output, vocabulary_size, max_draft_tokens)?;
    if row.work.purpose == WorkPurpose::Advance
        && row.work.sampling == SamplingMode::None
        && output.reasoning_tokens != verified_reasoning_tokens
    {
        return Err(invalid(
            "non-sampling result changed cumulative reasoning usage",
        ));
    }
    if row
        .work
        .expected_tokens(output.num_accepted_draft_tokens as usize)
        != Some(output.token_ids.len())
    {
        return Err(invalid(
            "execution reply token count disagrees with sampling",
        ));
    }
    if row.work.purpose == WorkPurpose::Finalize
        && (!output.new_draft_token_ids.is_empty()
            || !output.text.is_empty()
            || output.finish_reason.is_some()
            || output.reasoning_tokens != 0)
    {
        return Err(invalid("finalize result advances generation state"));
    }
    let expected_end = match row.work.sampling {
        SamplingMode::Verify { drafts } => row
            .work
            .range()
            .1
            .checked_sub(drafts)
            .and_then(|end| end.checked_add(output.num_accepted_draft_tokens as usize)),
        _ => Some(row.work.range().1),
    };
    if Some(output.computed_end as usize) != expected_end {
        return Err(invalid(
            "execution reply computed endpoint differs from scheduled endpoint",
        ));
    }
    Ok(())
}
