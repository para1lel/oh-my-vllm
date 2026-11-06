use super::*;
use crate::protocol::{Action, ExecuteResultMsg, RequestResultMsg};
use oh_my_vllm_scheduler::execution::{ExecutionKey, SamplingMode, WorkAction, WorkSpec};

const EMPTY: BTreeMap<u64, usize> = BTreeMap::new();

fn row() -> ScheduledRequest<()> {
    ScheduledRequest {
        request_id: 7,
        execution: ExecutionKey {
            worker_epoch: 4,
            step_id: 9,
            request_id: 7,
            lease_id: 3,
        },
        work: WorkSpec {
            purpose: WorkPurpose::Finalize,
            action: WorkAction::ExportOnly { endpoint: 17 },
            sampling: SamplingMode::None,
        },
        token_ids: vec![],
        prefill_token_ids: None,
        num_computed_tokens: 17,
        state: (),
        accepted_end: 17,
    }
}

fn output(id: u64) -> RequestResultMsg {
    RequestResultMsg {
        request_id: id,
        lease_id: 3,
        purpose: WorkPurpose::Finalize,
        action: Action::ExportOnly,
        error: None,
        token_ids: vec![],
        num_accepted_draft_tokens: 0,
        new_draft_token_ids: vec![],
        text: String::new(),
        finish_reason: None,
        reasoning_tokens: 0,
        computed_end: 17,
        accepted_end: 17,
        validated_exports: vec![],
        committed_page_valid_ends: vec![],
        mtp_accepted_endpoint: None,
    }
}

fn reply(outputs: Vec<RequestResultMsg>) -> ExecuteResultMsg {
    ExecuteResultMsg {
        rpc_id: 11,
        worker_epoch: 4,
        step_id: 9,
        outputs,
        rank_completions: vec![],
    }
}

fn sampled_row(id: u64, sampling: SamplingMode) -> ScheduledRequest<()> {
    let mut scheduled = row();
    scheduled.request_id = id;
    scheduled.execution.request_id = id;
    scheduled.work = WorkSpec {
        purpose: WorkPurpose::Advance,
        action: WorkAction::Forward { start: 16, end: 21 },
        sampling,
    };
    scheduled.token_ids = vec![1; 5];
    scheduled.accepted_end = if matches!(sampling, SamplingMode::Verify { .. }) {
        17
    } else {
        21
    };
    assert!(scheduled.work.validate().is_ok());
    assert_eq!(
        scheduled.work.range().1 - scheduled.work.range().0,
        scheduled.token_ids.len()
    );
    scheduled
}

fn sampled_output(id: u64) -> RequestResultMsg {
    let mut result = output(id);
    result.purpose = WorkPurpose::Advance;
    result.action = Action::Forward;
    result.token_ids = vec![63];
    result.computed_end = 21;
    result
}

#[test]
fn generated_tokens_must_fit_the_negotiated_vocabulary_before_batch_acceptance() {
    let rows = [
        sampled_row(7, SamplingMode::Target),
        sampled_row(8, SamplingMode::Target),
    ];
    assert!(
        validate_reply(
            &rows,
            &reply(vec![sampled_output(7), sampled_output(8)]),
            4,
            9,
            64,
            4,
            &EMPTY
        )
        .is_ok()
    );
    for token in [64, u32::MAX] {
        let mut outputs = vec![sampled_output(7), sampled_output(8)];
        outputs[1].token_ids[0] = token;
        assert!(validate_reply(&rows, &reply(outputs), 4, 9, 64, 4, &EMPTY).is_err());
    }
}

#[test]
fn token_range_uses_the_ready_vocabulary_instead_of_a_model_constant() {
    let rows = [sampled_row(7, SamplingMode::Target)];
    for vocabulary in [64, 128, 512] {
        let mut result = sampled_output(7);
        result.token_ids[0] = vocabulary - 1;
        assert!(validate_reply(&rows, &reply(vec![result]), 4, 9, vocabulary, 4, &EMPTY).is_ok());
        let mut result = sampled_output(7);
        result.token_ids[0] = vocabulary;
        assert!(validate_reply(&rows, &reply(vec![result]), 4, 9, vocabulary, 4, &EMPTY).is_err());
    }
}

#[test]
fn new_drafts_must_fit_the_negotiated_vocabulary() {
    let rows = [sampled_row(7, SamplingMode::Target)];
    let mut valid = sampled_output(7);
    valid.new_draft_token_ids = vec![0, 63];
    assert!(validate_reply(&rows, &reply(vec![valid]), 4, 9, 64, 4, &EMPTY).is_ok());
    for token in [64, u32::MAX] {
        let mut result = sampled_output(7);
        result.new_draft_token_ids = vec![0, token];
        assert!(validate_reply(&rows, &reply(vec![result]), 4, 9, 64, 4, &EMPTY).is_err());
    }
}

#[test]
fn draft_limits_follow_configuration_and_allow_shorter_proposals() {
    for cap in [0, 2, 4, 5] {
        let rows = [sampled_row(7, SamplingMode::Target)];
        for count in 0..=cap + 1 {
            let mut result = sampled_output(7);
            result.new_draft_token_ids = vec![63; count];
            assert_eq!(
                validate_reply(&rows, &reply(vec![result]), 4, 9, 64, cap, &EMPTY).is_ok(),
                count <= cap,
            );
        }
    }
    let rows = [sampled_row(7, SamplingMode::Verify { drafts: 4 })];
    let mut result = sampled_output(7);
    result.computed_end = 17;
    assert!(validate_reply(&rows, &reply(vec![result]), 4, 9, 64, 2, &EMPTY).is_err());
}

#[test]
fn none_sampling_rejects_generation_effects_but_keeps_cumulative_reasoning_usage() {
    let rows = [sampled_row(7, SamplingMode::None)];
    let verified = BTreeMap::from([(7, 7)]);
    for effect in 0..4 {
        let mut result = sampled_output(7);
        result.token_ids.clear();
        result.reasoning_tokens = 7;
        match effect {
            1 => result.new_draft_token_ids.push(63),
            2 => result.text = "phantom output".into(),
            3 => result.finish_reason = Some("stop".into()),
            _ => {}
        }
        assert_eq!(
            validate_reply(&rows, &reply(vec![result]), 4, 9, 64, 4, &verified).is_ok(),
            effect == 0,
        );
    }
}

#[test]
fn none_sampling_requires_the_previous_verified_cumulative_count() {
    let rows = [sampled_row(7, SamplingMode::None)];
    for previous in [None, Some(0), Some(7)] {
        let verified = previous
            .map(|count| BTreeMap::from([(7, count)]))
            .unwrap_or_default();
        for returned in [0, 6, 7, 8, usize::MAX] {
            let mut result = sampled_output(7);
            result.token_ids.clear();
            result.reasoning_tokens = returned;
            assert_eq!(
                validate_reply(&rows, &reply(vec![result]), 4, 9, 64, 4, &verified).is_ok(),
                returned == previous.unwrap_or(0),
            );
        }
    }
}

#[test]
fn verify_validates_every_generated_and_new_draft_token() {
    let rows = [sampled_row(7, SamplingMode::Verify { drafts: 4 })];
    for tokens in [vec![0, 63], vec![64, 63], vec![0, u32::MAX]] {
        let valid = tokens.iter().all(|&token| token < 64);
        let mut result = sampled_output(7);
        result.token_ids = tokens;
        result.num_accepted_draft_tokens = 1;
        result.computed_end = 18;
        result.new_draft_token_ids = vec![63; 4];
        assert_eq!(
            validate_reply(&rows, &reply(vec![result]), 4, 9, 64, 4, &EMPTY).is_ok(),
            valid
        );
    }
    let mut result = sampled_output(7);
    result.computed_end = 17;
    result.new_draft_token_ids = vec![64];
    assert!(validate_reply(&rows, &reply(vec![result]), 4, 9, 64, 4, &EMPTY).is_err());
}

#[test]
fn zero_row_finalize_requires_one_complete_current_reply() {
    assert!(validate_reply(&[row()], &reply(vec![output(7)]), 4, 9, 64, 4, &EMPTY).is_ok());
    for outputs in [vec![], vec![output(8)], vec![output(7), output(7)]] {
        assert!(validate_reply(&[row()], &reply(outputs), 4, 9, 64, 4, &EMPTY).is_err());
    }
    let mut wrong = output(7);
    wrong.lease_id += 1;
    assert!(validate_reply(&[row()], &reply(vec![wrong]), 4, 9, 64, 4, &EMPTY).is_err());
    assert!(validate_reply(&[row()], &reply(vec![output(7)]), 5, 9, 64, 4, &EMPTY).is_err());
    assert!(validate_reply(&[row()], &reply(vec![output(7)]), 4, 10, 64, 4, &EMPTY).is_err());
    let mut sampled = output(7);
    sampled.token_ids.push(42);
    assert!(validate_reply(&[row()], &reply(vec![sampled]), 4, 9, 64, 4, &EMPTY).is_err());
    let mut usage = output(7);
    usage.reasoning_tokens = 1;
    assert!(validate_reply(&[row()], &reply(vec![usage]), 4, 9, 64, 4, &EMPTY).is_err());
}

#[test]
fn verify_uses_accepted_forward_endpoint_and_does_not_accept_target_counts() {
    let mut scheduled = row();
    scheduled.work = WorkSpec {
        purpose: WorkPurpose::Advance,
        action: WorkAction::Forward { start: 16, end: 21 },
        sampling: SamplingMode::Verify { drafts: 4 },
    };
    let mut result = output(7);
    result.purpose = WorkPurpose::Advance;
    result.action = Action::Forward;
    result.token_ids = vec![42, 43];
    result.num_accepted_draft_tokens = 1;
    result.computed_end = 18;
    assert!(
        validate_reply(
            &[scheduled.clone()],
            &reply(vec![result]),
            4,
            9,
            64,
            4,
            &EMPTY
        )
        .is_ok()
    );
    let mut result = output(7);
    result.purpose = WorkPurpose::Advance;
    result.action = Action::Forward;
    result.token_ids = vec![42, 43];
    result.num_accepted_draft_tokens = 1;
    result.computed_end = 21;
    assert!(validate_reply(&[scheduled], &reply(vec![result]), 4, 9, 64, 4, &EMPTY).is_err());
}
