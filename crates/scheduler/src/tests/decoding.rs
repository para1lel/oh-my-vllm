use super::*;
use crate::session::{RequestIdentity, SessionId, SessionSource};

fn identity(namespace: &str) -> RequestIdentity {
    RequestIdentity {
        session: SessionId {
            scope: "cpu".into(),
            source: SessionSource::Explicit,
            value: "one-session".into(),
        },
        namespace: namespace.into(),
    }
}

#[test]
fn admission_order_survives_newer_prepare_failure() {
    let mut scheduler = make_scheduler(64, 16);
    let first = scheduler.reserve_admission(identity("old"), 1);
    let mut request = make_req(1, 4, 2);
    request.ticket = Some(first.clone());
    assert!(scheduler.add_request(request));
    assert!(!scheduler.release_admission(&first));
    assert!(scheduler.sessions.is_latest(&first));
    let second = scheduler.reserve_admission(identity("new"), 2);
    assert!(!scheduler.sessions.is_latest(&first));
    assert!(scheduler.release_admission(&second));
    assert!(!scheduler.sessions.is_latest(&first));
    assert!(!scheduler.release_admission(&second));
    scheduler.abort(1);
    let next = scheduler.reserve_admission(identity("old"), 3);
    assert!(next.generation > second.generation);
}

#[test]
fn duplicate_missing_and_stale_reply_do_not_commit_any_request() {
    let mut scheduler = make_scheduler(64, 16);
    for id in 1..=2 {
        let mut request = make_req(id, 4, 3);
        request.record_partial_prefix = false;
        assert!(scheduler.add_request(request));
    }
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    assert_eq!(batch.scheduled.len(), 2);
    let valid = WorkerOutput {
        outputs: vec![
            resident_result(&scheduler, dummy_output(1)),
            resident_result(&scheduler, dummy_output(2)),
        ],
    };
    scheduler.validate_output(&valid).unwrap();
    let first = resident_result(&scheduler, dummy_output(1));
    assert!(
        scheduler
            .update(WorkerOutput {
                outputs: vec![first]
            })
            .is_err()
    );
    assert!(
        scheduler
            .running
            .iter()
            .all(|request| request.num_generated_tokens() == 0)
    );
    let mut second = resident_result(&scheduler, dummy_output(2));
    second.execution.lease_id += 1;
    let first = resident_result(&scheduler, dummy_output(1));
    assert!(
        scheduler
            .update(WorkerOutput {
                outputs: vec![first, second]
            })
            .is_err()
    );
    let first = resident_result(&scheduler, dummy_output(1));
    let repeated = resident_result(&scheduler, dummy_output(1));
    assert!(
        scheduler
            .update(WorkerOutput {
                outputs: vec![first, repeated]
            })
            .is_err()
    );
    assert!(
        scheduler
            .running
            .iter()
            .all(|request| request.num_generated_tokens() == 0)
    );
}

#[test]
fn submitted_cancel_keeps_slots_until_complete_reply_set() {
    let mut scheduler = make_scheduler(64, 16);
    let mut request = make_req(1, 4, 3);
    request.record_partial_prefix = false;
    assert!(scheduler.add_request(request));
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let before = scheduler.cache_resource_counts();
    scheduler.abort(1);
    assert_eq!(scheduler.running[0].status, RequestStatus::Retiring);
    assert_eq!(scheduler.cache_resource_counts(), before);
    assert!(scheduler.schedule().scheduled.is_empty());
    let output = resident_result(&scheduler, dummy_output(1));
    scheduler
        .update(WorkerOutput {
            outputs: vec![output],
        })
        .unwrap();
    assert_eq!(scheduler.num_running(), 0);
    assert_eq!(scheduler.cache_resource_counts().1, 63);
    assert_eq!(scheduler.schedule().finished_request_ids, [1]);
}

#[test]
fn worker_replacement_keeps_admission_sequence_and_uses_one_explicit_epoch_update() {
    let mut scheduler = make_scheduler(64, 16);
    let first = scheduler.reserve_admission(identity("a"), 1);
    let mut replacement = scheduler.empty_like();
    assert_eq!(replacement.worker_epoch, scheduler.worker_epoch);
    replacement.set_worker_epoch(2).unwrap();
    assert_eq!(replacement.worker_epoch, 2);
    let second = replacement.reserve_admission(identity("a"), 2);
    assert!(second.generation > first.generation);
    assert!(!replacement.sessions.consume(&first));
    let mut request = make_req(2, 4, 3);
    request.ticket = Some(second);
    assert!(replacement.add_request(request));
    assert_eq!(
        replacement.schedule().scheduled[0].execution.worker_epoch,
        2
    );
    assert!(replacement.set_worker_epoch(3).is_err());
}

#[test]
fn resident_finish_is_successful_without_extra_forward() {
    let mut scheduler = make_scheduler(64, 16);
    let mut request = make_req(1, 4, 1);
    request.record_partial_prefix = false;
    assert!(scheduler.add_request(request));
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let output = resident_result(&scheduler, dummy_output(1));
    scheduler
        .update(WorkerOutput {
            outputs: vec![output],
        })
        .unwrap();
    assert!(!scheduler.is_finalized(1));
    scheduler.finalize_request(1, 5).unwrap();
    assert!(scheduler.is_finalized(1));
    assert_eq!(scheduler.running[0].num_computed_tokens, 4);
    assert_eq!(scheduler.running[0].token_ids.len(), 5);
    assert!(scheduler.finalize_request(1, 5).is_ok());
    assert!(scheduler.schedule().scheduled.is_empty());
    scheduler.finish_success(1).unwrap();
    assert!(scheduler.finish_success(1).is_err());
}
