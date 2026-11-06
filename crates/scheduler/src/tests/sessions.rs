use super::radix::{add, scheduler, step};
use super::*;
use crate::session::{RequestIdentity, SessionId, SessionSource};

fn succeed(
    scheduler: &mut Scheduler<crate::policy::qwen::QwenResidentCache>,
    id: u64,
    prompt: Vec<u32>,
    token: u32,
    namespace: &str,
) {
    let end = prompt.len() + 1;
    add(scheduler, id, prompt, 1, namespace);
    step(scheduler, vec![token]);
    scheduler.finalize_request(id, end).unwrap();
    step(scheduler, vec![]);
    scheduler.finish_success(id).unwrap();
}

fn hit(
    scheduler: &Scheduler<crate::policy::qwen::QwenResidentCache>,
    tokens: Vec<u32>,
    namespace: &str,
) -> usize {
    let mut request = Request::new(100, tokens, 1, vec![]);
    request.cache_domain.namespace = namespace.into();
    scheduler.cache_policy().find_prefix(&request).length
}

#[test]
fn newer_prepare_failure_preserves_previous_success_and_disqualifies_old_late_finish() {
    let mut scheduler = scheduler(false);
    succeed(&mut scheduler, 1, vec![1, 2], 3, "a");
    add(&mut scheduler, 2, vec![4, 5], 1, "a");
    let newer = scheduler.reserve_admission(
        RequestIdentity {
            session: SessionId {
                scope: "cpu".into(),
                source: SessionSource::Explicit,
                value: "session".into(),
            },
            namespace: "b".into(),
        },
        3,
    );
    scheduler.release_admission(&newer);
    step(&mut scheduler, vec![6]);
    scheduler.finalize_request(2, 3).unwrap();
    step(&mut scheduler, vec![]);
    scheduler.finish_success(2).unwrap();
    assert_eq!(hit(&scheduler, vec![1, 2, 3], "a"), 3);
    assert_eq!(hit(&scheduler, vec![4, 5, 6], "a"), 0);
}

#[test]
fn one_session_success_in_new_namespace_replaces_the_previous_tail_role() {
    let mut scheduler = scheduler(false);
    succeed(&mut scheduler, 1, vec![1, 2], 3, "a");
    assert_eq!(hit(&scheduler, vec![1, 2, 3], "a"), 3);
    succeed(&mut scheduler, 2, vec![4, 5], 6, "b");
    assert_eq!(hit(&scheduler, vec![4, 5, 6], "b"), 3);
    assert_eq!(hit(&scheduler, vec![4, 5, 6], "a"), 0);
    assert_eq!(hit(&scheduler, vec![1, 2, 3], "a"), 0);
    assert_eq!(hit(&scheduler, vec![1, 2, 3], "b"), 0);
}

#[test]
fn cancelling_before_submit_reclaims_the_entire_unsubmitted_lease() {
    let mut scheduler = scheduler(false);
    add(&mut scheduler, 1, vec![1, 2, 3], 8, "a");
    let batch = scheduler.schedule();
    assert_eq!(batch.scheduled.len(), 1);
    scheduler.abort(1);
    assert_eq!(scheduler.num_running(), 0);
    let resources = scheduler.cache_resource_counts();
    assert_eq!(resources.available_pages, resources.usable_pages);
    assert_eq!(
        resources.available_checkpoints,
        resources.usable_checkpoints
    );
    assert_eq!(
        resources.available_private_checkpoints,
        resources.private_checkpoints
    );
    assert!(scheduler.mark_submitted(&batch).is_err());
}

#[test]
fn cancelling_finalize_retires_without_publishing_success() {
    let mut scheduler = scheduler(false);
    succeed(&mut scheduler, 1, vec![1, 2], 3, "a");
    add(&mut scheduler, 2, vec![4, 5], 1, "a");
    step(&mut scheduler, vec![6]);
    scheduler.finalize_request(2, 3).unwrap();
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    let result = super::radix::reply(row, vec![], 0);
    scheduler.abort(2);
    assert!(scheduler.num_running() > 0);
    scheduler
        .update(WorkerOutput {
            outputs: vec![result],
        })
        .unwrap();
    assert_eq!(scheduler.num_running(), 0);
    assert_eq!(hit(&scheduler, vec![1, 2, 3], "a"), 3);
    assert_eq!(hit(&scheduler, vec![4, 5, 6], "a"), 0);
}
