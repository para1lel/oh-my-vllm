use super::*;
use crate::policy::qwen::{QwenAssignment, QwenResidentCache};
use crate::session::{RequestIdentity, SessionId, SessionSource};

#[test]
fn one_usable_checkpoint_and_64_pages_finish_an_exact_full_capacity_tail() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 8192,
            max_num_seqs: 1,
            enable_mtp: false,
            mtp_draft_len: 0,
        },
        QwenResidentCache::new(1, 1, 8192, 8192),
    );
    let mut history = vec![7; 8190];
    add(&mut scheduler, 1, history.clone(), 1, "a");
    step(&mut scheduler, vec![8]);
    scheduler.finalize_request(1, 8191).unwrap();
    step(&mut scheduler, vec![]);
    scheduler.finish_success(1).unwrap();
    history.push(8);
    add(&mut scheduler, 2, history.clone(), 1, "a");
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert_eq!(row.num_computed_tokens, 0);
    assert_eq!(row.state.page_table.len(), 64);
    assert_eq!(
        row.state.kv_spans[0].mode,
        oh_my_vllm_kv_cache::radix::KvSpanMode::Reuse
    );
    scheduler
        .update(WorkerOutput {
            outputs: vec![reply(row, vec![9], 0)],
        })
        .unwrap();
    scheduler.finalize_request(2, 8192).unwrap();
    step(&mut scheduler, vec![]);
    scheduler.finish_success(2).unwrap();
    history.push(9);
    let mut request = Request::new(3, history, 1, vec![]);
    request.cache_domain.namespace = "a".into();
    assert_eq!(scheduler.cache_policy().find_prefix(&request).length, 8192);
}

#[test]
fn empty_mtp_lookahead_page_cannot_replace_the_last_full_prefix_page() {
    let mut scheduler = scheduler(true);
    add(&mut scheduler, 1, vec![7; 128], 8, "a");
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert_eq!(row.state.page_table.len(), 2);
    let full = row.state.page_table[0].id;
    let future = row.state.page_table[1].id;
    let mut result = reply(row, vec![8], 0);
    result.cache_result.mtp_accepted_end = Some(127);
    scheduler
        .update(WorkerOutput {
            outputs: vec![result],
        })
        .unwrap();
    let mut request = Request::new(9, vec![7; 128], 1, vec![]);
    request.cache_domain.namespace = "a".into();
    let hit = scheduler.cache_policy().find_prefix(&request);
    assert_eq!(hit.blocks.evidence.kv_contiguous_end, 128);
    assert_eq!(hit.blocks.pages.len(), 1);
    assert_eq!(hit.blocks.pages[0].id, full);
    assert_ne!(hit.blocks.pages[0].id, future);
}

pub(super) fn scheduler(mtp: bool) -> Scheduler<QwenResidentCache> {
    Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 512,
            max_num_seqs: 2,
            enable_mtp: mtp,
            mtp_draft_len: 4,
        },
        QwenResidentCache::new(8, 2, 4096, 512),
    )
}

pub(super) fn reply(
    row: &crate::ScheduledRequest<QwenAssignment>,
    tokens: Vec<u32>,
    accepted: usize,
) -> RequestOutput {
    let (_, end) = row.work.range();
    let computed = end
        - match row.work.sampling {
            SamplingMode::Verify { drafts } => drafts - accepted,
            _ => 0,
        };
    let finalizing = row.work.purpose == WorkPurpose::Finalize;
    RequestOutput {
        request_id: row.request_id,
        execution: row.execution,
        work: row.work,
        cache_result: CacheStepResult {
            execution: row.execution,
            computed_end: computed,
            accepted_end: row.accepted_end
                + if finalizing {
                    0
                } else {
                    tokens.len()
                },
            exported_checkpoints: row
                .state
                .exports
                .iter()
                .map(|export| {
                    (
                        export.end,
                        export.destination.0.slot,
                        export.destination.0.version,
                    )
                })
                .collect(),
            page_valid_ends: row
                .state
                .page_table
                .iter()
                .map(|page| {
                    let base = page.page_index * 128;
                    let at = computed.min(base + 128).max(base);
                    (
                        page.id.0.slot,
                        page.id.0.version,
                        if finalizing {
                            at
                        } else {
                            at.max(page.valid_end)
                        },
                    )
                })
                .collect(),
            mtp_accepted_end: None,
        },
        token_ids: tokens,
        num_accepted_draft_tokens: accepted,
        new_draft_token_ids: vec![],
    }
}

pub(super) fn add(
    scheduler: &mut Scheduler<QwenResidentCache>,
    id: u64,
    tokens: Vec<u32>,
    max: usize,
    namespace: &str,
) {
    let ticket = scheduler.reserve_admission(
        RequestIdentity {
            session: SessionId {
                scope: "cpu".into(),
                source: SessionSource::Explicit,
                value: "session".into(),
            },
            namespace: namespace.into(),
        },
        id,
    );
    let mut request = Request::new(id, tokens, max, vec![]);
    request.ticket = Some(ticket);
    assert!(scheduler.add_request(request));
}

pub(super) fn step(scheduler: &mut Scheduler<QwenResidentCache>, tokens: Vec<u32>) {
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    assert_eq!(batch.scheduled.len(), 1);
    scheduler
        .update(WorkerOutput {
            outputs: vec![reply(&batch.scheduled[0], tokens, 0)],
        })
        .unwrap();
}

#[test]
fn exact_prompt_hit_schedules_zero_row_head_and_validates_missing_reply() {
    let mut scheduler = scheduler(false);
    add(&mut scheduler, 1, vec![1, 2, 3], 1, "a");
    step(&mut scheduler, vec![4]);
    assert!(!scheduler.is_finalized(1));
    scheduler.finalize_request(1, 4).unwrap();
    step(&mut scheduler, vec![]);
    assert!(scheduler.is_finalized(1));
    scheduler.finish_success(1).unwrap();
    add(&mut scheduler, 2, vec![1, 2, 3, 4], 1, "a");
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert!(row.token_ids.is_empty());
    assert_eq!(row.work.action, WorkAction::HeadOnly { endpoint: 4 });
    assert_eq!(batch.num_batched_tokens, 0);
    scheduler.validate_schedule(&batch).unwrap();
    assert!(scheduler.update(WorkerOutput { outputs: vec![] }).is_err());
    let mut bad = reply(row, vec![], 0);
    assert!(
        scheduler
            .validate_output(&WorkerOutput { outputs: vec![bad] })
            .is_err()
    );
    bad = reply(row, vec![5], 0);
    scheduler
        .update(WorkerOutput { outputs: vec![bad] })
        .unwrap();
    assert_eq!(scheduler.running[0].num_computed_tokens, 4);
}

#[test]
fn finalize_is_not_visible_until_whole_export_reply_validates() {
    let mut scheduler = scheduler(false);
    add(&mut scheduler, 1, vec![7; 128], 1, "a");
    step(&mut scheduler, vec![8]);
    assert!(scheduler.finish_success(1).is_err());
    scheduler.finalize_request(1, 129).unwrap();
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert_eq!(row.work.purpose, WorkPurpose::Finalize);
    assert_eq!(row.token_ids, [8]);
    let mut bad = reply(row, vec![], 0);
    bad.cache_result.exported_checkpoints.clear();
    assert!(
        scheduler
            .update(WorkerOutput { outputs: vec![bad] })
            .is_err()
    );
    assert!(!scheduler.is_finalized(1));
    scheduler
        .update(WorkerOutput {
            outputs: vec![reply(row, vec![], 0)],
        })
        .unwrap();
    assert!(scheduler.is_finalized(1));
    scheduler.finish_success(1).unwrap();
    assert_eq!(
        scheduler
            .cache_resource_counts()
            .available_private_checkpoints,
        12
    );
}

#[test]
fn already_forwarded_stop_endpoint_uses_zero_row_finalize() {
    let mut scheduler = scheduler(false);
    add(&mut scheduler, 1, vec![1, 2, 3], 8, "a");
    step(&mut scheduler, vec![4]);
    step(&mut scheduler, vec![5]);
    scheduler.finalize_request(1, 4).unwrap();
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert_eq!(row.work.action, WorkAction::ExportOnly { endpoint: 4 });
    assert_eq!(row.work.sampling, SamplingMode::None);
    assert!(row.token_ids.is_empty());
    scheduler
        .update(WorkerOutput {
            outputs: vec![reply(row, vec![], 0)],
        })
        .unwrap();
    scheduler.finish_success(1).unwrap();
}

#[test]
fn speculative_accepted_endpoint_before_forwarded_endpoint_restores_without_sampling() {
    let mut scheduler = scheduler(true);
    add(&mut scheduler, 1, vec![1, 2, 3], 8, "a");
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let mut first = reply(&batch.scheduled[0], vec![4], 0);
    first.new_draft_token_ids = vec![5, 6, 7, 8];
    first.cache_result.mtp_accepted_end = Some(2);
    scheduler
        .update(WorkerOutput {
            outputs: vec![first],
        })
        .unwrap();
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert_eq!(row.work.sampling, SamplingMode::Verify { drafts: 4 });
    let mut verified = reply(row, vec![5, 6, 9], 2);
    verified.cache_result.mtp_accepted_end = Some(5);
    scheduler
        .update(WorkerOutput {
            outputs: vec![verified],
        })
        .unwrap();
    assert_eq!(scheduler.running[0].num_computed_tokens, 6);
    scheduler.finalize_request(1, 5).unwrap();
    let batch = scheduler.schedule();
    scheduler.mark_submitted(&batch).unwrap();
    let row = &batch.scheduled[0];
    assert_eq!(row.work.action, WorkAction::ExportOnly { endpoint: 5 });
    assert!(matches!(
        row.state.endpoint_source,
        oh_my_vllm_kv_cache::radix::EndpointSource::SavedHidden { end: 5, .. }
    ));
    let mut finalized = reply(row, vec![], 0);
    finalized.cache_result.mtp_accepted_end = Some(4);
    scheduler
        .update(WorkerOutput {
            outputs: vec![finalized],
        })
        .unwrap();
    scheduler.finish_success(1).unwrap();
}
