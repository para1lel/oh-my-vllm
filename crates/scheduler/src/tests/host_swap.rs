use super::*;

#[test]
fn host_swap_keeps_source_until_completion_and_cancel_drains_once() {
    use crate::cache::{CacheCompletion, CacheStatus};

    let mut scheduler = make_scheduler(64, 16);
    scheduler.enable_host_cache(1 << 30);
    let mut request = make_req(1, 8, 8);
    request.record_partial_prefix = false;
    assert!(scheduler.add_request(request));
    scheduler.schedule();
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    let before = scheduler.cache_resource_counts();
    assert!(scheduler.swap_out(1));
    let operations = scheduler.cache_operations();
    assert_eq!(operations.len(), 1);
    assert_eq!(scheduler.cache_resource_counts(), before);
    assert!(scheduler.schedule().scheduled.is_empty());
    scheduler.abort(1);
    assert_eq!(scheduler.cache_resource_counts(), before);
    let done = CacheCompletion {
        transfer_id: operations[0].transfer_id,
        status: CacheStatus::Complete,
        bytes: 1024,
    };
    scheduler
        .complete_cache(std::slice::from_ref(&done))
        .unwrap();
    assert!(scheduler.cache_resource_counts().1 > before.1);
    assert!(scheduler.complete_cache(&[done]).is_err());
    assert_eq!(
        scheduler.cache_operations()[0].kind,
        crate::cache::CacheKind::Drop
    );
    assert_eq!(scheduler.num_running() + scheduler.num_waiting(), 0);
}

#[test]
fn host_swap_restores_progress_and_drafts_without_recompute() {
    use crate::cache::{CacheCompletion, CacheKind, CacheStatus};

    let mut scheduler = make_scheduler(64, 16);
    scheduler.enable_host_cache(1 << 30);
    let mut request = make_req(1, 8, 8);
    request.record_partial_prefix = false;
    assert!(scheduler.add_request(request));
    scheduler.schedule();
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    scheduler.running[0].set_drafts(vec![7, 8]);
    assert!(scheduler.swap_out(1));
    let store = scheduler.cache_operations().remove(0);
    scheduler
        .complete_cache(&[CacheCompletion {
            transfer_id: store.transfer_id,
            status: CacheStatus::Complete,
            bytes: 1024,
        }])
        .unwrap();
    assert!(scheduler.schedule().scheduled.is_empty());
    let load = scheduler.cache_operations().remove(0);
    assert_eq!(load.kind, CacheKind::Load);
    assert_eq!(load.snapshot_id, store.snapshot_id);
    assert!(scheduler.schedule().scheduled.is_empty());
    scheduler
        .complete_cache(&[CacheCompletion {
            transfer_id: load.transfer_id,
            status: CacheStatus::Complete,
            bytes: 1024,
        }])
        .unwrap();
    assert_eq!(scheduler.running[0].num_computed_tokens, 8);
    assert_eq!(scheduler.running[0].draft_token_ids, vec![7, 8]);
    assert_eq!(scheduler.schedule().scheduled[0].num_computed_tokens, 8);
}

#[test]
fn declined_store_preserves_progress_and_does_not_block_other_waiters() {
    use crate::cache::{CacheCompletion, CacheStatus};

    let mut scheduler = make_scheduler(64, 16);
    scheduler.enable_host_cache(1 << 30);
    let mut request = make_req(1, 8, 8);
    request.record_partial_prefix = false;
    assert!(scheduler.add_request(request));
    scheduler.schedule();
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert!(scheduler.swap_out(1));
    let store = scheduler.cache_operations().remove(0);
    scheduler
        .complete_cache(&[CacheCompletion {
            transfer_id: store.transfer_id,
            status: CacheStatus::Capacity,
            bytes: 0,
        }])
        .unwrap();
    assert_eq!(scheduler.running[0].num_computed_tokens, 8);
    assert!(!scheduler.swap_out(1));
    assert!(scheduler.add_request(make_req(2, 2, 1)));
    let batch = scheduler.schedule();
    assert!(
        batch
            .scheduled
            .iter()
            .any(|r| r.request_id == 1 && r.num_computed_tokens == 8)
    );
    assert!(batch.scheduled.iter().any(|r| r.request_id == 2));

    let mut pressure = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 16,
            ..Default::default()
        },
        HybridCoordinator::new(8, BS, true, 0).with_mamba_capacity(128),
    );
    let mut resident = make_req(1, 16, 4);
    resident.record_partial_prefix = false;
    assert!(pressure.add_request(resident));
    pressure.schedule();
    apply(
        &mut pressure,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    pressure.pause(1);
    for (id, length) in [(2, 16), (3, 4)] {
        let mut request = make_req(id, length, 1);
        request.token_ids.fill(id as u32 + 100);
        assert!(pressure.add_request(request));
    }
    let batch = pressure.schedule();
    assert!(batch.scheduled.iter().any(|row| row.request_id == 3));
}

#[test]
fn resident_forward_wins_before_restore_and_restore_needs_next_step_capacity() {
    use crate::cache::{CacheCompletion, CacheKind, CacheStatus};

    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 16,
            max_num_seqs: 2,
            ..Default::default()
        },
        HybridCoordinator::new(6, BS, true, 0).with_mamba_capacity(16),
    );
    scheduler.enable_host_cache(1 << 30);
    let mut second = make_req(2, 7, 8);
    second.token_ids.iter_mut().for_each(|token| *token += 100);
    assert!(scheduler.add_request(make_req(1, 7, 8)));
    assert!(scheduler.add_request(second));
    while scheduler.running.len() < 2 || scheduler.running.iter().any(|r| r.num_computed_tokens < 8)
    {
        let batch = scheduler.schedule();
        let outputs = batch
            .scheduled
            .iter()
            .map(|row| {
                let mut result = dummy_output(row.request_id);
                if row.num_computed_tokens + row.token_ids.len() < 7 {
                    result.token_ids.clear();
                }
                result
            })
            .collect();
        apply(&mut scheduler, WorkerOutput { outputs });
    }
    assert!(scheduler.swap_out(1));
    let store = scheduler.cache_operations().remove(0);
    scheduler
        .complete_cache(&[CacheCompletion {
            transfer_id: store.transfer_id,
            status: CacheStatus::Complete,
            bytes: 1024,
        }])
        .unwrap();
    let batch = scheduler.schedule();
    assert!(
        batch
            .scheduled
            .iter()
            .any(|r| r.request_id == 2 && r.num_computed_tokens == 8)
    );
    assert!(
        !scheduler
            .cache_operations()
            .iter()
            .any(|op| op.kind == CacheKind::Load)
    );
    scheduler.abort(1);
    assert_eq!(scheduler.num_waiting(), 0);
}

#[test]
fn insufficient_host_capacity_fails_one_request_instead_of_deadlocking() {
    use crate::cache::{CacheCompletion, CacheStatus};

    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 16,
            max_num_seqs: 2,
            ..Default::default()
        },
        HybridCoordinator::new(5, BS, true, 0).with_mamba_capacity(16),
    );
    scheduler.enable_host_cache(1);
    let mut first = make_req(1, 7, 8);
    first.record_partial_prefix = false;
    let mut second = make_req(2, 7, 8);
    second.record_partial_prefix = false;
    second.token_ids.iter_mut().for_each(|token| *token += 100);
    assert!(scheduler.add_request(first));
    assert!(scheduler.add_request(second));
    for _ in 0..8 {
        if scheduler.running.len() == 2
            && scheduler.running.iter().all(|r| r.num_computed_tokens == 8)
        {
            break;
        }
        let batch = scheduler.schedule();
        let outputs = batch
            .scheduled
            .iter()
            .map(|row| {
                let mut result = dummy_output(row.request_id);
                if row.num_computed_tokens + row.token_ids.len() < 7 {
                    result.token_ids.clear();
                }
                result
            })
            .collect();
        apply(&mut scheduler, WorkerOutput { outputs });
        for request in &scheduler.running {
            if request.num_computed_tokens == 8 {
                scheduler.paused.insert(request.id);
            }
        }
    }
    assert_eq!(scheduler.running.len(), 2);
    assert!(scheduler.running.iter().all(|r| r.num_computed_tokens == 8));
    scheduler.resume(1);
    scheduler.resume(2);
    let mut failed = vec![];
    for _ in 0..4 {
        let batch = scheduler.schedule();
        assert!(batch.scheduled.is_empty());
        let results = scheduler
            .cache_operations()
            .iter()
            .map(|op| CacheCompletion {
                transfer_id: op.transfer_id,
                status: CacheStatus::Capacity,
                bytes: 0,
            })
            .collect::<Vec<_>>();
        scheduler.complete_cache(&results).unwrap();
        failed.extend(scheduler.take_cache_failures());
        if !failed.is_empty() {
            break;
        }
    }
    assert_eq!(failed.len(), 1);
    assert!(failed[0].1.contains("--host-cache-bytes"));
    let next = scheduler.schedule();
    assert_eq!(next.scheduled.len(), 1);
    assert_eq!(next.scheduled[0].num_computed_tokens, 8);
}
