use super::*;
use crate::cache::{CacheCompletion, CacheKind, CacheStatus};

#[test]
fn host_prefix_restores_partial_endpoint_after_gpu_eviction() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 4096,
            max_num_seqs: 2,
            ..Default::default()
        },
        HybridCoordinator::new(5, 784, true, 0).with_mamba_capacity(128),
    );
    scheduler.enable_host_cache(1 << 30);
    cache_evicted_prefix(&mut scheduler);
    prepare_resident_decode(&mut scheduler);
    restore_prefix_with_active_decode(&mut scheduler);
    repeat_host_restore(&mut scheduler);
    validate_restored_checkpoint(&mut scheduler);
    cancel_or_defer_host_load(&mut scheduler);
}

fn cache_evicted_prefix(scheduler: &mut Scheduler<HybridCoordinator>) {
    assert!(scheduler.add_request(make_req(1, 1567, 1)));
    while scheduler.num_running() + scheduler.num_waiting() > 0 {
        let batch = scheduler.schedule();
        let row = &batch.scheduled[0];
        let mut result = dummy_output(1);
        if row.num_computed_tokens + row.token_ids.len() < 1567 {
            result.token_ids.clear();
        }
        apply(
            scheduler,
            WorkerOutput {
                outputs: vec![result],
            },
        );
    }
    assert!(scheduler.cache_operations().is_empty());
    let mut pressure = make_req(99, 2351, 1);
    pressure
        .token_ids
        .iter_mut()
        .for_each(|token| *token += 1000);
    assert!(scheduler.add_request(pressure));
    let first = scheduler.schedule();
    assert_eq!(first.scheduled[0].request_id, 99);
    apply(
        scheduler,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 99,
                token_ids: vec![],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![],
            }],
        },
    );
    assert!(scheduler.schedule().scheduled.is_empty());
    let store = scheduler.cache_operations().remove(0);
    assert_eq!(store.kind, CacheKind::Store);
    assert_eq!(store.request_id, None);
    scheduler
        .complete_cache(&[CacheCompletion {
            transfer_id: store.transfer_id,
            status: CacheStatus::Complete,
            bytes: 1024,
        }])
        .unwrap();
    scheduler.abort(99);
    assert!(scheduler.kv.reset_prefix_cache());
}

fn prepare_resident_decode(scheduler: &mut Scheduler<HybridCoordinator>) {
    let mut resident = make_req(6, 1567, 8);
    resident.record_partial_prefix = false;
    resident
        .token_ids
        .iter_mut()
        .for_each(|token| *token += 10000);
    assert!(scheduler.add_request(resident));
    while scheduler
        .running
        .front()
        .is_none_or(|r| r.num_computed_tokens < 1567)
    {
        let batch = scheduler.schedule();
        let row = &batch.scheduled[0];
        let mut result = dummy_output(6);
        if row.num_computed_tokens + row.token_ids.len() < 1567 {
            result.token_ids.clear();
        }
        apply(
            scheduler,
            WorkerOutput {
                outputs: vec![result],
            },
        );
    }
}

fn restore_prefix_with_active_decode(scheduler: &mut Scheduler<HybridCoordinator>) {
    assert!(scheduler.add_request(make_req(2, 2000, 1)));
    let turn = scheduler.schedule();
    assert_eq!(turn.scheduled.len(), 1);
    assert_eq!(turn.scheduled[0].request_id, 6);
    assert_eq!(turn.scheduled[0].num_computed_tokens, 1567);
    let load = scheduler.cache_operations().remove(0);
    assert_eq!(load.kind, CacheKind::Load);
    assert_eq!(load.request_id, None);
    apply(
        scheduler,
        WorkerOutput {
            outputs: vec![dummy_output(6)],
        },
    );
    scheduler
        .complete_cache(&[CacheCompletion {
            transfer_id: load.transfer_id,
            status: CacheStatus::Complete,
            bytes: 1024,
        }])
        .unwrap();
    let read = scheduler.kv.prefix_state_slot(2).unwrap();
    assert!(scheduler.schedule().scheduled.is_empty());
    let store = scheduler.cache_operations().remove(0);
    assert_eq!(store.kind, CacheKind::Store);
    assert_eq!(store.request_id, Some(2));
    assert!(store.state.mamba_blocks.contains(&read));
    finish_transfer(scheduler, store.transfer_id);
    while scheduler.running.iter().any(|r| r.id == 6) {
        let batch = scheduler.schedule();
        assert_eq!(batch.scheduled.len(), 1);
        assert_eq!(batch.scheduled[0].request_id, 6);
        apply(
            scheduler,
            WorkerOutput {
                outputs: vec![dummy_output(6)],
            },
        );
    }
    assert!(scheduler.finished_ids.contains(&6));
    assert!(scheduler.schedule().scheduled.is_empty());
    let load = scheduler.cache_operations().remove(0);
    assert_eq!(load.kind, CacheKind::Load);
    assert_eq!(load.request_id, Some(2));
    finish_transfer(scheduler, load.transfer_id);
    for drop in scheduler.cache_operations() {
        assert_eq!(drop.kind, CacheKind::Drop);
        finish_transfer(scheduler, drop.transfer_id);
    }
}

fn repeat_host_restore(scheduler: &mut Scheduler<HybridCoordinator>) {
    for _ in 0..2 {
        let read = scheduler.kv.prefix_state_slot(2).unwrap();
        assert!(scheduler.swap_out(2));
        let store = scheduler.cache_operations().remove(0);
        assert_eq!(store.kind, CacheKind::Store);
        assert_eq!(store.request_id, Some(2));
        assert!(store.state.mamba_blocks.contains(&read));
        scheduler
            .complete_cache(&[CacheCompletion {
                transfer_id: store.transfer_id,
                status: CacheStatus::Complete,
                bytes: 1024,
            }])
            .unwrap();
        assert_eq!(scheduler.cache_resource_counts().4, 127);
        assert!(scheduler.schedule().scheduled.is_empty());
        let load = scheduler.cache_operations().remove(0);
        assert_eq!(load.kind, CacheKind::Load);
        assert_eq!(
            load.state.mamba_blocks.len(),
            store.state.mamba_blocks.len()
        );
        scheduler
            .complete_cache(&[CacheCompletion {
                transfer_id: load.transfer_id,
                status: CacheStatus::Complete,
                bytes: 1024,
            }])
            .unwrap();
        assert_eq!(scheduler.running[0].num_computed_tokens, 1566);
        for drop in scheduler.cache_operations() {
            assert_eq!(drop.kind, CacheKind::Drop);
            scheduler
                .complete_cache(&[CacheCompletion {
                    transfer_id: drop.transfer_id,
                    status: CacheStatus::Complete,
                    bytes: 1024,
                }])
                .unwrap();
        }
    }
}

fn validate_restored_checkpoint(scheduler: &mut Scheduler<HybridCoordinator>) {
    let batch = scheduler.schedule();
    assert_eq!(batch.scheduled[0].num_computed_tokens, 1566);
    assert_eq!(batch.scheduled[0].token_ids.len(), 2);
    let read = batch.scheduled[0].state.prefix_state_slot.unwrap();
    assert_ne!(batch.scheduled[0].state.mamba_block_table[1], read);
    let before_commit = scheduler.cache_resource_counts().4;
    assert!(
        scheduler
            .update(WorkerOutput {
                outputs: vec![dummy_output(2)],
            })
            .is_err()
    );
    assert_eq!(scheduler.cache_resource_counts().4, before_commit);
    assert_eq!(scheduler.kv.prefix_state_slot(2), Some(read));
    let mut result = dummy_output(2);
    result.token_ids.clear();
    apply(
        scheduler,
        WorkerOutput {
            outputs: vec![result],
        },
    );
    assert_eq!(scheduler.cache_resource_counts().4, before_commit + 1);
    assert_eq!(scheduler.kv.prefix_state_slot(2), None);
    scheduler.abort(2);
    assert_eq!(scheduler.cache_resource_counts().4, 127);
}

fn cancel_or_defer_host_load(scheduler: &mut Scheduler<HybridCoordinator>) {
    for (id, cancel_pending, status) in [
        (3, true, CacheStatus::Complete),
        (4, false, CacheStatus::Deferred),
        (5, false, CacheStatus::Complete),
    ] {
        assert!(scheduler.kv.reset_prefix_cache());
        assert!(scheduler.add_request(make_req(id, 2000, 1)));
        assert!(scheduler.schedule().scheduled.is_empty());
        let load = scheduler.cache_operations().remove(0);
        assert_eq!(load.kind, CacheKind::Load);
        let loading = scheduler.cache_resource_counts();
        if cancel_pending {
            scheduler.abort(id);
            assert_eq!(scheduler.cache_resource_counts(), loading);
        }
        scheduler
            .complete_cache(&[CacheCompletion {
                transfer_id: load.transfer_id,
                status,
                bytes: 1024,
            }])
            .unwrap();
        if !cancel_pending {
            scheduler.abort(id);
        }
        assert_eq!(scheduler.kv.prefix_state_slot(id), None);
        assert_eq!(scheduler.cache_resource_counts().1, 4);
        assert_eq!(scheduler.cache_resource_counts().4, 127);
    }
}
