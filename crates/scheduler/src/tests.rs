use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;

use crate::{
    Scheduler, SchedulerConfig,
    output::{RequestOutput, WorkerOutput},
    policy::CachePolicy,
    request::Request,
};

const BS: usize = 4;

#[test]
fn host_swap_keeps_source_until_completion_and_cancel_drains_once() {
    use crate::cache::{CacheCompletion, CacheStatus};

    let mut scheduler = make_scheduler(64, 16);
    scheduler.enable_host_cache(1 << 30);
    assert!(scheduler.add_request(make_req(1, 8, 8)));
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
    assert!(scheduler.add_request(make_req(1, 8, 8)));
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
fn host_prefix_restores_partial_endpoint_after_gpu_eviction() {
    use crate::cache::{CacheCompletion, CacheKind, CacheStatus};

    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 4096,
            max_num_seqs: 2,
            ..Default::default()
        },
        HybridCoordinator::new(5, 784, true, 0).with_mamba_capacity(128),
    );
    scheduler.enable_host_cache(1 << 30);
    assert!(scheduler.add_request(make_req(1, 1567, 1)));
    while scheduler.num_running() + scheduler.num_waiting() > 0 {
        let batch = scheduler.schedule();
        let row = &batch.scheduled[0];
        let mut result = dummy_output(1);
        if row.num_computed_tokens + row.token_ids.len() < 1567 {
            result.token_ids.clear();
        }
        apply(
            &mut scheduler,
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
        &mut scheduler,
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
            &mut scheduler,
            WorkerOutput {
                outputs: vec![result],
            },
        );
    }
    assert!(scheduler.add_request(make_req(2, 2000, 1)));
    let turn = scheduler.schedule();
    assert_eq!(turn.scheduled.len(), 1);
    assert_eq!(turn.scheduled[0].request_id, 6);
    assert_eq!(turn.scheduled[0].num_computed_tokens, 1567);
    let load = scheduler.cache_operations().remove(0);
    assert_eq!(load.kind, CacheKind::Load);
    assert_eq!(load.request_id, None);
    apply(
        &mut scheduler,
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
    finish_transfer(&mut scheduler, store.transfer_id);
    while scheduler.running.iter().any(|r| r.id == 6) {
        let batch = scheduler.schedule();
        assert_eq!(batch.scheduled.len(), 1);
        assert_eq!(batch.scheduled[0].request_id, 6);
        apply(
            &mut scheduler,
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
    finish_transfer(&mut scheduler, load.transfer_id);
    for drop in scheduler.cache_operations() {
        assert_eq!(drop.kind, CacheKind::Drop);
        finish_transfer(&mut scheduler, drop.transfer_id);
    }
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
        &mut scheduler,
        WorkerOutput {
            outputs: vec![result],
        },
    );
    assert_eq!(scheduler.cache_resource_counts().4, before_commit + 1);
    assert_eq!(scheduler.kv.prefix_state_slot(2), None);
    scheduler.abort(2);
    assert_eq!(scheduler.cache_resource_counts().4, 127);

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

#[test]
fn declined_store_preserves_progress_and_does_not_block_other_waiters() {
    use crate::cache::{CacheCompletion, CacheStatus};

    let mut scheduler = make_scheduler(64, 16);
    scheduler.enable_host_cache(1 << 30);
    assert!(scheduler.add_request(make_req(1, 8, 8)));
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
    assert!(pressure.add_request(make_req(1, 16, 4)));
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

fn make_coord(num_blocks: u32) -> HybridCoordinator {
    HybridCoordinator::new(num_blocks, BS, true, 0)
}

fn make_scheduler(num_blocks: u32, max_tokens_per_step: usize) -> Scheduler<HybridCoordinator> {
    let config = SchedulerConfig {
        max_num_batched_tokens: max_tokens_per_step,
        max_num_seqs: 64,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    Scheduler::new(config, make_coord(num_blocks))
}

fn make_req(id: u64, num_tokens: usize, max_output: usize) -> Request {
    let tokens: Vec<u32> = (0..num_tokens as u32).collect();
    Request::new(id, tokens, max_output, Vec::new())
}

fn dummy_output(rid: u64) -> RequestOutput {
    RequestOutput {
        request_id: rid,
        token_ids: vec![42],
        num_accepted_draft_tokens: 0,
        new_draft_token_ids: Vec::new(),
    }
}

fn finish_transfer(scheduler: &mut Scheduler<HybridCoordinator>, transfer_id: u64) {
    scheduler
        .complete_cache(&[crate::cache::CacheCompletion {
            transfer_id,
            status: crate::cache::CacheStatus::Complete,
            bytes: 1024,
        }])
        .unwrap();
}

fn apply(scheduler: &mut Scheduler<HybridCoordinator>, output: WorkerOutput) -> WorkerOutput {
    scheduler.update(output).expect("test output must be valid")
}

fn production_long_scheduler(max_seqs: usize) -> Scheduler<HybridCoordinator> {
    Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 32768,
            max_num_seqs: max_seqs,
            enable_mtp: false,
            mtp_draft_len: 0,
        },
        HybridCoordinator::new(1400, 784, true, 0).with_mamba_capacity(128),
    )
}

fn unique_long_request(id: u64) -> Request {
    let mut request = make_req(id, 131072, 1);
    for token in &mut request.token_ids {
        *token += id as u32 * 1000;
    }
    request
}

#[test]
fn media_cache_salts_isolate_identical_placeholder_tokens() {
    use oh_my_vllm_kv_cache::hash::BlockHash;

    for length in [3, 9, 10] {
        let mut scheduler = make_scheduler(64, 4);
        let mut first = make_req(1, length, 1);
        first.cache_salt = Some(BlockHash(11));
        assert!(scheduler.add_request(first));
        loop {
            let batch = scheduler.schedule();
            let row = &batch.scheduled[0];
            let final_chunk = row.num_computed_tokens + row.token_ids.len() == length;
            let mut result = dummy_output(1);
            if !final_chunk {
                result.token_ids.clear();
            }
            apply(
                &mut scheduler,
                WorkerOutput {
                    outputs: vec![result],
                },
            );
            if final_chunk {
                break;
            }
        }
        for (id, salt, expected_hit) in [
            (2, Some(BlockHash(22)), 0),
            (3, None, 0),
            (4, Some(BlockHash(11)), length - 1),
        ] {
            let mut request = make_req(id, length, 1);
            request.cache_salt = salt;
            assert!(scheduler.add_request(request));
            assert_eq!(scheduler.schedule().cache_hit_tokens, expected_hit);
            scheduler.abort(id);
        }
    }
}

#[test]
fn malformed_output_does_not_publish_pending_prefix_and_keeps_valid_history() {
    for (length, completed) in [(9, 4), (10, 8)] {
        let mut sched = make_scheduler(64, 4);
        assert!(sched.add_request(make_req(1, length, 1)));
        for _ in 0..completed / 4 {
            sched.schedule();
            apply(
                &mut sched,
                WorkerOutput {
                    outputs: vec![RequestOutput {
                        token_ids: vec![],
                        ..dummy_output(1)
                    }],
                },
            );
        }
        sched.schedule();
        assert!(
            sched
                .update(WorkerOutput {
                    outputs: vec![dummy_output(1)]
                })
                .is_err()
        );
        sched.abort(1);
        assert!(sched.add_request(make_req(2, length, 1)));
        let batch = sched.schedule();
        assert_eq!(batch.scheduled[0].num_computed_tokens, completed);
        assert_eq!(batch.cache_hit_tokens, completed);
    }
}

#[test]
fn preemption_resends_accepted_history_without_drafts() {
    let mut sched = make_scheduler(9, 128);
    for id in 1..=2 {
        let mut request = make_req(id, 8, 10);
        request.record_partial_prefix = false;
        sched.add_request(request);
    }
    let first = sched.schedule();
    for request in &first.scheduled {
        assert_eq!(request.prefill_token_ids, Some((0..8).collect()));
    }
    let mut second_output = dummy_output(2);
    second_output.new_draft_token_ids = vec![90, 91];
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), second_output],
        },
    );
    let next = sched.schedule();
    assert_eq!(next.preempted_request_ids, vec![2]);
    assert!(next.scheduled[0].prefill_token_ids.is_none());
    let resumed = next
        .scheduled
        .iter()
        .find(|request| request.request_id == 2)
        .unwrap();
    assert_eq!(resumed.num_computed_tokens, 8);
    assert_eq!(
        resumed.prefill_token_ids,
        Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 42])
    );
    assert!(sched.running.back().unwrap().draft_token_ids.is_empty());
    let readmitted = sched.running.back().unwrap();
    assert!(!readmitted.record_partial_prefix);
    assert_eq!(
        readmitted.block_hashes,
        sched.kv.compute_block_hashes(&readmitted.token_ids)
    );
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );
    sched.abort(1);
    let running = sched.schedule();
    assert_eq!(running.scheduled.len(), 1);
    let request = &running.scheduled[0];
    assert_eq!(request.request_id, 2);
    assert!(request.prefill_token_ids.is_none());
}

#[test]
fn mtp_acceptance_preserves_tokens_and_rejects_only_scheduled_drafts() {
    let mut kv = make_coord(64);
    kv.set_speculative_blocks(2);
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 16,
            max_num_seqs: 4,
            enable_mtp: true,
            mtp_draft_len: 2,
        },
        kv,
    );
    scheduler.add_request(make_req(1, 4, 10));
    scheduler.schedule();
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![10, 11],
            }],
        },
    );
    let with_drafts = scheduler.running.front().unwrap();
    assert_eq!(with_drafts.draft_token_ids, vec![10, 11]);
    assert_eq!(
        with_drafts.block_hashes,
        scheduler.kv.compute_block_hashes(&with_drafts.token_ids)
    );
    assert_eq!(scheduler.schedule().num_batched_tokens, 3);
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![10, 99],
                num_accepted_draft_tokens: 1,
                new_draft_token_ids: vec![],
            }],
        },
    );
    let request = scheduler.running.front().unwrap();
    assert_eq!(request.num_computed_tokens, 6);
    assert_eq!(&request.token_ids[4..], &[42, 10, 99]);
    assert!(request.draft_token_ids.is_empty());
    assert_eq!(
        request.block_hashes,
        scheduler.kv.compute_block_hashes(&request.token_ids)
    );
    assert_eq!(scheduler.schedule().scheduled[0].token_ids, vec![99]);
    let mut ending = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 4,
            max_num_seqs: 1,
            enable_mtp: true,
            mtp_draft_len: 4,
        },
        HybridCoordinator::new(2, 4, true, 0).with_mamba_capacity(8),
    );
    assert!(ending.add_request(make_req(2, 4, 1)));
    assert_eq!(ending.schedule().num_batched_tokens, 4);

    let mut concurrent = make_scheduler(1024, 16);
    concurrent.config.enable_mtp = true;
    concurrent.config.mtp_draft_len = 2;
    for id in 1..=2 {
        assert!(concurrent.add_request(make_req(id, 4, 8)));
    }
    concurrent.schedule();
    apply(
        &mut concurrent,
        WorkerOutput {
            outputs: (1..=2)
                .map(|id| RequestOutput {
                    new_draft_token_ids: if id == 1 {
                        vec![10, 11]
                    } else {
                        vec![]
                    },
                    ..dummy_output(id)
                })
                .collect(),
        },
    );
    concurrent.config.max_num_batched_tokens = 2;
    let batch = concurrent.schedule();
    assert_eq!(
        batch
            .scheduled
            .iter()
            .map(|row| (row.request_id, row.token_ids.len()))
            .collect::<Vec<_>>(),
        vec![(1, 1), (2, 1)]
    );
    apply(
        &mut concurrent,
        WorkerOutput {
            outputs: vec![
                RequestOutput {
                    new_draft_token_ids: vec![10, 11],
                    ..dummy_output(1)
                },
                dummy_output(2),
            ],
        },
    );
    concurrent.pause(2);
    concurrent.config.max_num_batched_tokens = 785;
    let mut prefill = make_req(3, 1000, 1);
    prefill.token_ids.fill(999);
    assert!(concurrent.add_request(prefill));
    let batch = concurrent.schedule();
    assert_eq!(
        batch
            .scheduled
            .iter()
            .map(|row| (row.request_id, row.token_ids.len()))
            .collect::<Vec<_>>(),
        vec![(1, 1), (3, 784)]
    );

    let mut residual = make_scheduler(64, 16);
    residual.config.enable_mtp = true;
    residual.config.mtp_draft_len = 4;
    assert!(residual.add_request(make_req(1, 4, 8)));
    residual.schedule();
    apply(
        &mut residual,
        WorkerOutput {
            outputs: vec![RequestOutput {
                new_draft_token_ids: vec![10, 11, 12, 13],
                ..dummy_output(1)
            }],
        },
    );
    let mut tail = make_req(2, 3, 1);
    tail.record_partial_prefix = false;
    assert!(residual.add_request(tail));
    residual.config.max_num_batched_tokens = 5;
    let batch = residual.schedule();
    assert_eq!(
        batch
            .scheduled
            .iter()
            .map(|row| (row.request_id, row.token_ids.len()))
            .collect::<Vec<_>>(),
        vec![(1, 2), (2, 3)]
    );
}

#[test]
fn prefill_batches_preserve_intervals_and_make_progress() {
    for budget in [1, 3, 783, 784, 785, 32768] {
        for source in ["cold", "prefix", "host", "history"] {
            let mut scheduler = production_long_scheduler(4);
            scheduler.config.max_num_batched_tokens = budget;
            let requests = [317, 1234, 1569].map(|length| make_req(length as u64, length, 4));
            if source == "prefix" {
                for request in &requests {
                    let mut seed = request.clone();
                    seed.id += 10000;
                    seed.max_tokens = 1;
                    assert!(scheduler.add_request(seed));
                    finish_prefills(&mut scheduler);
                }
            }
            if source == "host" {
                scheduler.enable_host_cache(1 << 30);
            }
            for request in requests {
                assert!(scheduler.add_request(request));
            }
            if source == "prefix" {
                assert!(scheduler.waiting.iter().any(|request| {
                    !scheduler.kv.find_prefix(request).length.is_multiple_of(784)
                }));
            }
            if source == "host" {
                restore_partial_prefills(&mut scheduler);
            }
            if source == "history" {
                preempt_generated_history(&mut scheduler);
            }
            let history_prefill = finish_prefills(&mut scheduler);
            if source == "history" && budget == 1 {
                assert!(
                    history_prefill,
                    "recompute must reach beyond the original prompt"
                );
            }
            for operation in scheduler.cache_operations() {
                finish_transfer(&mut scheduler, operation.transfer_id);
            }
            assert!(!scheduler.cache_pending());
        }
    }
}

fn preempt_generated_history(scheduler: &mut Scheduler<HybridCoordinator>) {
    let budget = scheduler.config.max_num_batched_tokens;
    scheduler.config.max_num_batched_tokens = 32768;
    for _ in 0..32 {
        if scheduler.num_waiting() == 0
            && scheduler
                .running
                .iter()
                .all(|request| request.num_generated_tokens() == 2)
        {
            break;
        }
        let batch = scheduler.schedule();
        let outputs = batch
            .scheduled
            .iter()
            .map(|row| {
                let request = scheduler
                    .running
                    .iter()
                    .find(|request| request.id == row.request_id)
                    .unwrap();
                let mut output = dummy_output(row.request_id);
                if row.num_computed_tokens + row.token_ids.len() < request.token_ids.len() {
                    output.token_ids.clear();
                }
                output
            })
            .collect();
        apply(scheduler, WorkerOutput { outputs });
        for request in &scheduler.running {
            if request.num_generated_tokens() == 2 {
                scheduler.paused.insert(request.id);
            }
        }
    }
    let mut output = crate::SchedulerOutput::default();
    while let Some(request) = scheduler.running.pop_front() {
        assert_eq!(request.num_generated_tokens(), 2);
        assert_eq!(request.num_computed_tokens, request.token_ids.len() - 1);
        scheduler.resume(request.id);
        scheduler.preempt_request(request, &mut output);
    }
    assert_eq!(output.preempted_request_ids.len(), 3);
    assert!(scheduler.kv.reset_prefix_cache());
    scheduler.config.max_num_batched_tokens = budget;
}

fn restore_partial_prefills(scheduler: &mut Scheduler<HybridCoordinator>) {
    let budget = scheduler.config.max_num_batched_tokens;
    scheduler.config.max_num_batched_tokens = 3;
    for _ in 0..3 {
        let batch = scheduler.schedule();
        assert_eq!(batch.scheduled.len(), 1);
        assert_eq!(batch.scheduled[0].num_computed_tokens, 0);
        apply(
            scheduler,
            WorkerOutput {
                outputs: vec![RequestOutput {
                    token_ids: vec![],
                    ..dummy_output(batch.scheduled[0].request_id)
                }],
            },
        );
    }
    for id in [317, 1234, 1569] {
        assert!(scheduler.swap_out(id));
        let store = scheduler.cache_operations().remove(0);
        finish_transfer(scheduler, store.transfer_id);
    }
    scheduler.config.max_num_batched_tokens = budget;
    assert!(scheduler.schedule().scheduled.is_empty());
    let loads = scheduler.cache_operations();
    assert_eq!(loads.len(), 3);
    for load in loads {
        finish_transfer(scheduler, load.transfer_id);
    }
    assert_eq!(scheduler.num_running(), 3);
    assert!(
        scheduler
            .running
            .iter()
            .all(|request| request.num_computed_tokens == 3)
    );
}

fn finish_prefills(scheduler: &mut Scheduler<HybridCoordinator>) -> bool {
    let mut history_prefill = false;
    for _ in 1..=10000 {
        if scheduler.num_waiting() + scheduler.num_running() == 0 {
            return history_prefill;
        }
        let batch = scheduler.schedule();
        assert!(
            !batch.scheduled.is_empty(),
            "prefill must make progress: budget {}, running {}, waiting {}",
            scheduler.config.max_num_batched_tokens,
            scheduler.num_running(),
            scheduler.num_waiting()
        );
        assert!(batch.num_batched_tokens <= scheduler.config.max_num_batched_tokens);
        let outputs = batch
            .scheduled
            .iter()
            .map(|row| {
                let request = scheduler
                    .running
                    .iter()
                    .find(|request| request.id == row.request_id)
                    .unwrap();
                let prefill_end = request.prompt_len.max(request.token_ids.len() - 1);
                if row.num_computed_tokens < prefill_end {
                    history_prefill |= row.num_computed_tokens >= request.prompt_len;
                }
                assert_eq!(row.num_computed_tokens, request.num_computed_tokens);
                assert_eq!(
                    row.token_ids,
                    request
                        .token_ids
                        .iter()
                        .chain(&request.draft_token_ids)
                        .skip(row.num_computed_tokens)
                        .take(row.token_ids.len())
                        .copied()
                        .collect::<Vec<_>>()
                );
                let mut output = dummy_output(row.request_id);
                if row.num_computed_tokens + row.token_ids.len() < request.token_ids.len() {
                    output.token_ids.clear();
                }
                output
            })
            .collect();
        apply(scheduler, WorkerOutput { outputs });
    }
    panic!("prefill requests did not finish");
}

#[test]
fn short_prefill_tails_cannot_take_a_later_active_decode_budget() {
    let mut scheduler = production_long_scheduler(6);
    let requests = [
        unique_long_request(1),
        unique_long_request(2),
        make_req(3, 1408, 1),
        make_req(4, 1408, 1),
        make_req(5, 944, 1),
        make_req(6, 784, 8),
    ];
    scheduler.kv.new_step_starts();
    for request in requests {
        assert!(scheduler.add_request(request));
        let mut request = scheduler.waiting.pop_front().unwrap();
        assert!(
            scheduler
                .kv
                .allocate_slots(&request, (Vec::new(), Vec::new()), 0, 784, 0, true)
                .is_some()
        );
        request.status = crate::RequestStatus::Running;
        request.num_in_flight_tokens = 784;
        scheduler.running.push_back(request);
    }
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: (1..=6)
                .map(|id| RequestOutput {
                    request_id: id,
                    token_ids: if id == 6 {
                        vec![42]
                    } else {
                        Vec::new()
                    },
                    num_accepted_draft_tokens: 0,
                    new_draft_token_ids: Vec::new(),
                })
                .collect(),
        },
    );
    let step = scheduler.schedule();
    assert!(step.num_batched_tokens <= 32768);
    let decode = step
        .scheduled
        .iter()
        .find(|row| row.request_id == 6)
        .unwrap();
    assert_eq!(decode.token_ids.len(), 1);
    assert_eq!(
        step.scheduled
            .iter()
            .map(|row| row.request_id)
            .collect::<Vec<_>>(),
        vec![6, 1]
    );
    assert_eq!(step.num_batched_tokens, 32768);
    assert!(step.preempted_request_ids.is_empty());
}

#[test]
fn prefill_budget_prefers_blocks_then_tokens_then_fewer_suffixes() {
    for (budget, lengths, expected) in [
        (32768, vec![131072], vec![(1, 32768)]),
        (1000, vec![784, 317], vec![(1, 784), (2, 216)]),
    ] {
        let mut scheduler = production_long_scheduler(4);
        scheduler.config.max_num_batched_tokens = budget;
        for (index, length) in lengths.into_iter().enumerate() {
            let mut request = make_req(index as u64 + 1, length, 1);
            request.token_ids.fill(index as u32 + 100);
            request.record_partial_prefix = false;
            assert!(scheduler.add_request(request));
        }
        let batch = scheduler.schedule();
        assert_eq!(
            batch
                .scheduled
                .iter()
                .map(|row| (row.request_id, row.token_ids.len()))
                .collect::<Vec<_>>(),
            expected
        );
    }
    for (length, record_partial, fragments) in [
        (317, true, vec![316, 1]),
        (1234, true, vec![784, 449, 1]),
        (32768, true, vec![32144, 623, 1]),
        (178, false, vec![178]),
        (1234, false, vec![784, 450]),
        (32768, false, vec![32144, 624]),
    ] {
        let mut scheduler = production_long_scheduler(1);
        let mut request = make_req(1, length, 1);
        request.record_partial_prefix = record_partial;
        assert!(scheduler.add_request(request));
        for count in fragments {
            let batch = scheduler.schedule();
            assert_eq!(batch.scheduled.len(), 1);
            assert_eq!(batch.scheduled[0].token_ids.len(), count);
            let final_chunk = batch.scheduled[0].num_computed_tokens + count == length;
            apply(
                &mut scheduler,
                WorkerOutput {
                    outputs: vec![RequestOutput {
                        token_ids: if final_chunk {
                            vec![42]
                        } else {
                            vec![]
                        },
                        ..dummy_output(1)
                    }],
                },
            );
        }
    }
    for (budget, lengths, expected) in [
        (24, [20, 100, 100], vec![20, 4]),
        (24, [4, 100, 100], vec![4, 20]),
        (3, [20, 100, 100], vec![3]),
    ] {
        let mut scheduler = make_scheduler(128, budget);
        for (id, length) in lengths.into_iter().enumerate() {
            let mut request = make_req(id as u64 + 1, length, 1);
            for token in &mut request.token_ids {
                *token += (id as u32 + 1) * 1000;
            }
            assert!(scheduler.add_request(request));
        }
        let step = scheduler.schedule();
        assert_eq!(step.num_batched_tokens, budget);
        assert_eq!(
            step.scheduled
                .iter()
                .map(|row| row.token_ids.len())
                .collect::<Vec<_>>(),
            expected
        );
    }
    for length in [1568, 131072] {
        let mut scheduler = production_long_scheduler(2);
        scheduler.config.max_num_batched_tokens = 784;
        assert!(scheduler.add_request(make_req(1, 317, 1)));
        let mut long = unique_long_request(2);
        long.token_ids.truncate(length);
        long.prompt_len = length;
        assert!(scheduler.add_request(long));
        let batch = scheduler.schedule();
        assert_eq!(batch.scheduled.len(), 1);
        assert_eq!(batch.scheduled[0].request_id, 2);
        assert_eq!(batch.num_batched_tokens, 784);
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![RequestOutput {
                    token_ids: vec![],
                    ..dummy_output(2)
                }],
            },
        );
        finish_prefills(&mut scheduler);
    }
    for budget in [1, 784] {
        let mut scheduler = production_long_scheduler(4);
        scheduler.config.max_num_batched_tokens = budget;
        for id in 1..=4 {
            assert!(scheduler.add_request(unique_long_request(id)));
        }
        for id in 1..=4 {
            let step = scheduler.schedule();
            assert_eq!(step.scheduled.len(), 1);
            assert_eq!(step.scheduled[0].request_id, id);
            assert_eq!(step.num_batched_tokens, budget);
            apply(
                &mut scheduler,
                WorkerOutput {
                    outputs: vec![RequestOutput {
                        token_ids: Vec::new(),
                        ..dummy_output(id)
                    }],
                },
            );
        }
    }
    for (budget, lengths, expected) in [
        (767, vec![317, 450], vec![(1, 316), (2, 449)]),
        (800, vec![401, 401], vec![(1, 400), (2, 400)]),
        (784, vec![317, 450, 1000], vec![(3, 784)]),
        (2353, vec![784, 3136], vec![(1, 784), (2, 1569)]),
        (100, vec![51, 51, 101], vec![(3, 100)]),
    ] {
        let mut scheduler = production_long_scheduler(if budget == 784 {
            1
        } else {
            4
        });
        scheduler.config.max_num_batched_tokens = budget;
        for (index, length) in lengths.into_iter().enumerate() {
            let mut request = make_req(index as u64 + 1, length, 1);
            for token in &mut request.token_ids {
                *token += index as u32 * 1000;
            }
            assert!(scheduler.add_request(request));
        }
        let batch = scheduler.schedule();
        assert_eq!(
            batch
                .scheduled
                .iter()
                .map(|row| (row.request_id, row.token_ids.len()))
                .collect::<Vec<_>>(),
            expected
        );
    }
    for budget in 1..=12 {
        for slots in 1..=3 {
            let mut scheduler = make_scheduler(128, budget);
            scheduler.config.max_num_seqs = slots;
            for (index, length) in [3, 7, 11].into_iter().enumerate() {
                let mut request = make_req(index as u64 + 1, length, 1);
                request.token_ids.fill(index as u32 + 100);
                assert!(scheduler.add_request(request));
            }
            let choices = scheduler
                .waiting
                .iter()
                .map(|request| {
                    let mut counts = (0..=budget.min(request.token_ids.len()))
                        .map(|count| scheduler.kv.align_prefill(request, 0, count, budget))
                        .collect::<Vec<_>>();
                    counts.sort_unstable();
                    counts.dedup();
                    counts
                })
                .collect::<Vec<_>>();
            let score = |counts: &[usize]| {
                (
                    counts.iter().map(|count| count / BS).sum::<usize>(),
                    counts.iter().sum::<usize>(),
                    std::cmp::Reverse(
                        counts
                            .iter()
                            .filter(|count| !count.is_multiple_of(BS))
                            .count(),
                    ),
                )
            };
            let mut expected = (score(&[0; 3]), [0; 3]);
            for &first in &choices[0] {
                for &second in &choices[1] {
                    for &third in &choices[2] {
                        let counts = [first, second, third];
                        if counts.iter().sum::<usize>() <= budget
                            && counts.iter().filter(|count| **count > 0).count() <= slots
                        {
                            expected = expected.max((score(&counts), counts));
                        }
                    }
                }
            }
            let batch = scheduler.schedule();
            let actual = [1, 2, 3].map(|id| {
                batch
                    .scheduled
                    .iter()
                    .find(|row| row.request_id == id)
                    .map_or(0, |row| row.token_ids.len())
            });
            assert_eq!(
                (score(&actual), actual),
                expected,
                "budget {budget}, slots {slots}"
            );
        }
    }
    let mut scheduler = production_long_scheduler(4);
    for budget in [3, 783, 784, 785, 1568] {
        scheduler.config.max_num_batched_tokens = budget;
        for length in [317, 784, 1234] {
            let mut request = make_req(1, length, 4);
            request.token_ids.resize(1600, 42);
            for start in [0, 3, 316, 317, 783, 784, 1233] {
                let limit = budget.min(request.token_ids.len() - start);
                let mut actual = scheduler
                    .kv
                    .prefill_ranges(&request, start, limit, budget)
                    .into_iter()
                    .flat_map(|(first, last)| first..=last)
                    .collect::<Vec<_>>();
                let mut expected = (1..=limit)
                    .map(|count| scheduler.kv.align_prefill(&request, start, count, budget))
                    .collect::<Vec<_>>();
                for counts in [&mut actual, &mut expected] {
                    counts.sort_unstable();
                    counts.dedup();
                }
                assert_eq!(
                    actual, expected,
                    "length {length}, start {start}, budget {budget}"
                );
            }
        }
    }
}

#[test]
fn fair_prefill_admission_reserves_the_next_private_state() {
    for (parallel, admitted) in [(1, true), (2, false)] {
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: parallel,
                ..SchedulerConfig::default()
            },
            HybridCoordinator::new(128, 784, true, 0).with_mamba_capacity(2),
        );
        assert_eq!(scheduler.add_request(make_req(1, 23520, 1)), admitted);
        if admitted {
            let step = scheduler.schedule();
            assert_eq!(step.scheduled.len(), 1);
            assert_eq!(step.scheduled[0].token_ids.len(), 23520);
        }
    }

    let mut kv = HybridCoordinator::new(3, 784, true, 0).with_mamba_capacity(7);
    kv.set_speculative_blocks(4);
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs: 1,
            enable_mtp: true,
            mtp_draft_len: 4,
        },
        kv,
    );
    assert!(scheduler.add_request(make_req(1, 317, 1)));
    while scheduler.num_waiting() + scheduler.num_running() > 0 {
        let batch = scheduler.schedule();
        let row = &batch.scheduled[0];
        let final_chunk = row.num_computed_tokens + row.token_ids.len() == 317;
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![RequestOutput {
                    token_ids: if final_chunk {
                        vec![42]
                    } else {
                        vec![]
                    },
                    ..dummy_output(1)
                }],
            },
        );
    }
    assert!(scheduler.add_request(make_req(2, 1500, 1)));
    let batch = scheduler.schedule();
    assert_eq!(batch.scheduled.len(), 1);
    assert_eq!(batch.cache_hit_tokens, 0);
    assert_eq!(batch.scheduled[0].num_computed_tokens, 0);
    assert!(!batch.scheduled[0].token_ids.is_empty());
}
