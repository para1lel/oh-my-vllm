use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;

use crate::{
    Scheduler, SchedulerConfig,
    output::{RequestOutput, WorkerOutput},
    request::Request,
};

const BS: usize = 4;

fn make_coord(num_blocks: u32) -> HybridCoordinator {
    HybridCoordinator::new(num_blocks, BS, true, 0)
}

fn make_scheduler(num_blocks: u32, max_tokens_per_step: usize) -> Scheduler {
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

fn apply(scheduler: &mut Scheduler, output: WorkerOutput) -> WorkerOutput {
    scheduler.update(output).expect("test output must be valid")
}

fn production_long_scheduler(max_seqs: usize) -> Scheduler {
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

    let mut scheduler = make_scheduler(64, 4);
    let mut first = make_req(1, 9, 1);
    first.cache_salt = Some(BlockHash(11));
    assert!(scheduler.add_request(first));
    for step in 0..3 {
        scheduler.schedule();
        let mut result = dummy_output(1);
        if step < 2 {
            result.token_ids.clear();
        }
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![result],
            },
        );
    }
    for (id, salt, expected_hit) in [
        (2, Some(BlockHash(22)), 0),
        (3, None, 0),
        (4, Some(BlockHash(11)), 8),
    ] {
        let mut request = make_req(id, 9, 1);
        request.cache_salt = salt;
        assert!(scheduler.add_request(request));
        assert_eq!(scheduler.schedule().cache_hit_tokens, expected_hit);
        scheduler.abort(id);
    }
}

#[test]
fn malformed_output_does_not_publish_pending_prefix_and_keeps_valid_history() {
    let mut sched = make_scheduler(64, 4);
    assert!(sched.add_request(make_req(1, 9, 1)));
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
    sched.schedule();
    assert!(
        sched
            .update(WorkerOutput {
                outputs: vec![dummy_output(1)],
            })
            .is_err()
    );
    sched.abort(1);
    assert!(sched.add_request(make_req(2, 9, 1)));
    let batch = sched.schedule();
    assert_eq!(batch.scheduled[0].num_computed_tokens, 4);
    assert_eq!(batch.cache_hit_tokens, 4);
}

#[test]
fn preemption_resends_accepted_history_without_drafts() {
    let mut sched = make_scheduler(9, 128);
    sched.add_request(make_req(1, 8, 10));
    sched.add_request(make_req(2, 8, 10));
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
    assert_eq!(
        resumed.prefill_token_ids,
        Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 42])
    );
    assert!(sched.running.back().unwrap().draft_token_ids.is_empty());
    let readmitted = sched.running.back().unwrap();
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
        vec![6, 1, 2, 3, 4, 5]
    );
    assert!(step.preempted_request_ids.is_empty());
}

#[test]
fn prefill_budget_is_shared_by_short_and_long_requests() {
    for (budget, lengths, expected) in [
        (24, [20, 100, 100], vec![8, 8, 8]),
        (24, [4, 100, 100], vec![4, 12, 8]),
        (3, [20, 100, 100], vec![1, 1, 1]),
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
}
