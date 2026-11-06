use super::*;

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
fn aligned_prompt_reuses_prefix_before_last_token() {
    for (prompt_len, speculative) in [
        (784, 0),
        (784, 4),
        (1568, 0),
        (1568, 4),
        (1569, 0),
        (1569, 4),
    ] {
        let mut kv = HybridCoordinator::new(1400, 784, true, 0).with_mamba_capacity(128);
        kv.set_speculative_blocks(speculative);
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_num_batched_tokens: 32768,
                max_num_seqs: 4,
                enable_mtp: false,
                mtp_draft_len: 0,
            },
            kv,
        );
        assert!(scheduler.add_request(make_req(1, prompt_len, 1)));
        finish_prefills(&mut scheduler);
        assert!(scheduler.add_request(make_req(2, prompt_len, 1)));
        let batch = scheduler.schedule();
        assert_eq!(batch.cache_hit_tokens, prompt_len - 1);
        assert_eq!(batch.scheduled.len(), 1);
        assert_eq!(batch.scheduled[0].num_computed_tokens, prompt_len - 1);
        assert_eq!(batch.scheduled[0].token_ids.len(), 1);
        scheduler.abort(2);
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
    verify_mtp_accepted_history();
    verify_final_token_budget();
    verify_concurrent_draft_budget();
    verify_residual_draft_budget();
}

fn verify_residual_draft_budget() {
    let mut residual = make_scheduler(64, 16);
    residual.config.enable_mtp = true;
    residual.config.mtp_draft_len = 4;
    let mut request = make_req(1, 4, 8);
    request.record_partial_prefix = false;
    assert!(residual.add_request(request));
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

fn verify_concurrent_draft_budget() {
    let mut concurrent = make_scheduler(1024, 16);
    concurrent.config.enable_mtp = true;
    concurrent.config.mtp_draft_len = 2;
    for id in 1..=2 {
        let mut request = make_req(id, 4, 8);
        request.record_partial_prefix = false;
        assert!(concurrent.add_request(request));
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
}

fn verify_final_token_budget() {
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

fn verify_mtp_accepted_history() {
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
    let mut request = make_req(1, 4, 10);
    request.record_partial_prefix = false;
    scheduler.add_request(request);
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
}
