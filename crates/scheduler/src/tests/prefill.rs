use super::*;

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
