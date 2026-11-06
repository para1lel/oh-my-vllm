use super::*;

#[test]
fn prefill_budget_prefers_blocks_then_tokens_then_fewer_suffixes() {
    verify_long_budget();
    verify_prefill_fragments();
    verify_small_budget();
    verify_whole_block_priority();
    verify_cursor_rotation();
    verify_suffix_priority();
    verify_small_exhaustive_quotas();
    verify_prefill_ranges();
}

fn verify_long_budget() {
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
}

fn verify_prefill_fragments() {
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
}

fn verify_small_budget() {
    for (budget, lengths, expected) in [
        (24, [20, 100, 100], vec![16, 8]),
        (24, [4, 100, 100], vec![24]),
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
}

fn verify_whole_block_priority() {
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
}

fn verify_cursor_rotation() {
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

fn verify_suffix_priority() {
    for (budget, lengths, expected) in [
        (767, vec![317, 450], vec![(1, 316), (2, 449)]),
        (800, vec![401, 401], vec![(1, 400), (2, 400)]),
        (784, vec![317, 450, 1000], vec![(3, 784)]),
        (2353, vec![784, 3136], vec![(1, 1), (2, 2352)]),
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
}

fn verify_small_exhaustive_quotas() {
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
}

fn verify_prefill_ranges() {
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
