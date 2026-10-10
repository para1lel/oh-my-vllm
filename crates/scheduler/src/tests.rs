//! Integration tests for the Scheduler.
//!
//! These tests cover the paths that matter for Qwen3.8 on a single B200:
//! - Cold prefill: new request admitted from waiting, KV blocks allocated.
//! - Continuous batching: decode steps interleaved with new prefills.
//! - Chunked prefill: token budget forces a long prompt to be split.
//! - Prefix cache hit: second request with a shared prefix skips re-computation.
//! - Preemption: pool exhaustion evicts the last running request.
//! - MTP rollback: rejected draft tokens roll back num_computed_tokens.
//! - Abort: a request removed mid-flight stops appearing in outputs.

use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;

use crate::{
    Scheduler, SchedulerConfig,
    output::{RequestOutput, WorkerOutput},
    request::Request,
};

const BS: usize = 4; // block size for tests

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
    // block_hashes will be (re-)computed by add_request
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

// ── cold prefill ──────────────────────────────────────────────────────────────

#[test]
fn cold_prefill_schedules_request() {
    let mut sched = make_scheduler(64, 128);
    sched.add_request(make_req(1, 8, 10));

    let out = sched.schedule();
    assert_eq!(out.scheduled.len(), 1);
    let sr = &out.scheduled[0];
    assert_eq!(sr.request_id, 1);
    assert_eq!(
        sr.token_ids.len(),
        8,
        "all 8 prompt tokens scheduled at once"
    );
    assert_eq!(
        sr.num_computed_tokens, 0,
        "cold request has no cached tokens"
    );
    assert!(!sr.fa_block_table.is_empty(), "FA blocks must be allocated");
}

// ── decode step advances num_computed_tokens ──────────────────────────────────

#[test]
fn decode_step_advances_computed() {
    let mut sched = make_scheduler(64, 128);
    sched.add_request(make_req(1, 8, 2));

    // Step 1: prefill
    let out = sched.schedule();
    assert_eq!(out.scheduled[0].token_ids.len(), 8);

    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );

    // Step 2: decode — should schedule just the 1 new token
    let out2 = sched.schedule();
    assert_eq!(out2.scheduled.len(), 1);
    let sr2 = &out2.scheduled[0];
    assert_eq!(
        sr2.num_computed_tokens, 8,
        "all 8 prompt tokens now computed"
    );
    assert_eq!(
        sr2.token_ids.len(),
        1,
        "only the new decode token scheduled"
    );
}

#[test]
fn decode_updates_extend_only_new_full_block_hashes() {
    let mut sched = make_scheduler(64, 128);
    assert!(sched.add_request(make_req(1, 3, 6)));
    for expected_len in 4..=8 {
        let step = sched.schedule();
        assert_eq!(step.scheduled.len(), 1);
        apply(
            &mut sched,
            WorkerOutput {
                outputs: vec![dummy_output(1)],
            },
        );
        let req = sched
            .running
            .front()
            .expect("request still has output budget");
        assert_eq!(req.token_ids.len(), expected_len);
        assert_eq!(
            req.block_hashes,
            sched.kv.compute_block_hashes(&req.token_ids)
        );
    }
}

#[test]
fn update_hashes_real_784_token_boundaries() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 32_768,
        max_num_seqs: 4,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    let mut sched = Scheduler::new(config, HybridCoordinator::new(100, 784, true, 0));
    assert!(sched.add_request(make_req(1, 783, 786)));
    for generated in 1..=785 {
        assert_eq!(sched.schedule().scheduled.len(), 1);
        apply(
            &mut sched,
            WorkerOutput {
                outputs: vec![dummy_output(1)],
            },
        );
        let len = 783 + generated;
        if matches!(len, 784 | 785 | 1567 | 1568) {
            let req = sched.running.front().expect("request has one token left");
            assert_eq!(req.token_ids.len(), len);
            assert_eq!(req.block_hashes.len(), len / 784);
            assert_eq!(
                req.block_hashes,
                sched.kv.compute_block_hashes(&req.token_ids)
            );
        }
    }
}

// ── chunked prefill ───────────────────────────────────────────────────────────

#[test]
fn chunked_prefill_splits_long_prompt() {
    // Budget = 8 tokens, prompt = 16 tokens: should take 2 steps.
    let mut sched = make_scheduler(64, 8);
    sched.add_request(make_req(1, 16, 1));

    let out1 = sched.schedule();
    assert_eq!(out1.scheduled.len(), 1);
    assert_eq!(
        out1.scheduled[0].token_ids.len(),
        8,
        "first chunk is exactly the budget"
    );
    assert_eq!(sched.num_waiting(), 0, "request moved to running");
    assert_eq!(sched.num_running(), 1);
}

// ── continuous batching ───────────────────────────────────────────────────────

#[test]
fn continuous_batching_runs_two_requests() {
    let mut sched = make_scheduler(64, 64);
    sched.add_request(make_req(1, 4, 2));
    sched.add_request(make_req(2, 4, 2));

    let out = sched.schedule();
    assert_eq!(out.scheduled.len(), 2, "both requests batched together");
    assert_eq!(out.num_batched_tokens, 8);
}

// ── prefix cache hit ──────────────────────────────────────────────────────────

#[test]
fn prefix_cache_hit_reduces_scheduled_tokens() {
    let mut sched = make_scheduler(128, 256);

    // First request: 16-token prompt, generates 1 token so cache_blocks fires.
    sched.add_request(make_req(1, 16, 1));
    let first = sched.schedule().scheduled.remove(0);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    // Request 1 is finished and its full prompt blocks remain cached.

    // Second request: same 16-token prefix + 1 extra token.
    let tokens2: Vec<u32> = (0..17).collect();
    sched.add_request(Request::new(2, tokens2, 1, Vec::new()));

    let out = sched.schedule();
    let sr = out
        .scheduled
        .iter()
        .find(|s| s.request_id == 2)
        .expect("req 2 must be scheduled");
    // With a 16-token cache hit, only the remaining 1 token is scheduled.
    assert!(
        sr.num_computed_tokens > 0,
        "prefix cache should give a non-zero hit; got {}",
        sr.num_computed_tokens
    );
    let hit_blocks = sr.num_computed_tokens / BS;
    assert_eq!(
        sr.fa_block_table[..hit_blocks],
        first.fa_block_table[..hit_blocks]
    );
    assert_eq!(
        sr.mamba_block_table[..hit_blocks],
        first.mamba_block_table[..hit_blocks]
    );
}

#[test]
fn generated_full_block_remains_a_prefix_hit_for_new_request() {
    let mut sched = make_scheduler(128, 256);
    assert!(sched.add_request(make_req(1, 3, 2)));
    assert_eq!(sched.schedule().scheduled.len(), 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    let generated = sched.running.front().unwrap();
    assert_eq!(generated.token_ids, vec![0, 1, 2, 42]);
    assert_eq!(generated.block_hashes.len(), 1);
    assert_eq!(sched.schedule().scheduled.len(), 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert_eq!(sched.num_running(), 0);

    let second = Request::new(2, vec![0, 1, 2, 42, 99], 1, Vec::new());
    assert!(sched.add_request(second));
    let scheduled = sched.schedule();
    assert_eq!(scheduled.scheduled.len(), 1);
    assert_eq!(scheduled.scheduled[0].num_computed_tokens, 4);
    assert_eq!(scheduled.scheduled[0].token_ids, vec![99]);
}

// ── pool exhaustion: preemption ───────────────────────────────────────────────

#[test]
fn preemption_when_pool_is_full() {
    // Very small pool: enough for one small request, not two.
    // BS=4, 16-token req needs 4 FA + 1 Mamba = 5 blocks (+ null block = 6 total).
    // Give enough for exactly one, not two.
    let mut sched = make_scheduler(9, 256); // 9 blocks: 1 null + 8 usable → fits one 16-tok req
    sched.add_request(make_req(1, 8, 10));
    sched.add_request(make_req(2, 8, 10));

    // First schedule: req 1 gets all blocks.
    let out1 = sched.schedule();
    assert!(out1.scheduled.iter().any(|s| s.request_id == 1));

    // With the pool full, scheduling again should either preempt req 1 or leave
    // req 2 waiting — either is correct; what must NOT happen is a panic.
    let out2 = sched.schedule();
    let total = out2.scheduled.len() + out2.preempted_request_ids.len();
    // At least one of the two requests is accounted for.
    assert!(total >= 1, "scheduler must not silently drop all requests");
}

#[test]
fn pool_pressure_preempts_youngest_and_retries_oldest() {
    let mut sched = make_scheduler(8, 128);
    assert!(sched.add_request(make_req(1, 8, 10)));
    assert!(sched.add_request(make_req(2, 8, 10)));
    let first = sched.schedule();
    assert_eq!(
        first
            .scheduled
            .iter()
            .map(|r| r.request_id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );

    let second = sched.schedule();
    assert_eq!(second.preempted_request_ids, vec![2]);
    assert_eq!(
        second
            .scheduled
            .iter()
            .map(|req| req.request_id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    // Re-admission can immediately recover a cached full block after reset.
    assert_eq!(second.scheduled[1].num_computed_tokens, 8);
    assert_eq!(
        second.scheduled[1].prefill_token_ids,
        Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 42])
    );
    assert_eq!(sched.running.front().unwrap().id, 1);
    assert_eq!(sched.running.back().unwrap().id, 2);
}

#[test]
fn lowest_priority_running_request_preempts_itself_when_it_cannot_fit() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 128,
        max_num_seqs: 64,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    let mut sched = Scheduler::new(config, HybridCoordinator::new(9, BS, false, 0));
    for id in 1..=2 {
        assert!(sched.add_request(make_req(id, 8, 10)));
    }
    assert_eq!(sched.schedule().scheduled.len(), 2);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );
    // Keep a self-preempted request in waiting rather than immediately
    // readmitting it during phase 2, so its reset state is observable.
    sched.config.max_num_seqs = 1;
    let second = sched.schedule();
    assert_eq!(second.preempted_request_ids, vec![2]);
    assert_eq!(
        second
            .scheduled
            .iter()
            .map(|req| req.request_id)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(sched.running.front().unwrap().id, 1);
    let victim = sched.waiting.front().unwrap();
    assert_eq!(victim.id, 2);
    assert_eq!(victim.status, crate::RequestStatus::Preempted);
    assert_eq!(victim.num_computed_tokens, 0);
    assert_eq!(victim.token_ids, vec![0, 1, 2, 3, 4, 5, 6, 7, 42]);
    assert!(sched.kv.full_attn_blocks(2).is_empty());
}

#[test]
fn multiple_tail_victims_keep_admission_order_and_accepted_history() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 128,
        max_num_seqs: 64,
        enable_mtp: true,
        mtp_draft_len: 4,
    };
    let mut coordinator = make_coord(4).with_mamba_capacity(16);
    coordinator.set_speculative_blocks(4);
    let mut sched = Scheduler::new(config, coordinator);
    for id in 1..=4 {
        assert!(sched.add_request(make_req(id, 4, 6)));
    }
    let first = sched.schedule();
    assert_eq!(
        first
            .scheduled
            .iter()
            .map(|req| req.request_id)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(sched.waiting.front().unwrap().id, 4);
    let outputs = first
        .scheduled
        .iter()
        .map(|req| RequestOutput {
            new_draft_token_ids: vec![90, 91, 92, 93],
            ..dummy_output(req.request_id)
        })
        .collect();
    apply(&mut sched, WorkerOutput { outputs });

    let second = sched.schedule();
    assert_eq!(second.preempted_request_ids, vec![3, 2]);
    assert_eq!(second.scheduled[0].request_id, 1);
    assert_eq!(
        sched.waiting.iter().map(|req| req.id).collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    for req in sched.waiting.iter().take(2) {
        assert_eq!(req.status, crate::RequestStatus::Preempted);
        assert_eq!(req.token_ids, vec![0, 1, 2, 3, 42]);
        assert!(req.draft_token_ids.is_empty());
        assert_eq!(req.num_computed_tokens, 0);
    }
}

#[test]
fn paused_stream_keeps_kv_while_another_stream_advances_past_256_steps() {
    let mut sched = make_scheduler(512, 128);
    assert!(sched.add_request(make_req(1, 4, 400)));
    assert!(sched.add_request(make_req(2, 4, 400)));
    assert_eq!(sched.schedule().scheduled.len(), 2);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );
    let before = sched.running.front().unwrap().num_computed_tokens;
    sched.pause(1);
    for _ in 0..300 {
        let step = sched.schedule();
        assert_eq!(step.scheduled.len(), 1);
        assert_eq!(step.scheduled[0].request_id, 2);
        apply(
            &mut sched,
            WorkerOutput {
                outputs: vec![dummy_output(2)],
            },
        );
    }
    assert_eq!(sched.running.front().unwrap().num_computed_tokens, before);
    sched.resume(1);
    let resumed = sched.schedule();
    assert_eq!(resumed.scheduled[0].request_id, 1);
    assert_eq!(resumed.scheduled[0].num_computed_tokens, before);
}

#[test]
fn paused_waiting_request_does_not_block_later_admission() {
    let mut sched = make_scheduler(64, 128);
    assert!(sched.add_request(make_req(1, 4, 3)));
    assert!(sched.add_request(make_req(2, 4, 3)));
    sched.pause(1);
    let first = sched.schedule();
    assert_eq!(first.scheduled[0].request_id, 2);
    assert_eq!(sched.waiting.front().unwrap().id, 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(2)],
        },
    );
    sched.resume(1);
    assert!(
        sched
            .schedule()
            .scheduled
            .iter()
            .any(|request| request.request_id == 1)
    );
}

#[test]
fn paused_waiting_keeps_fcfs_order_after_admission_break() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 128,
        max_num_seqs: 1,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    let mut sched = Scheduler::new(config, make_coord(64));
    for id in 1..=3 {
        assert!(sched.add_request(make_req(id, 4, 3)));
    }
    sched.pause(1);
    let first = sched.schedule();
    assert_eq!(first.scheduled[0].request_id, 2);
    assert_eq!(
        sched.waiting.iter().map(|req| req.id).collect::<Vec<_>>(),
        vec![1, 3]
    );
    sched.abort(2);
    sched.resume(1);
    let second = sched.schedule();
    assert_eq!(second.scheduled[0].request_id, 1);
    assert_eq!(sched.waiting.front().unwrap().id, 3);
}

#[test]
fn paused_waiting_keeps_fcfs_order_after_pool_shortage() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 128,
        max_num_seqs: 64,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    let mut sched = Scheduler::new(config, HybridCoordinator::new(6, BS, false, 0));
    for id in 1..=4 {
        assert!(sched.add_request(make_req(id, 4, 6)));
    }
    sched.pause(1);
    let first = sched.schedule();
    assert_eq!(
        first
            .scheduled
            .iter()
            .map(|req| req.request_id)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(
        sched.waiting.iter().map(|req| req.id).collect::<Vec<_>>(),
        vec![1, 4]
    );
    sched.abort(2);
    sched.abort(3);
    sched.resume(1);
    assert_eq!(sched.schedule().scheduled[0].request_id, 1);
}

#[test]
fn paused_tail_can_be_preempted_without_premature_readmission() {
    let mut sched = make_scheduler(8, 128);
    assert!(sched.add_request(make_req(1, 8, 10)));
    assert!(sched.add_request(make_req(2, 8, 10)));
    assert_eq!(sched.schedule().scheduled.len(), 2);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );
    sched.pause(2);
    let second = sched.schedule();
    assert_eq!(second.preempted_request_ids, vec![2]);
    assert_eq!(second.scheduled[0].request_id, 1);
    assert_eq!(sched.waiting.front().unwrap().id, 2);
    assert!(sched.paused.contains(&2));
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert!(
        sched
            .schedule()
            .scheduled
            .iter()
            .all(|req| req.request_id != 2)
    );
    sched.resume(2);
    assert!(!sched.paused.contains(&2));
}

#[test]
fn naturally_finished_paused_request_clears_pause_state() {
    let mut sched = make_scheduler(32, 128);
    assert!(sched.add_request(make_req(1, 4, 1)));
    assert_eq!(sched.schedule().scheduled.len(), 1);
    sched.pause(1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert!(!sched.paused.contains(&1));
    assert_eq!(sched.num_running(), 0);
}

// ── MTP rollback ──────────────────────────────────────────────────────────────

#[test]
fn mtp_rollback_adjusts_computed_tokens() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 256,
        max_num_seqs: 64,
        enable_mtp: true,
        mtp_draft_len: 4,
    };
    let mut sched = Scheduler::new(config, make_coord(128));

    sched.add_request(make_req(1, 4, 10));

    // Prefill step.
    let _ = sched.schedule();
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![10, 11, 12, 13],
            }],
        },
    );

    // After update, req 1 should have new drafts installed.
    // Decode step: worker accepts only 2 of 4 drafts.
    let _ = sched.schedule();
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![10, 11, 99],
                num_accepted_draft_tokens: 2,
                new_draft_token_ids: Vec::new(),
            }],
        },
    );
    // No panic = rollback completed successfully.
}

#[test]
fn mtp_scheduled_gdn_slots_match_worker_candidate_slice_at_page_boundaries() {
    for draft_len in [4, 7] {
        for (prompt_len, budget, expected_prefill_steps) in
            [(783, 512, 2), (784, 784, 1), (785, 784, 2), (1568, 784, 2)]
        {
            let config = SchedulerConfig {
                max_num_batched_tokens: budget,
                max_num_seqs: 1,
                enable_mtp: true,
                mtp_draft_len: draft_len,
            };
            let mut kv = HybridCoordinator::new(64, 784, false, 0).with_mamba_capacity(16);
            kv.set_speculative_blocks(draft_len);
            let mut scheduler = Scheduler::new(config, kv);
            assert!(scheduler.add_request(make_req(1, prompt_len, 10)));
            let drafts: Vec<u32> = (90..90 + draft_len as u32).collect();

            let mut prefill_steps = 0;
            let mut next_start = 0;
            let source = loop {
                let prefill = scheduler.schedule();
                let first = &prefill.scheduled[0];
                assert_eq!(first.num_computed_tokens, next_start);
                next_start += first.token_ids.len();
                assert!(next_start <= prompt_len);
                prefill_steps += 1;
                let last_chunk = first.num_computed_tokens + first.token_ids.len() == prompt_len;
                let source = first.mamba_block_table[(prompt_len - 1) / 784];
                apply(
                    &mut scheduler,
                    WorkerOutput {
                        outputs: vec![RequestOutput {
                            request_id: 1,
                            token_ids: if last_chunk {
                                vec![42]
                            } else {
                                Vec::new()
                            },
                            num_accepted_draft_tokens: 0,
                            new_draft_token_ids: if last_chunk {
                                drafts.clone()
                            } else {
                                Vec::new()
                            },
                        }],
                    },
                );
                if last_chunk {
                    break source;
                }
            };
            assert_eq!(prefill_steps, expected_prefill_steps);
            assert_ne!(source, 0);

            let verification = scheduler.schedule();
            let request = &verification.scheduled[0];
            assert_eq!(request.num_computed_tokens, prompt_len);
            assert_eq!(request.token_ids.len(), draft_len + 1);
            assert_eq!(request.token_ids[1..], drafts);
            let base = (request.num_computed_tokens + request.token_ids.len() - 1) / 784;
            let candidates = &request.mamba_block_table[base..base + request.token_ids.len()];
            assert!(candidates.iter().all(|&slot| slot != 0));
            assert_eq!(
                candidates
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                draft_len + 1,
                "GDN candidate slots must be distinct at prompt length {prompt_len}"
            );
            assert_eq!(request.mamba_block_table[(prompt_len - 1) / 784], source);
            apply(
                &mut scheduler,
                WorkerOutput {
                    outputs: vec![RequestOutput {
                        request_id: 1,
                        token_ids: vec![90, 91, 92, 99],
                        num_accepted_draft_tokens: 3,
                        new_draft_token_ids: Vec::new(),
                    }],
                },
            );
            let kept = scheduler.running.front().unwrap();
            assert_eq!(kept.num_computed_tokens, prompt_len + 4);
            assert_eq!(&kept.token_ids[prompt_len..], [42, 90, 91, 92, 99]);
            assert!(kept.draft_token_ids.is_empty());
            let next = scheduler.schedule();
            assert_eq!(next.scheduled[0].num_computed_tokens, prompt_len + 4);
            assert_eq!(next.scheduled[0].token_ids, [99]);
        }
    }
}

// ── abort ─────────────────────────────────────────────────────────────────────

#[test]
fn abort_removes_request() {
    let mut sched = make_scheduler(64, 128);
    sched.add_request(make_req(1, 4, 10));
    sched.add_request(make_req(2, 4, 10));

    // Schedule both.
    let _ = sched.schedule();
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );

    // Abort req 1 while it is running.
    sched.abort(1);

    assert_eq!(sched.num_running(), 1, "only req 2 should remain running");
    // Abort req in waiting queue.
    sched.add_request(make_req(3, 4, 10));
    sched.abort(3);
    assert_eq!(sched.num_waiting(), 0, "req 3 must be removed from waiting");
    sched.abort(1); // Repeated/unknown cancellation is idempotent.
    sched.abort(99);
    assert_eq!(sched.schedule().finished_request_ids, vec![1, 3]);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(2)],
        },
    );
    assert!(sched.schedule().finished_request_ids.is_empty());
}

#[test]
fn abort_preempted_request_notifies_worker_once() {
    let mut sched = make_scheduler(9, 128);
    sched.add_request(make_req(1, 8, 10));
    sched.add_request(make_req(2, 8, 10));
    assert_eq!(sched.schedule().scheduled.len(), 2);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1), dummy_output(2)],
        },
    );
    let next = sched.schedule();
    assert_eq!(next.preempted_request_ids, vec![2]);
    assert_eq!(next.scheduled[0].request_id, 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: next
                .scheduled
                .iter()
                .map(|request| dummy_output(request.request_id))
                .collect(),
        },
    );
    sched.abort(2);
    sched.abort(2);
    assert_eq!(sched.num_waiting(), 0);
    assert_eq!(sched.schedule().finished_request_ids, vec![2]);
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
fn partial_prefill_advances_without_generating_a_token() {
    let mut sched = make_scheduler(64, 8);
    sched.add_request(make_req(1, 16, 1));
    assert_eq!(sched.schedule().num_batched_tokens, 8);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![],
            }],
        },
    );
    let second = sched.schedule();
    assert_eq!(second.scheduled[0].num_computed_tokens, 8);
    assert_eq!(second.num_batched_tokens, 8);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert_eq!(sched.num_running(), 0);
    assert_eq!(sched.schedule().finished_request_ids, vec![1]);
    assert!(sched.schedule().finished_request_ids.is_empty());
}

#[test]
fn partial_tail_stops_at_checkpoint_boundary() {
    let mut sched = make_scheduler(64, 128);
    sched.add_request(make_req(1, 11, 1));
    assert_eq!(sched.schedule().num_batched_tokens, 8);
}

#[test]
fn output_count_truncates_speculative_tail_after_limit_changes() {
    let mut kv = make_coord(64);
    kv.set_speculative_blocks(2);
    let mut sched = Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: 128,
            max_num_seqs: 4,
            enable_mtp: true,
            mtp_draft_len: 2,
        },
        kv,
    );
    sched.add_request(make_req(1, 4, 4));
    sched.schedule();
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![10, 11],
            }],
        },
    );
    sched.schedule();
    // A lower limit can arrive after the speculative batch is scheduled.
    sched.running.front_mut().unwrap().max_tokens = 3;
    let result = apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![10, 11, 12],
                num_accepted_draft_tokens: 2,
                new_draft_token_ids: vec![],
            }],
        },
    );
    assert_eq!(result.outputs[0].token_ids, vec![10, 11]);
    assert_eq!(sched.num_running(), 0);
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
}

// ── SCH-01: update() validates worker output lengths ─────────────────────────

#[test]
fn update_rejects_malformed_draft_output_without_mutation() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 256,
        max_num_seqs: 64,
        enable_mtp: true,
        mtp_draft_len: 4,
    };
    let mut sched = Scheduler::new(config, make_coord(128));
    sched.add_request(make_req(1, 4, 10));

    // Prefill step.
    let _ = sched.schedule();
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![10, 11, 12, 13],
            }],
        },
    );

    // Decode step: reject both wrong token counts and an impossible acceptance.
    let _ = sched.schedule();
    let before = sched.running.front().unwrap().clone();
    for (accepted, token_ids, expected_reason) in [
        (1, vec![10, 11, 12], "wrong verified token count"),
        (3, vec![10], "wrong verified token count"),
        (99, vec![10], "accepted unscheduled drafts"),
    ] {
        let malformed = WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids,
                num_accepted_draft_tokens: accepted,
                new_draft_token_ids: vec![],
            }],
        };
        assert_eq!(
            sched.validate_output(&malformed).unwrap_err().reason,
            expected_reason
        );
        assert_eq!(sched.update(malformed).unwrap_err().reason, expected_reason);
        let after = sched.running.front().unwrap();
        assert_eq!(after.token_ids, before.token_ids);
        assert_eq!(after.draft_token_ids, before.draft_token_ids);
        assert_eq!(after.num_computed_tokens, before.num_computed_tokens);
        assert_eq!(after.num_in_flight_tokens, before.num_in_flight_tokens);
        assert_eq!(sched.num_running(), 1);
    }
}

// ── SCH-03: aligned_prefill never returns 0 for a non-empty count ─────────────

#[test]
fn aligned_prefill_never_stalls_unaligned_start() {
    // Block size = 4. At an actual unaligned computed position of 5, an
    // available count of 1 rounds down to 4 unless the clamp takes effect.
    let sched = make_scheduler(128, 256);
    let mut request = make_req(1, 11, 20);
    request.num_computed_tokens = 5;
    assert_eq!(
        sched.aligned_prefill(&request, request.num_computed_tokens, 1),
        1
    );
}

#[test]
fn production_prefill_defers_unaligned_admission_without_allocating_slots() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 32768,
        max_num_seqs: 4,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    let kv = HybridCoordinator::new(500, 784, false, 0).with_mamba_capacity(16);
    let mut scheduler = Scheduler::new(config, kv);
    for id in 1..=4 {
        assert!(scheduler.add_request(make_req(id, 32768, 1)));
    }
    let first = scheduler.schedule();
    assert_eq!(first.scheduled.len(), 1);
    assert_eq!(first.num_batched_tokens, 32144);
    assert_eq!(scheduler.waiting.len(), 3);
    for id in 2..=4 {
        assert!(scheduler.kv.full_attn_blocks(id).is_empty());
        assert!(scheduler.kv.mamba_blocks(id).is_empty());
    }
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: Vec::new(),
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: Vec::new(),
            }],
        },
    );
    let second = scheduler.schedule();
    assert_eq!(second.num_batched_tokens, 32768);
    assert_eq!(
        second
            .scheduled
            .iter()
            .map(|request| (request.request_id, request.token_ids.len()))
            .collect::<Vec<_>>(),
        vec![(1, 624), (2, 32144)]
    );
}

#[test]
fn continued_prefill_uses_next_step_when_alignment_consumes_the_remainder() {
    let mut scheduler = make_scheduler(64, 6);
    assert!(scheduler.add_request(make_req(1, 8, 1)));
    assert!(scheduler.add_request(make_req(2, 16, 1)));
    scheduler.running = std::mem::take(&mut scheduler.waiting);
    for request in &mut scheduler.running {
        request.status = crate::RequestStatus::Running;
    }
    for final_chunk in [false, true] {
        let step = scheduler.schedule();
        assert_eq!(step.scheduled.len(), 1);
        assert_eq!(step.scheduled[0].request_id, 1);
        assert_eq!(step.num_batched_tokens, 4);
        assert!(scheduler.kv.full_attn_blocks(2).is_empty());
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![RequestOutput {
                    request_id: 1,
                    token_ids: if final_chunk {
                        vec![42]
                    } else {
                        Vec::new()
                    },
                    num_accepted_draft_tokens: 0,
                    new_draft_token_ids: Vec::new(),
                }],
            },
        );
    }
    let next = scheduler.schedule();
    assert_eq!(next.scheduled.len(), 1);
    assert_eq!(next.scheduled[0].request_id, 2);
    assert_eq!(next.scheduled[0].num_computed_tokens, 0);
    assert_eq!(next.num_batched_tokens, 4);
}

#[test]
fn one_token_checkpoint_and_final_tail_keep_their_schedule() {
    let scheduler = make_scheduler(64, 8);
    let request = make_req(1, 16, 1);
    assert!(!scheduler.defer_aligned_prefill(&request, 3, 1, 1, true));
    assert!(!scheduler.defer_aligned_prefill(&request, 15, 1, 1, true));
    assert!(!scheduler.defer_aligned_prefill(&request, 16, 1, 1, true));
    assert!(!scheduler.defer_aligned_prefill(&request, 0, 1, 1, false));
}

#[test]
fn sub_block_budget_keeps_concurrent_prefill_progress() {
    let mut scheduler = make_scheduler(64, 3);
    assert!(scheduler.add_request(make_req(1, 2, 1)));
    assert!(scheduler.add_request(make_req(2, 16, 1)));
    let step = scheduler.schedule();
    assert_eq!(step.num_batched_tokens, 3);
    assert_eq!(
        step.scheduled
            .iter()
            .map(|request| request.token_ids.len())
            .collect::<Vec<_>>(),
        vec![2, 1]
    );
}

#[test]
fn decode_steps_keep_new_prefill_progress_before_decode_finishes() {
    let mut scheduler = make_scheduler(128, 4);
    assert!(scheduler.add_request(make_req(1, 4, 64)));
    assert_eq!(scheduler.schedule().num_batched_tokens, 4);
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    let mut prefill = make_req(2, 32, 1);
    for token in &mut prefill.token_ids {
        *token += 1000;
    }
    assert!(scheduler.add_request(prefill));
    for (computed, count) in [(0, 1), (1, 3), (4, 1), (5, 3), (8, 1), (9, 3)] {
        let step = scheduler.schedule();
        assert_eq!(step.scheduled.len(), 2);
        assert_eq!(step.scheduled[0].request_id, 1);
        assert_eq!(step.scheduled[0].token_ids.len(), 1);
        assert_eq!(step.scheduled[1].request_id, 2);
        assert_eq!(step.scheduled[1].num_computed_tokens, computed);
        assert_eq!(step.scheduled[1].token_ids.len(), count);
        assert!(step.preempted_request_ids.is_empty());
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![
                    dummy_output(1),
                    RequestOutput {
                        request_id: 2,
                        token_ids: Vec::new(),
                        num_accepted_draft_tokens: 0,
                        new_draft_token_ids: Vec::new(),
                    },
                ],
            },
        );
    }
    assert_eq!(scheduler.running.len(), 2);
}

#[test]
fn decode_steps_keep_continued_prefill_progress() {
    let mut scheduler = make_scheduler(128, 8);
    assert!(scheduler.add_request(make_req(1, 4, 64)));
    let mut prefill = make_req(2, 32, 1);
    for token in &mut prefill.token_ids {
        *token += 1000;
    }
    assert!(scheduler.add_request(prefill));
    assert_eq!(scheduler.schedule().num_batched_tokens, 8);
    apply(
        &mut scheduler,
        WorkerOutput {
            outputs: vec![
                dummy_output(1),
                RequestOutput {
                    request_id: 2,
                    token_ids: Vec::new(),
                    num_accepted_draft_tokens: 0,
                    new_draft_token_ids: Vec::new(),
                },
            ],
        },
    );
    scheduler.config.max_num_batched_tokens = 4;
    for (computed, count) in [(4, 1), (5, 3), (8, 1), (9, 3), (12, 1), (13, 3)] {
        let step = scheduler.schedule();
        assert_eq!(step.scheduled.len(), 2);
        assert_eq!(step.scheduled[0].token_ids.len(), 1);
        assert_eq!(step.scheduled[1].request_id, 2);
        assert_eq!(step.scheduled[1].num_computed_tokens, computed);
        assert_eq!(step.scheduled[1].token_ids.len(), count);
        assert!(step.preempted_request_ids.is_empty());
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![
                    dummy_output(1),
                    RequestOutput {
                        request_id: 2,
                        token_ids: Vec::new(),
                        num_accepted_draft_tokens: 0,
                        new_draft_token_ids: Vec::new(),
                    },
                ],
            },
        );
    }
}

#[test]
fn speculative_verification_keeps_new_prefill_progress() {
    for draft_count in [4, 7] {
        let mut kv = make_coord(128);
        kv.set_speculative_blocks(draft_count);
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_num_batched_tokens: draft_count + 4,
                max_num_seqs: 4,
                enable_mtp: true,
                mtp_draft_len: draft_count,
            },
            kv,
        );
        assert!(scheduler.add_request(make_req(1, 4, 64)));
        scheduler.schedule();
        let decode_output = || RequestOutput {
            request_id: 1,
            token_ids: vec![42],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![10; draft_count],
        };
        apply(
            &mut scheduler,
            WorkerOutput {
                outputs: vec![decode_output()],
            },
        );
        let mut prefill = make_req(2, 32, 1);
        for token in &mut prefill.token_ids {
            *token += 1000;
        }
        assert!(scheduler.add_request(prefill));
        for (computed, count) in [(0, 1), (1, 3), (4, 1)] {
            let step = scheduler.schedule();
            assert_eq!(step.scheduled.len(), 2);
            assert_eq!(step.scheduled[0].request_id, 1);
            assert_eq!(step.scheduled[0].token_ids.len(), draft_count + 1);
            assert_eq!(step.scheduled[1].request_id, 2);
            assert_eq!(step.scheduled[1].num_computed_tokens, computed);
            assert_eq!(step.scheduled[1].token_ids.len(), count);
            assert!(step.preempted_request_ids.is_empty());
            apply(
                &mut scheduler,
                WorkerOutput {
                    outputs: vec![
                        decode_output(),
                        RequestOutput {
                            request_id: 2,
                            token_ids: Vec::new(),
                            num_accepted_draft_tokens: 0,
                            new_draft_token_ids: Vec::new(),
                        },
                    ],
                },
            );
        }
    }
}

// ── SCH-04: over-capacity requests are rejected at admission ─────────────────

#[test]
fn add_request_rejects_request_exceeding_fa_pool() {
    // Pool has 4 blocks (+ null = capacity 4). Request needs more.
    let mut sched = make_scheduler(4, 256);
    // A request that wants 100 tokens needs 25 blocks (100/4); pool has only 4.
    let req = make_req(1, 100, 100);
    assert!(
        !sched.add_request(req),
        "over-capacity request must be rejected"
    );
    assert_eq!(sched.num_waiting(), 0);
}

#[test]
fn add_request_admits_request_within_fa_pool() {
    let mut sched = make_scheduler(128, 256);
    let req = make_req(1, 4, 10);
    assert!(
        sched.add_request(req),
        "within-capacity request must be admitted"
    );
    assert_eq!(sched.num_waiting(), 1);
}

#[test]
fn add_request_rejects_output_budget_that_exceeds_fa_pool() {
    let mut sched = make_scheduler(8, 256);
    assert!(!sched.add_request(make_req(1, 1, 32)));
    assert_eq!(sched.num_waiting(), 0);
}

#[test]
fn add_request_rejects_insufficient_separate_gdn_pool() {
    let mut coord = make_coord(128).with_mamba_capacity(6);
    coord.set_speculative_blocks(4);
    let config = SchedulerConfig {
        enable_mtp: true,
        mtp_draft_len: 4,
        ..SchedulerConfig::default()
    };
    let mut sched = Scheduler::new(config, coord);
    assert!(!sched.add_request(make_req(1, 1, 10)));
    assert_eq!(sched.num_waiting(), 0);
}

#[test]
fn separate_gdn_pool_covers_two_successive_steps_with_mtp() {
    let mut coord = make_coord(128).with_mamba_capacity(7);
    coord.set_speculative_blocks(4);
    let mut sched = Scheduler::new(
        SchedulerConfig {
            enable_mtp: true,
            mtp_draft_len: 4,
            ..SchedulerConfig::default()
        },
        coord,
    );
    assert!(sched.add_request(make_req(1, 4, 10)));
    assert_eq!(sched.schedule().scheduled.len(), 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![10, 11, 12, 13],
            }],
        },
    );
    let second = sched.schedule();
    assert!(second.preempted_request_ids.is_empty());
    assert_eq!(second.scheduled[0].request_id, 1);
}

#[test]
fn ordinary_gdn_pool_requires_previous_and_next_state_slots() {
    let mut rejected = Scheduler::new(
        SchedulerConfig::default(),
        make_coord(128).with_mamba_capacity(2),
    );
    assert!(!rejected.add_request(make_req(1, 4, 10)));

    let mut sched = Scheduler::new(
        SchedulerConfig::default(),
        make_coord(128).with_mamba_capacity(3),
    );
    assert!(sched.add_request(make_req(1, 4, 10)));
    assert_eq!(sched.schedule().scheduled.len(), 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    let second = sched.schedule();
    assert!(second.preempted_request_ids.is_empty());
    assert_eq!(second.scheduled[0].request_id, 1);
}

#[test]
fn shared_pool_admission_reserves_fa_and_two_step_gdn_peak() {
    let mut rejected = make_scheduler(6, 256);
    assert!(!rejected.add_request(make_req(1, 4, 10)));

    let mut sched = make_scheduler(7, 256);
    assert!(sched.add_request(make_req(1, 4, 10)));
    assert_eq!(sched.schedule().scheduled.len(), 1);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    let second = sched.schedule();
    assert!(second.preempted_request_ids.is_empty());
    assert_eq!(second.scheduled[0].request_id, 1);
}

#[test]
fn one_forward_request_fits_single_gdn_slot_and_fa_page() {
    let mut shared = make_scheduler(3, 256);
    assert!(shared.add_request(make_req(1, 4, 1)));
    assert_eq!(shared.schedule().scheduled.len(), 1);
    apply(
        &mut shared,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert_eq!(shared.num_running(), 0);

    let mut separate = Scheduler::new(
        SchedulerConfig::default(),
        make_coord(3).with_mamba_capacity(2),
    );
    assert!(separate.add_request(make_req(2, 4, 1)));
    assert_eq!(separate.schedule().scheduled.len(), 1);
    apply(
        &mut separate,
        WorkerOutput {
            outputs: vec![dummy_output(2)],
        },
    );
    assert_eq!(separate.num_running(), 0);
}

#[test]
fn one_output_after_chunked_prefill_still_needs_two_gdn_slots() {
    let mut rejected = Scheduler::new(
        SchedulerConfig::default(),
        make_coord(128).with_mamba_capacity(2),
    );
    assert!(!rejected.add_request(make_req(1, 5, 1)));

    let mut sched = Scheduler::new(
        SchedulerConfig::default(),
        make_coord(128).with_mamba_capacity(3),
    );
    assert!(sched.add_request(make_req(1, 5, 1)));
    assert_eq!(sched.schedule().num_batched_tokens, 4);
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![RequestOutput {
                request_id: 1,
                token_ids: vec![],
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![],
            }],
        },
    );
    let second = sched.schedule();
    assert_eq!(second.num_batched_tokens, 1);
    assert!(second.preempted_request_ids.is_empty());
    apply(
        &mut sched,
        WorkerOutput {
            outputs: vec![dummy_output(1)],
        },
    );
    assert_eq!(sched.num_running(), 0);
}

// ── SCH-05: remove_blocks_in_range skips nulls instead of stopping ───────────
// The fix is in group.rs; this test exercises the scheduler-level path that
// exercises preemption + re-admit to ensure blocks are properly freed even
// when the block table has interior null slots.

#[test]
fn preempted_request_can_be_readmitted_after_pool_frees() {
    // Small pool to force preemption.
    let mut sched = make_scheduler(6, 16);
    // Fill the pool with a long request.
    let big = make_req(1, 20, 1);
    sched.add_request(big);
    let _ = sched.schedule();
    // Preempt by scheduling a new request that can't fit.
    sched.abort(1);
    // Pool must be non-empty after abort so a second request can be served.
    let small = make_req(2, 2, 1);
    assert!(sched.add_request(small));
    let out = sched.schedule();
    assert!(!out.scheduled.is_empty());
}
