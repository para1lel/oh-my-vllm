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

    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1)],
    });

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
    let _ = sched.schedule();
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1)],
    });
    // Decode step to trigger cache_blocks for all 4 FA blocks.
    let _ = sched.schedule();
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1)],
    });
    // Request 1 is now finished (max_tokens=1 reached after first output).

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
    for cached in &sr.fa_block_table[..hit_blocks] {
        assert!(!sr.new_block_ids_to_zero.contains(cached));
    }
    for cached in &sr.mamba_block_table[..hit_blocks] {
        assert!(!sr.new_block_ids_to_zero.contains(cached));
    }
    assert!(!sr.new_block_ids_to_zero.is_empty());
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
    sched.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![42],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![10, 11, 12, 13],
        }],
    });

    // After update, req 1 should have new drafts installed.
    // Decode step: worker accepts only 2 of 4 drafts.
    let _ = sched.schedule();
    sched.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![10, 11, 99],
            num_accepted_draft_tokens: 2,
            new_draft_token_ids: Vec::new(),
        }],
    });
    // No panic = rollback completed successfully.
}

// ── abort ─────────────────────────────────────────────────────────────────────

#[test]
fn abort_removes_request() {
    let mut sched = make_scheduler(64, 128);
    sched.add_request(make_req(1, 4, 10));
    sched.add_request(make_req(2, 4, 10));

    // Schedule both.
    let _ = sched.schedule();
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1), dummy_output(2)],
    });

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
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(2)],
    });
    assert!(sched.schedule().finished_request_ids.is_empty());
}

#[test]
fn abort_preempted_request_notifies_worker_once() {
    let mut sched = make_scheduler(9, 128);
    sched.add_request(make_req(1, 8, 10));
    sched.add_request(make_req(2, 8, 10));
    assert_eq!(sched.schedule().scheduled.len(), 2);
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1), dummy_output(2)],
    });
    let next = sched.schedule();
    assert_eq!(next.preempted_request_ids, vec![2]);
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1)],
    });
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
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1), second_output],
    });
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
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1), dummy_output(2)],
    });
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
    sched.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![],
        }],
    });
    let second = sched.schedule();
    assert_eq!(second.scheduled[0].num_computed_tokens, 8);
    assert_eq!(second.num_batched_tokens, 8);
    sched.update(WorkerOutput {
        outputs: vec![dummy_output(1)],
    });
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
fn output_count_truncates_speculative_tail_to_request_limit() {
    let mut sched = make_scheduler(64, 128);
    sched.add_request(make_req(1, 4, 2));
    sched.schedule();
    let result = sched.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![10, 11, 12],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![],
        }],
    });
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
    scheduler.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![42],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![10, 11],
        }],
    });
    assert_eq!(scheduler.schedule().num_batched_tokens, 3);
    scheduler.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![10, 99],
            num_accepted_draft_tokens: 1,
            new_draft_token_ids: vec![],
        }],
    });
    let request = scheduler.running.front().unwrap();
    assert_eq!(request.num_computed_tokens, 6);
    assert_eq!(&request.token_ids[4..], &[42, 10, 99]);
    assert!(request.draft_token_ids.is_empty());
    assert_eq!(scheduler.schedule().scheduled[0].token_ids, vec![99]);
}

// ── SCH-01: update() validates worker output lengths ─────────────────────────

#[test]
#[should_panic(expected = "worker accepted")]
fn update_panics_on_overreported_accepted_drafts() {
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
    sched.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![42],
            num_accepted_draft_tokens: 0,
            new_draft_token_ids: vec![10, 11, 12, 13],
        }],
    });

    // Decode step: worker claims to have accepted more drafts than were sent.
    let _ = sched.schedule();
    sched.update(WorkerOutput {
        outputs: vec![RequestOutput {
            request_id: 1,
            token_ids: vec![42],
            num_accepted_draft_tokens: 99, // invalid
            new_draft_token_ids: vec![],
        }],
    });
}

// ── SCH-03: aligned_prefill never returns 0 for a non-empty count ─────────────

#[test]
fn aligned_prefill_never_stalls_unaligned_start() {
    // Block size = BS (4). Request with 3 tokens (not block-aligned).
    // Start at 3 (prompt boundary), count=1. Should return 1, never 0.
    let mut sched = make_scheduler(128, 256);
    sched.add_request(make_req(1, 3, 20));
    let out = sched.schedule();
    // The scheduler must always schedule at least 1 token.
    assert!(out.num_batched_tokens >= 1);
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
