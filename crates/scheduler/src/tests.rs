mod decoding;
mod host_prefix;
mod host_swap;
mod prefill;
mod priorities;

use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;

use crate::{
    Scheduler, SchedulerConfig,
    output::{RequestOutput, WorkerOutput},
    policy::CachePolicy,
    request::Request,
};

const BS: usize = 4;

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
