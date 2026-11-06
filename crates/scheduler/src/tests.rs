mod decoding;
mod prefill;
mod priorities;
mod radix;
mod sessions;

use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;

use crate::{
    Request, RequestStatus, Scheduler, SchedulerConfig,
    execution::{CacheStepResult, ExecutionKey, SamplingMode, WorkAction, WorkPurpose, WorkSpec},
    output::{RequestOutput, WorkerOutput},
    policy::CachePolicy,
};

const BS: usize = 4;

fn make_coord(num_blocks: u32) -> HybridCoordinator {
    HybridCoordinator::new(num_blocks, BS, true, 0)
}

fn make_scheduler(num_blocks: u32, max_tokens_per_step: usize) -> Scheduler<HybridCoordinator> {
    Scheduler::new(
        SchedulerConfig {
            max_num_batched_tokens: max_tokens_per_step,
            max_num_seqs: 64,
            enable_mtp: false,
            mtp_draft_len: 0,
        },
        make_coord(num_blocks),
    )
}

fn make_req(id: u64, num_tokens: usize, max_output: usize) -> Request {
    Request::new(id, (0..num_tokens as u32).collect(), max_output, Vec::new())
}

fn dummy_output(rid: u64) -> RequestOutput {
    let execution = ExecutionKey {
        worker_epoch: 0,
        step_id: 0,
        request_id: rid,
        lease_id: 0,
    };
    RequestOutput {
        request_id: rid,
        execution,
        work: WorkSpec {
            purpose: WorkPurpose::Advance,
            action: WorkAction::Forward { start: 0, end: 1 },
            sampling: SamplingMode::Target,
        },
        cache_result: CacheStepResult {
            execution,
            computed_end: 0,
            accepted_end: 0,
            exported_checkpoints: vec![],
            page_valid_ends: vec![],
            mtp_accepted_end: None,
        },
        token_ids: vec![42],
        num_accepted_draft_tokens: 0,
        new_draft_token_ids: vec![],
    }
}

fn resident_result(
    scheduler: &Scheduler<HybridCoordinator>,
    mut result: RequestOutput,
) -> RequestOutput {
    let request = scheduler
        .running
        .iter()
        .find(|request| request.id == result.request_id)
        .unwrap();
    result.execution = request.in_flight.unwrap();
    result.work = request.work.unwrap();
    result.cache_result.execution = result.execution;
    let (_, end) = result.work.range();
    result.cache_result.computed_end = end
        - match result.work.sampling {
            SamplingMode::Verify { drafts } => drafts - result.num_accepted_draft_tokens,
            _ => 0,
        };
    result.cache_result.accepted_end = request.accepted_end.unwrap_or(
        request.token_ids.len()
            + result.token_ids.len().min(
                request
                    .max_tokens
                    .saturating_sub(request.num_generated_tokens()),
            ),
    );
    result
}

fn apply(scheduler: &mut Scheduler<HybridCoordinator>, mut output: WorkerOutput) -> WorkerOutput {
    for request in &mut scheduler.running {
        if request.in_flight.is_some() {
            request.submitted = true;
        }
    }
    output.outputs = output
        .outputs
        .into_iter()
        .map(|result| resident_result(scheduler, result))
        .collect();
    let output = scheduler.update(output).expect("test output must be valid");
    let ending = scheduler
        .running
        .iter()
        .filter(|request| request.is_finished())
        .map(|request| (request.id, request.token_ids.len()))
        .collect::<Vec<_>>();
    for (id, end) in ending {
        scheduler.finalize_request(id, end).unwrap();
    }
    let finalized = scheduler
        .running
        .iter()
        .filter(|request| request.status == RequestStatus::Finalized)
        .map(|request| request.id)
        .collect::<Vec<_>>();
    for id in finalized {
        scheduler.finish_success(id).unwrap();
    }
    output
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

fn finish_prefills(scheduler: &mut Scheduler<HybridCoordinator>) -> bool {
    let mut history = false;
    for _ in 0..10000 {
        if scheduler.num_waiting() + scheduler.num_running() == 0 {
            return history;
        }
        let batch = scheduler.schedule();
        assert!(!batch.scheduled.is_empty());
        let outputs = batch
            .scheduled
            .iter()
            .map(|row| {
                history |= row.num_computed_tokens
                    > scheduler
                        .running
                        .iter()
                        .find(|request| request.id == row.request_id)
                        .unwrap()
                        .prompt_len;
                let mut result = dummy_output(row.request_id);
                if row.work.sampling == SamplingMode::None {
                    result.token_ids.clear();
                }
                result
            })
            .collect();
        apply(scheduler, WorkerOutput { outputs });
    }
    panic!("prefill requests did not finish");
}
