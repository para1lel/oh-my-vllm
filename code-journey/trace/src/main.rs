use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;
use oh_my_vllm_scheduler::{Request, RequestOutput, Scheduler, SchedulerConfig, WorkerOutput};

#[derive(Debug, PartialEq)]
struct Step {
    start: usize,
    count: usize,
    computed: usize,
    outputs: usize,
}

fn trace(prompt_len: usize) -> Vec<Step> {
    let config = SchedulerConfig {
        max_num_batched_tokens: 32768,
        max_num_seqs: 1,
        enable_mtp: false,
        mtp_draft_len: 0,
    };
    let kv = HybridCoordinator::new(128, 784, false, 0).with_mamba_capacity(16);
    let mut scheduler = Scheduler::new(config, kv);
    let tokens = (0..prompt_len as u32).collect();
    assert!(scheduler.add_request(Request::new(1, tokens, 2, Vec::new())));
    let mut steps = Vec::new();
    let mut computed = 0;
    while computed < prompt_len {
        let scheduled = scheduler.schedule();
        assert!(scheduled.preempted_request_ids.is_empty());
        assert_eq!(scheduled.scheduled.len(), 1);
        let request = &scheduled.scheduled[0];
        let count = request.token_ids.len();
        assert!(count > 0);
        assert_eq!(request.num_computed_tokens, computed);
        let start = computed;
        computed += count;
        let outputs = usize::from(computed == prompt_len);
        // This CPU trace uses synthetic worker feedback for the scheduler contract.
        // Token 42 is a fixture, never model output or performance evidence.
        scheduler
            .update(WorkerOutput {
                outputs: vec![RequestOutput {
                    request_id: 1,
                    token_ids: if outputs == 1 {
                        vec![42]
                    } else {
                        Vec::new()
                    },
                    num_accepted_draft_tokens: 0,
                    new_draft_token_ids: Vec::new(),
                }],
            })
            .expect("fixture must satisfy the real scheduler output contract");
        steps.push(Step {
            start,
            count,
            computed,
            outputs,
        });
    }
    steps
}

fn main() {
    print!("{{");
    for (case, length) in [784, 785, 1568, 1569, 32768].iter().enumerate() {
        if case > 0 {
            print!(",");
        }
        print!("\"{length}\":[");
        for (index, step) in trace(*length).iter().enumerate() {
            if index > 0 {
                print!(",");
            }
            print!(
                concat!(
                    "{{\"start\":{},\"count\":{},",
                    "\"computed\":{},\"outputs\":{}}}"
                ),
                step.start, step.count, step.computed, step.outputs
            );
        }
        print!("]");
    }
    println!("}}");
}

#[cfg(test)]
mod tests {
    use super::trace;

    #[test]
    fn actual_scheduler_matches_published_boundary_examples() {
        for (length, counts) in [
            (784, vec![784]),
            (785, vec![784, 1]),
            (1568, vec![1568]),
            (1569, vec![1568, 1]),
            (32768, vec![32144, 624]),
        ] {
            let steps = trace(length);
            assert_eq!(
                steps.iter().map(|step| step.count).collect::<Vec<_>>(),
                counts
            );
            assert_eq!(steps.last().unwrap().computed, length);
            assert_eq!(steps.iter().map(|step| step.outputs).sum::<usize>(), 1);
            assert!(
                steps[..steps.len() - 1]
                    .iter()
                    .all(|step| step.outputs == 0)
            );
        }
    }
}
