use super::tests::{checkpoint, page};
use super::*;
use crate::{Request, Scheduler, SchedulerConfig, execution::SamplingMode, policy::CachePolicy};

#[test]
fn ordinary_and_mtp_batches_leave_room_for_context_at_cold_and_resumed_boundaries() {
    for mtp in [false, true] {
        for sequences in [1, 4] {
            for resume in [0, 8192] {
                verify_batch(mtp, sequences, resume);
            }
        }
    }
}

fn verify_batch(mtp: bool, sequences: usize, resume: usize) {
    let config = SchedulerConfig {
        max_num_batched_tokens: 32768,
        max_num_seqs: sequences,
        enable_mtp: mtp,
        mtp_draft_len: 4,
    };
    let mut cache = QwenResidentCache::new(32, sequences, 262144, 32768);
    let rows = 32768 / sequences;
    if resume > 0 {
        let seed = Request::new(100, (0..(resume + rows) as u32).collect(), 1, vec![]);
        let id = checkpoint(&mut cache, &seed, resume);
        if mtp {
            cache.mtp_contexts.insert(id, resume - 1);
        }
        for end in (128..=resume).step_by(128) {
            page(&mut cache, &seed, end);
        }
    }
    let expected = cache.batch_token_limit(&config);
    let mut scheduler = Scheduler::new(config, cache);
    for id in 1..=sequences as u64 {
        assert!(scheduler.add_request(Request::new(
            id,
            (0..(resume + rows) as u32).collect(),
            1,
            vec![]
        )));
    }
    let batch = scheduler.schedule();
    assert_eq!(batch.num_batched_tokens, expected);
    assert_eq!(batch.scheduled.len(), sequences);
    let context_upper = batch
        .scheduled
        .iter()
        .map(|row| {
            row.token_ids.len()
                + usize::from(
                    mtp && row.num_computed_tokens > 0 && row.work.sampling != SamplingMode::None,
                )
        })
        .sum::<usize>();
    assert!(context_upper <= 32768);
    assert!(
        batch
            .scheduled
            .iter()
            .all(|row| row.num_computed_tokens == resume)
    );
    scheduler.validate_schedule(&batch).unwrap();
}
