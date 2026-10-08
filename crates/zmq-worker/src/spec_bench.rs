//! A paired comparison shares target weights and physical caches between methods.
use anyhow::{Result, ensure};
use clap::Args;
use oh_my_vllm_scheduler::Scheduler;
use oh_my_vllm_zmq_worker::client::WorkerClient;
use tracing::info;

const ROUNDS: usize = 3;
const WARMUPS: usize = 2;
const PAIRS: usize = 5;
const INPUT_LEN: usize = 32768;
const OUTPUT_LEN: usize = 4096;

#[derive(Args)]
pub struct SpecBenchArgs {
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
}

impl SpecBenchArgs {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [1, 2, 4].contains(&self.batch_size),
            "batch size must be 1, 2, or 4"
        );
        Ok(())
    }
}

fn order(round: usize, pair: usize) -> [&'static str; 2] {
    if (round + pair).is_multiple_of(2) {
        ["mtp", "dspark"]
    } else {
        ["dspark", "mtp"]
    }
}

pub async fn run(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
    next_id: &mut u64,
    args: SpecBenchArgs,
) -> Result<()> {
    args.validate()?;
    let prompts: Vec<Vec<u32>> = (0..args.batch_size)
        .map(|request| {
            (0..INPUT_LEN)
                .map(|i| ((i + request * 997) % 32000 + 1) as u32)
                .collect()
        })
        .collect();
    for round in 0..ROUNDS {
        for mode in ["mtp", "dspark"] {
            for index in 0..WARMUPS {
                sample(
                    client, scheduler, next_id, &prompts, round, "warmup", index, 0, mode,
                )
                .await?;
            }
        }
        for pair in 0..PAIRS {
            for (slot, mode) in order(round, pair).into_iter().enumerate() {
                sample(
                    client, scheduler, next_id, &prompts, round, "measured", pair, slot, mode,
                )
                .await?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn sample(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
    next_id: &mut u64,
    prompts: &[Vec<u32>],
    round: usize,
    phase: &str,
    index: usize,
    slot: usize,
    mode: &str,
) -> Result<()> {
    ensure!(
        scheduler.reset_prefix_cache(),
        "cache reset with active requests"
    );
    ensure!(
        scheduler.set_speculative_tokens(if mode == "mtp" {
            4
        } else {
            7
        }),
        "speculative mode switch with active requests"
    );
    client.set_speculative_mode(mode).await?;
    info!(
        round,
        phase,
        index,
        slot,
        speculative_mode = mode,
        measured = phase == "measured",
        "BENCH_PHASE"
    );
    let result = super::execute_batch(client, scheduler, next_id, prompts, OUTPUT_LEN, 0).await?;
    let output_tokens: usize = result.outputs.values().map(Vec::len).sum();
    let record = serde_json::json!({
        "round": round,
        "phase": phase,
        "index": index,
        "slot": slot,
        "speculative_mode": mode,
        "batch_size": prompts.len(),
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "output_tokens": output_tokens,
        "elapsed_s": result.elapsed,
        "output_tps": output_tokens as f64 / result.elapsed,
        "steps": result.steps,
        "ttft_s": result.ttft_s,
        "prefix_hit_tokens": result.prefix_hit_tokens,
        "initial_prefix_hit_tokens": result.initial_prefix_hit_tokens,
        "preemptions": result.preemptions,
        "proposed_draft_tokens": result.proposed_draft_tokens,
        "verified_draft_tokens": result.verified_draft_tokens,
        "accepted_draft_tokens": result.accepted_draft_tokens,
    });
    println!("SPEC_BENCH_RESULT {record}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PAIRS, ROUNDS, SpecBenchArgs, order};

    #[test]
    fn pair_orders_alternate_across_round_boundaries() {
        let orders: Vec<_> = (0..ROUNDS)
            .flat_map(|round| (0..PAIRS).map(move |pair| order(round, pair)))
            .collect();
        assert!(orders.windows(2).all(|pair| pair[0] != pair[1]));
        assert_eq!(orders[0], ["mtp", "dspark"]);
    }

    #[test]
    fn only_acceptance_batch_sizes_are_permitted() {
        for batch_size in [0, 3, 8] {
            assert!(SpecBenchArgs { batch_size }.validate().is_err());
        }
        for batch_size in [1, 2, 4] {
            assert!(SpecBenchArgs { batch_size }.validate().is_ok());
        }
    }
}
