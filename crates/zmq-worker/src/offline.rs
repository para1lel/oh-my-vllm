mod batch;

use crate::{BenchArgs, Cli, Cmd};
use anyhow::{Result, ensure};
use batch::{BatchResult, execute_batch};
use oh_my_vllm_scheduler::{Scheduler, SchedulerConfig};
use oh_my_vllm_zmq_worker::client::WorkerClient;
use oh_my_vllm_zmq_worker::models::ServiceCache;
use tracing::info;

pub(super) async fn run_scheduler<P: ServiceCache>(
    cli: Cli,
    mut client: WorkerClient,
    kv: P,
    run_prompts: Option<Vec<Vec<u32>>>,
    bench_prompts: Option<Vec<Vec<u32>>>,
) -> Result<()> {
    kv.attach_client(&mut client);
    let config = client.configuration();
    let speculative_tokens = config.num_speculative_tokens;
    let mode = config
        .speculative_mode
        .clone()
        .expect("resolved speculative mode");
    let sched_max_num_seqs = config.max_num_seqs as usize;
    let sched_max_batched = config.max_num_batched_tokens as usize;
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_seqs: sched_max_num_seqs,
            max_num_batched_tokens: sched_max_batched,
            enable_mtp: speculative_tokens > 0,
            mtp_draft_len: speculative_tokens,
        },
        kv,
    );
    if let Cmd::Serve(args) = cli.cmd {
        return oh_my_vllm_zmq_worker::serving::serve(client, scheduler, args).await;
    }
    let mut next_id = 1;
    match cli.cmd {
        Cmd::Serve(_) => unreachable!(),
        Cmd::Run {
            tokens: _,
            prompt_file: _,
            arrival_interval,
            max_tokens,
            prefix_hit,
        } => {
            run_inputs(
                &mut client,
                &mut scheduler,
                &mut next_id,
                run_prompts.expect("validated Run input"),
                max_tokens,
                arrival_interval,
                prefix_hit,
            )
            .await?;
        }
        Cmd::Bench(args) => {
            benchmark(
                &mut client,
                &mut scheduler,
                &mut next_id,
                &args,
                &bench_prompts.expect("validated benchmark input"),
                &mode,
            )
            .await?;
        }
    }
    client.shutdown().await?;
    Ok(())
}

async fn run_inputs<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
    next_id: &mut u64,
    prompts: Vec<Vec<u32>>,
    max_tokens: usize,
    arrival_interval: usize,
    prefix_hit: bool,
) -> Result<()> {
    if prefix_hit {
        execute_batch(client, scheduler, next_id, &prompts, 1, 0).await?;
    }
    let result = execute_batch(
        client,
        scheduler,
        next_id,
        &prompts,
        max_tokens,
        arrival_interval,
    )
    .await?;
    ensure!(
        !prefix_hit || result.initial_prefix_hit_tokens > 0,
        "prompt did not hit prefix cache"
    );
    println!(
        "output token ids: {:?}",
        result.outputs.values().next().unwrap()
    );
    println!(
        "output batches: {:?}",
        result.outputs.values().collect::<Vec<_>>()
    );
    println!(
        concat!(
            "batch stats: {{\"steps\":{},\"prefix_hit_tokens\":{},",
            "\"initial_prefix_hit_tokens\":{},\"preemptions\":{},",
            "\"accepted_draft_tokens\":{}}}"
        ),
        result.steps,
        result.prefix_hit_tokens,
        result.initial_prefix_hit_tokens,
        result.preemptions,
        result.accepted_draft_tokens
    );
    Ok(())
}

async fn benchmark<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
    next_id: &mut u64,
    args: &BenchArgs,
    prompts: &[Vec<u32>],
    mode: &str,
) -> Result<()> {
    for iteration in 0..args.warmup + args.repetitions {
        info!(
            iteration,
            measured = iteration >= args.warmup,
            "BENCH_PHASE"
        );
        ensure!(
            scheduler.reset_prefix_cache(),
            "cache reset with live requests"
        );
        if args.prefix_hit {
            execute_batch(client, scheduler, next_id, prompts, 1, 0).await?;
        }
        let result = execute_batch(
            client,
            scheduler,
            next_id,
            prompts,
            args.output_len,
            args.arrival_interval,
        )
        .await?;
        report_benchmark(args, &result, iteration, mode)?;
    }
    Ok(())
}

fn report_benchmark(
    args: &BenchArgs,
    result: &BatchResult,
    iteration: usize,
    mode: &str,
) -> Result<()> {
    let count: usize = result.outputs.values().map(Vec::len).sum();
    println!(
        concat!(
            "{} {{\"speculative_mode\":\"{}\",",
            "\"batch_size\":{},\"input_len\":{},",
            "\"output_len\":{},\"output_tokens\":{},\"elapsed_s\":{},",
            "\"output_tps\":{},\"steps\":{},\"prefix_hit_tokens\":{},",
            "\"initial_prefix_hit_tokens\":{},\"preemptions\":{},",
            "\"proposed_draft_tokens\":{},\"verified_draft_tokens\":{},",
            "\"accepted_draft_tokens\":{},",
            "\"ttft_s\":{:?},\"request_phases\":{},",
            "\"cleanup_s\":{}}}"
        ),
        if iteration >= args.warmup {
            "BENCH_RESULT"
        } else {
            "WARMUP_RESULT"
        },
        mode,
        args.batch_size,
        args.input_len,
        args.output_len,
        count,
        result.elapsed,
        count as f64 / result.elapsed,
        result.steps,
        result.prefix_hit_tokens,
        result.initial_prefix_hit_tokens,
        result.preemptions,
        result.proposed_draft_tokens,
        result.verified_draft_tokens,
        result.accepted_draft_tokens,
        result.ttft_s,
        serde_json::to_string(&result.request_phases)?,
        result.cleanup_s
    );
    Ok(())
}
