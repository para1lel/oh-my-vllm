//! Direct inference driver and reproducible fixed-token benchmark.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;
use oh_my_vllm_scheduler::{Request, Scheduler, SchedulerConfig};
use oh_my_vllm_zmq_worker::client::{WorkerClient, WorkerConfig};
use tracing::{Instrument, info};
use tracing_subscriber::EnvFilter;

mod spec_bench;

#[derive(Parser)]
struct Cli {
    #[arg(long, env = "OH_MY_VLLM_MODEL")]
    model: PathBuf,
    #[arg(long, default_value = "/tmp/oh-my-vllm.ipc")]
    socket: PathBuf,
    #[arg(long, default_value_t = 1024)]
    num_gpu_blocks: u32,
    /// Optional independent GDN-state capacity (including immutable zero slot).
    #[arg(long)]
    mamba_blocks: Option<u32>,
    #[arg(long, default_value_t = 784)]
    block_size: u32,
    #[arg(long, default_value_t = 1)]
    tp: u32,
    #[arg(long, default_value_t = 65536)]
    max_model_len: u32,
    #[arg(long, default_value_t = 0)]
    num_speculative_tokens: usize,
    /// Explicit speculative algorithm; the legacy token count still selects MTP4.
    #[arg(long, value_parser = ["none", "mtp", "dspark"])]
    speculative_mode: Option<String>,
    /// Separate DSpark checkpoint directory.
    #[arg(long, env = "OH_MY_VLLM_DRAFT_MODEL")]
    draft_model: Option<PathBuf>,
    /// DSpark cumulative prefix confidence cutoff; zero keeps all legal candidates.
    #[arg(long, default_value_t = 0.2)]
    dspark_confidence_threshold: f64,
    /// Restrict Rust's pool for preemption tests; cannot exceed worker capacity.
    #[arg(long)]
    scheduler_blocks: Option<u32>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Run {
        #[arg(long, num_args = 1..)]
        tokens: Vec<u32>,
        /// Whitespace-separated token IDs, one request per line.
        #[arg(long, conflicts_with = "tokens")]
        prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        arrival_interval: usize,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Seed this prompt first and require a real prefix-cache hit.
        #[arg(long)]
        prefix_hit: bool,
    },
    Bench(BenchArgs),
    /// Interleaved native MTP4 / DSpark comparison on one shared target model.
    SpecBench(spec_bench::SpecBenchArgs),
    Serve(oh_my_vllm_zmq_worker::serving::ServeArgs),
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 32768)]
    input_len: usize,
    #[arg(long, default_value_t = 4096)]
    output_len: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    /// Whitespace-separated token IDs, one equal-length request per line.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long)]
    prefix_hit: bool,
    /// Admit one request every N steps to exercise continuous batching.
    #[arg(long, default_value_t = 0)]
    arrival_interval: usize,
}

impl BenchArgs {
    fn prompts(&self, max_model_len: usize) -> Result<Vec<Vec<u32>>> {
        ensure!(
            (1..=32).contains(&self.batch_size)
                && self.input_len > 0
                && self.output_len > 0
                && self.repetitions > 0,
            "invalid benchmark dimensions"
        );
        ensure!(
            self.input_len.saturating_add(self.output_len) <= max_model_len,
            "workload exceeds context limit"
        );
        if let Some(path) = &self.prompt_file {
            return self.parse_prompts(&std::fs::read_to_string(path)?);
        }
        // Preserve the acceptance workload's deterministic valid token IDs.
        Ok((0..self.batch_size)
            .map(|request| {
                (0..self.input_len)
                    .map(|i| ((i + request * 997) % 32000 + 1) as u32)
                    .collect()
            })
            .collect())
    }

    fn parse_prompts(&self, text: &str) -> Result<Vec<Vec<u32>>> {
        let prompts = text
            .lines()
            .map(|line| line.split_whitespace().map(str::parse::<u32>).collect())
            .collect::<std::result::Result<Vec<Vec<u32>>, _>>()?;
        ensure!(
            prompts.len() == self.batch_size
                && prompts.iter().all(|tokens| tokens.len() == self.input_len),
            "prompt file must match benchmark batch size and input length"
        );
        ensure!(
            prompts.iter().flatten().all(|&token| token < 248_320),
            "prompt token is outside the model vocabulary"
        );
        Ok(prompts)
    }
}

// The controller has one inference stream; keep IPC wakeups on the same thread.
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    let run_id = std::env::var("OH_MY_VLLM_RUN_ID")
        .unwrap_or_else(|_| format!("pid-{}", std::process::id()));
    run()
        .instrument(tracing::info_span!("inference", run_id))
        .await
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.block_size == 784 && cli.tp == 1,
        "requires block_size=784 and TP=1"
    );
    ensure!(cli.num_gpu_blocks >= 6, "physical cache is too small");
    ensure!(
        cli.dspark_confidence_threshold.is_finite()
            && (0.0..1.0).contains(&cli.dspark_confidence_threshold),
        "DSpark confidence threshold must be in [0,1)"
    );
    let comparison = matches!(&cli.cmd, Cmd::SpecBench(_));
    let mode = cli.speculative_mode.clone().unwrap_or_else(|| {
        if cli.num_speculative_tokens == 4 || comparison {
            "mtp"
        } else {
            "none"
        }
        .to_owned()
    });
    let speculative_tokens = match mode.as_str() {
        "none" => 0,
        "mtp" => 4,
        "dspark" => 7,
        _ => unreachable!("Clap validates the explicit mode"),
    };
    ensure!(
        cli.num_speculative_tokens == 0 || cli.num_speculative_tokens == speculative_tokens,
        "speculative token count disagrees with mode"
    );
    if let Cmd::SpecBench(args) = &cli.cmd {
        args.validate()?;
        ensure!(
            cli.max_model_len >= 36864,
            "comparison requires at least 36864 tokens"
        );
    }
    // Reject malformed input before loading the model or allocating GPU memory.
    let run_prompts = if let Cmd::Run {
        tokens,
        prompt_file,
        max_tokens,
        ..
    } = &cli.cmd
    {
        let prompts = if let Some(path) = prompt_file {
            std::fs::read_to_string(path)?
                .lines()
                .map(|line| line.split_whitespace().map(str::parse::<u32>).collect())
                .collect::<std::result::Result<Vec<Vec<u32>>, _>>()?
        } else {
            vec![tokens.clone()]
        };
        ensure!(
            !prompts.is_empty()
                && prompts.len() <= 32
                && prompts.iter().all(|tokens| !tokens.is_empty())
                && *max_tokens > 0,
            "nonempty input and positive output required"
        );
        ensure!(
            prompts.iter().all(
                |tokens| tokens.len().saturating_add(*max_tokens) <= cli.max_model_len as usize
            ),
            "request exceeds context limit"
        );
        ensure!(
            prompts.iter().flatten().all(|&token| token < 248_320),
            "prompt token is outside the model vocabulary"
        );
        Some(prompts)
    } else {
        None
    };
    let bench_prompts = if let Cmd::Bench(args) = &cli.cmd {
        Some(args.prompts(cli.max_model_len as usize)?)
    } else {
        None
    };
    let mut client = WorkerClient::launch(WorkerConfig {
        model_path: cli.model,
        socket_path: cli.socket,
        num_gpu_blocks: cli.num_gpu_blocks,
        mamba_blocks: cli.mamba_blocks,
        block_size: cli.block_size,
        tensor_parallel_size: cli.tp,
        max_model_len: cli.max_model_len,
        num_speculative_tokens: speculative_tokens,
        speculative_mode: Some(mode.clone()),
        draft_model_path: cli.draft_model,
        comparison,
        dspark_confidence_threshold: cli.dspark_confidence_threshold,
        ..WorkerConfig::default()
    })
    .await
    .context("launch worker")?;
    let blocks = cli.scheduler_blocks.unwrap_or(client.logical_num_blocks);
    ensure!(
        blocks > 1 && blocks <= client.logical_num_blocks,
        "invalid scheduler pool capacity"
    );
    ensure!(client.mamba_blocks > 1, "invalid worker GDN pool capacity");
    if let Some(capacity) = cli.mamba_blocks {
        ensure!(
            capacity > 1 && capacity <= client.mamba_blocks,
            "invalid scheduler GDN pool capacity"
        );
    }
    let mut kv = HybridCoordinator::new(blocks, 784, true, 0);
    if let Some(capacity) = cli.mamba_blocks {
        kv = kv.with_mamba_capacity(capacity);
    }
    kv.set_speculative_blocks(speculative_tokens);
    let sched_max_num_seqs = 32.min((cli.num_gpu_blocks - 1) as usize);
    let sched_max_batched: usize = SchedulerConfig::default().max_num_batched_tokens;
    info!(
        max_num_seqs = sched_max_num_seqs,
        max_num_batched_tokens = sched_max_batched,
        block_size = cli.block_size,
        max_model_len = cli.max_model_len,
        num_speculative_tokens = speculative_tokens,
        speculative_mode = if comparison {
            "comparison"
        } else {
            mode.as_str()
        },
        comparison,
        dspark_confidence_threshold = cli.dspark_confidence_threshold,
        worker_fa_pool_blocks = client.logical_num_blocks,
        worker_gdn_pool_blocks = client.mamba_blocks,
        fa_pool_blocks = blocks,
        gdn_pool_blocks = cli.mamba_blocks.unwrap_or(blocks),
        rust_sources_sha256 = env!("OH_MY_VLLM_RUST_SOURCES_SHA256"),
        "BENCH_CONFIG"
    );
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_seqs: sched_max_num_seqs,
            enable_mtp: speculative_tokens > 0,
            mtp_draft_len: speculative_tokens,
            ..SchedulerConfig::default()
        },
        kv,
    );
    if let Cmd::Serve(args) = cli.cmd {
        return oh_my_vllm_zmq_worker::serving::serve(client, scheduler, args).await;
    }
    let mut next_id = 1;
    match cli.cmd {
        Cmd::Serve(_) => unreachable!(),
        Cmd::SpecBench(args) => {
            spec_bench::run(&mut client, &mut scheduler, &mut next_id, args).await?;
        }
        Cmd::Run {
            tokens: _,
            prompt_file: _,
            arrival_interval,
            max_tokens,
            prefix_hit,
        } => {
            let prompts = run_prompts.expect("validated Run input");
            if prefix_hit {
                execute_batch(&mut client, &mut scheduler, &mut next_id, &prompts, 1, 0).await?;
            }
            let result = execute_batch(
                &mut client,
                &mut scheduler,
                &mut next_id,
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
        }
        Cmd::Bench(args) => {
            let prompts = bench_prompts.expect("validated benchmark input");
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
                    execute_batch(&mut client, &mut scheduler, &mut next_id, &prompts, 1, 0)
                        .await?;
                }
                let result = execute_batch(
                    &mut client,
                    &mut scheduler,
                    &mut next_id,
                    &prompts,
                    args.output_len,
                    args.arrival_interval,
                )
                .await?;
                {
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
                }
            }
        }
    }
    client.shutdown().await?;
    Ok(())
}

struct BatchResult {
    outputs: BTreeMap<u64, Vec<u32>>,
    elapsed: f64,
    ttft_s: Vec<f64>,
    request_phases: Vec<RequestPhases>,
    cleanup_s: f64,
    steps: usize,
    prefix_hit_tokens: usize,
    initial_prefix_hit_tokens: usize,
    preemptions: usize,
    proposed_draft_tokens: usize,
    accepted_draft_tokens: usize,
    /// Candidate tokens actually present in executed verification inputs.
    verified_draft_tokens: usize,
}

/// Caller-observed token boundaries, relative to the batch monotonic clock.
#[derive(serde::Serialize)]
struct RequestPhases {
    request_id: u64,
    /// Submission precedes registration and includes the request's queue time.
    submitted_s: f64,
    /// Receipt of the first kept output token after scheduler validation.
    first_token_s: f64,
    /// Receipt of the last kept output token, before finished-request cleanup.
    last_token_s: f64,
    first_step: usize,
    last_step: usize,
}

async fn execute_batch(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler,
    next_id: &mut u64,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    arrival_interval: usize,
) -> Result<BatchResult> {
    let started = Instant::now();
    let mut outputs = BTreeMap::<u64, Vec<u32>>::new();
    let mut first_tokens = BTreeMap::<u64, f64>::new();
    let mut submitted = BTreeMap::<u64, f64>::new();
    let mut last_tokens = BTreeMap::<u64, f64>::new();
    let mut first_steps = BTreeMap::new();
    let mut last_steps = BTreeMap::new();
    let cleanup_s;
    let mut pending = prompts.iter().peekable();
    let mut steps: usize = 0;
    let mut prefix_hit_tokens = 0;
    let mut initial_prefix_hit_tokens = 0;
    let mut awaiting_first_schedule = BTreeSet::new();
    let mut preemptions = 0;
    let mut proposed_draft_tokens = 0;
    let mut accepted_draft_tokens = 0;
    let mut verified_draft_tokens = 0;
    let mut prompt_lengths = BTreeMap::new();
    loop {
        let idle = scheduler.num_running() + scheduler.num_waiting() == 0;
        if arrival_interval == 0 || idle || steps.is_multiple_of(arrival_interval) {
            for prompt in pending.by_ref() {
                let id = *next_id;
                *next_id += 1;
                submitted.insert(
                    id,
                    if arrival_interval == 0 {
                        0.0
                    } else {
                        started.elapsed().as_secs_f64()
                    },
                );
                if !scheduler.add_request(Request::new(id, prompt.clone(), max_tokens, vec![])) {
                    tracing::error!(id, "prompt rejected: exceeds pool capacity");
                    continue;
                }
                client
                    .register_request_with_limit(id, prompt.clone(), max_tokens)
                    .await;
                outputs.insert(id, vec![]);
                prompt_lengths.insert(id, prompt.len());
                awaiting_first_schedule.insert(id);
                if arrival_interval > 0 {
                    break;
                }
            }
        }
        let step = scheduler.schedule();
        for request in &step.scheduled {
            let known = prompt_lengths[&request.request_id] + outputs[&request.request_id].len();
            verified_draft_tokens +=
                (request.num_computed_tokens + request.token_ids.len()).saturating_sub(known);
        }
        prefix_hit_tokens += step.cache_hit_tokens;
        if !awaiting_first_schedule.is_empty() {
            for request in &step.scheduled {
                if awaiting_first_schedule.remove(&request.request_id) {
                    initial_prefix_hit_tokens += request.num_computed_tokens;
                }
            }
        }
        preemptions += step.preempted_request_ids.len();
        if step.scheduled.is_empty() {
            ensure!(
                scheduler.num_running() + scheduler.num_waiting() == 0,
                "scheduler stalled with unfinished requests"
            );
        }
        let result = client
            .execute_one_step(
                &step.scheduled,
                &step.finished_request_ids,
                &step.preempted_request_ids,
                step.num_batched_tokens,
            )
            .await?;
        if !step.scheduled.is_empty() {
            steps += 1;
            for output in &result.outputs {
                proposed_draft_tokens += output.new_draft_token_ids.len();
                accepted_draft_tokens += output.num_accepted_draft_tokens;
            }
            let result = scheduler.update(result)?;
            for output in result.outputs {
                if !output.token_ids.is_empty() {
                    let received = started.elapsed().as_secs_f64();
                    first_tokens.entry(output.request_id).or_insert(received);
                    first_steps.entry(output.request_id).or_insert(steps);
                    last_tokens.insert(output.request_id, received);
                    last_steps.insert(output.request_id, steps);
                }
                outputs
                    .get_mut(&output.request_id)
                    .context("unexpected request ID")?
                    .extend(output.token_ids);
            }
        }
        if scheduler.num_running() + scheduler.num_waiting() == 0 && pending.peek().is_none() {
            let cleanup = scheduler.schedule();
            let cleanup_started = Instant::now();
            client
                .execute_one_step(&[], &cleanup.finished_request_ids, &[], 0)
                .await?;
            cleanup_s = cleanup_started.elapsed().as_secs_f64();
            break;
        }
    }
    ensure!(
        outputs.len() == prompts.len() && outputs.values().all(|tokens| tokens.len() == max_tokens),
        "incomplete output"
    );
    let elapsed = started.elapsed().as_secs_f64();
    info!(
        steps,
        elapsed, prefix_hit_tokens, initial_prefix_hit_tokens, preemptions, "batch_complete"
    );
    ensure!(
        first_tokens.len() == outputs.len(),
        "missing first-token timestamp"
    );
    Ok(BatchResult {
        ttft_s: first_tokens
            .iter()
            .map(|(id, first)| first - submitted[id])
            .collect(),
        request_phases: first_tokens
            .iter()
            .map(|(id, first)| RequestPhases {
                request_id: *id,
                submitted_s: submitted[id],
                first_token_s: *first,
                last_token_s: last_tokens[id],
                first_step: first_steps[id],
                last_step: last_steps[id],
            })
            .collect(),
        cleanup_s,
        outputs,
        elapsed,
        steps,
        prefix_hit_tokens,
        initial_prefix_hit_tokens,
        preemptions,
        proposed_draft_tokens,
        accepted_draft_tokens,
        verified_draft_tokens,
    })
}

#[cfg(test)]
mod benchmark_input_tests {
    use super::BenchArgs;

    fn args() -> BenchArgs {
        BenchArgs {
            batch_size: 2,
            input_len: 2,
            output_len: 3,
            warmup: 2,
            repetitions: 5,
            prompt_file: None,
            prefix_hit: false,
            arrival_interval: 0,
        }
    }

    #[test]
    fn custom_prompts_preserve_each_request_and_reject_invalid_shapes() {
        let args = args();
        assert_eq!(
            args.parse_prompts("7 0\n248319 9\n").unwrap(),
            vec![vec![7, 0], vec![248319, 9]]
        );
        for text in [
            "7 0\n",
            "7 0\n\n",
            "7 0 1\n8 9\n",
            "7 0\n8 248320\n",
            "7 x\n8 9\n",
        ] {
            assert!(args.parse_prompts(text).is_err());
        }
    }

    #[test]
    fn synthetic_acceptance_inputs_and_capacity_validation_stay_stable() {
        assert_eq!(args().prompts(5).unwrap(), vec![vec![1, 2], vec![998, 999]]);
        assert!(args().prompts(4).is_err());
        let mut invalid = args();
        invalid.repetitions = 0;
        assert!(invalid.prompts(5).is_err());
        invalid.repetitions = 1;
        invalid.batch_size = 33;
        assert!(invalid.prompts(5).is_err());
    }
}
