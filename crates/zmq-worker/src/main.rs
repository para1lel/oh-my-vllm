//! oh-my-vllm binary — wires the Rust scheduler to the Python model worker.
//!
//! Subcommands:
//!   run   — serve a hard-coded prompt for smoke-testing the end-to-end loop.
//!   bench — run a synthetic throughput benchmark and print token/s numbers.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;
use oh_my_vllm_scheduler::{Request, Scheduler, SchedulerConfig};
use tracing::{Instrument, info};
use tracing_subscriber::EnvFilter;

use oh_my_vllm_zmq_worker::client::{WorkerClient, WorkerConfig};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "oh-my-vllm", about = "Rust scheduler + Python model worker")]
struct Cli {
    /// Path to the model weights directory.
    #[arg(long, default_value = "/data0/shared/Qwen3.8-27B-FP8")]
    model: PathBuf,

    /// ZMQ IPC socket path (without ipc:// prefix).
    #[arg(long, default_value = "/tmp/oh-my-vllm.ipc")]
    socket: PathBuf,

    /// Number of GPU KV cache blocks to pre-allocate.
    #[arg(long, default_value_t = 1024)]
    num_gpu_blocks: u32,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 784)]
    block_size: u32,

    /// Tensor parallel size (1 = single GPU).
    #[arg(long, default_value_t = 1)]
    tp: u32,

    /// Maximum sequence length.
    #[arg(long, default_value_t = 65536)]
    max_model_len: u32,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a single prompt end-to-end and print the generated tokens.
    Run {
        /// Prompt token ids (space-separated integers).
        #[arg(long, num_args = 1.., value_delimiter = ' ')]
        tokens: Vec<u32>,

        /// Maximum output tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },

    /// Synthetic throughput benchmark.
    Bench {
        /// Batch size (number of concurrent requests).
        #[arg(long, default_value_t = 1)]
        batch_size: usize,

        /// Input sequence length in tokens.
        #[arg(long, default_value_t = 512)]
        input_len: usize,

        /// Output sequence length in tokens.
        #[arg(long, default_value_t = 128)]
        output_len: usize,

        /// Warm-up steps before timing.
        #[arg(long, default_value_t = 2)]
        warmup: usize,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
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
    let span = tracing::info_span!("inference", run_id);
    run().instrument(span).await
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    let worker_cfg = WorkerConfig {
        model_path: cli.model,
        socket_path: cli.socket,
        num_gpu_blocks: cli.num_gpu_blocks,
        block_size: cli.block_size,
        tensor_parallel_size: cli.tp,
        max_model_len: cli.max_model_len,
        ..WorkerConfig::default()
    };

    info!("Launching Python model worker…");
    let mut worker_client = WorkerClient::launch(worker_cfg)
        .await
        .context("failed to launch Python worker")?;
    info!("Python worker ready");

    match cli.cmd {
        Cmd::Run { tokens, max_tokens } => {
            run_single(
                &mut worker_client,
                tokens,
                max_tokens,
                cli.num_gpu_blocks,
                cli.block_size,
            )
            .await?
        }
        Cmd::Bench {
            batch_size,
            input_len,
            output_len,
            warmup,
        } => {
            bench(
                &mut worker_client,
                batch_size,
                input_len,
                output_len,
                warmup,
                cli.num_gpu_blocks,
                cli.block_size,
            )
            .await?
        }
    }

    worker_client.shutdown().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_scheduler(num_gpu_blocks: u32, block_size: u32) -> Scheduler {
    let kv = HybridCoordinator::new(
        num_gpu_blocks,
        block_size as usize,
        true, // enable_caching
        0,    // watermark_blocks
    );
    Scheduler::new(SchedulerConfig::default(), kv)
}

/// Smoke-test: run one request to completion and print the output token ids.
async fn run_single(
    client: &mut WorkerClient,
    tokens: Vec<u32>,
    max_tokens: usize,
    num_gpu_blocks: u32,
    block_size: u32,
) -> Result<()> {
    let mut sched = make_scheduler(num_gpu_blocks, block_size);
    let req = Request::new(1, tokens.clone(), max_tokens, Vec::new());
    client
        .register_request(1, tokens)
        .await
        .context("register_request")?;
    sched.add_request(req);

    let mut output_tokens: Vec<u32> = Vec::new();
    loop {
        let step = sched.schedule();
        if step.scheduled.is_empty() {
            break;
        }
        let worker_out = client
            .execute_one_step(
                &step.scheduled,
                &step.finished_request_ids,
                &step.preempted_request_ids,
                step.num_batched_tokens,
            )
            .await
            .context("execute_one_step")?;

        for o in &worker_out.outputs {
            output_tokens.push(o.next_token_id);
        }
        sched.update(worker_out);

        if sched.num_running() == 0 && sched.num_waiting() == 0 {
            break;
        }
    }

    println!("output token ids: {:?}", output_tokens);
    Ok(())
}

/// Throughput benchmark: run `batch_size` requests of fixed length and report tokens/s.
async fn bench(
    client: &mut WorkerClient,
    batch_size: usize,
    input_len: usize,
    output_len: usize,
    warmup: usize,
    num_gpu_blocks: u32,
    block_size: u32,
) -> Result<()> {
    info!("Warming up ({warmup} runs)…");
    for _ in 0..warmup {
        bench_run(
            client,
            batch_size,
            input_len,
            output_len,
            num_gpu_blocks,
            block_size,
        )
        .await?;
    }

    info!("Benchmarking…");
    let t0 = Instant::now();
    let output_tokens = bench_run(
        client,
        batch_size,
        input_len,
        output_len,
        num_gpu_blocks,
        block_size,
    )
    .await?;
    let elapsed = t0.elapsed().as_secs_f64();

    let input_tokens = batch_size * input_len;
    let total_tokens = input_tokens + output_tokens;
    println!(
        "batch_size={batch_size}  input_len={input_len}  output_len={output_len}\n\
         output tokens : {output_tokens}\n\
         total tokens  : {total_tokens}\n\
         wall time     : {elapsed:.3}s\n\
         throughput    : {:.1} output tok/s  |  {:.1} total tok/s",
        output_tokens as f64 / elapsed,
        total_tokens as f64 / elapsed,
    );
    Ok(())
}

async fn bench_run(
    client: &mut WorkerClient,
    batch_size: usize,
    input_len: usize,
    output_len: usize,
    num_gpu_blocks: u32,
    block_size: u32,
) -> Result<usize> {
    let mut sched = make_scheduler(num_gpu_blocks, block_size);
    for i in 0..batch_size {
        let id = (i + 1) as u64;
        let tokens: Vec<u32> = (0..input_len as u32).collect();
        let req = Request::new(id, tokens.clone(), output_len, Vec::new());
        client.register_request(id, tokens).await?;
        sched.add_request(req);
    }
    let mut total_output = 0usize;
    loop {
        let step = sched.schedule();
        if step.scheduled.is_empty() {
            break;
        }
        let worker_out = client
            .execute_one_step(
                &step.scheduled,
                &step.finished_request_ids,
                &step.preempted_request_ids,
                step.num_batched_tokens,
            )
            .await?;
        total_output += worker_out.outputs.len();
        sched.update(worker_out);
        if sched.num_running() == 0 && sched.num_waiting() == 0 {
            break;
        }
    }
    Ok(total_output)
}
