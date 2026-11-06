mod offline;

use offline::run_scheduler;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;
use oh_my_vllm_zmq_worker::client::{WorkerClient, WorkerConfig};
use oh_my_vllm_zmq_worker::models::{ModelPolicy, select_model};
use tracing::{Instrument, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Cli {
    #[arg(long, env = "OH_MY_VLLM_MODEL")]
    model: PathBuf,
    #[arg(long, default_value = "/tmp/oh-my-vllm.ipc")]
    socket: PathBuf,
    #[arg(long, env = "OH_MY_VLLM_NUM_GPU_BLOCKS",
        value_parser = clap::value_parser!(u32).range(2..),
        help = "Cache blocks (default depends on model and speculative mode)")]
    num_gpu_blocks: Option<u32>,
    #[arg(long)]
    mamba_blocks: Option<u32>,
    #[arg(long)]
    host_cache_bytes: Option<u64>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    max_num_batched_tokens: Option<u32>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    max_num_seqs: Option<u32>,
    #[arg(long)]
    block_size: Option<u32>,
    #[arg(long)]
    tp: Option<u32>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    max_model_len: Option<u32>,
    #[arg(long, default_value_t = 0)]
    num_speculative_tokens: usize,
    #[arg(long)]
    speculative_mode: Option<String>,
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
        #[arg(long, conflicts_with = "tokens")]
        prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        arrival_interval: usize,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long)]
        prefix_hit: bool,
    },
    Bench(BenchArgs),
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
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long)]
    prefix_hit: bool,
    #[arg(long, default_value_t = 0)]
    arrival_interval: usize,
    #[arg(long)]
    torch_profile: Option<PathBuf>,
    #[arg(long, default_value_t = 8, requires = "torch_profile")]
    torch_profile_skip_steps: usize,
    #[arg(long, default_value_t = 2, requires = "torch_profile")]
    torch_profile_steps: usize,
}

impl Cli {
    fn worker_config(&self, model: ModelPolicy) -> Result<WorkerConfig> {
        let defaults = WorkerConfig::for_model(self.model.clone(), model);
        let mode = model.resolve_speculation(
            self.speculative_mode.as_deref(),
            self.num_speculative_tokens,
        )?;
        let config = WorkerConfig {
            socket_path: self.socket.clone(),
            num_gpu_blocks: self.num_gpu_blocks,
            mamba_blocks: self.mamba_blocks.or(defaults.mamba_blocks),
            host_cache_bytes: self.host_cache_bytes,
            block_size: self.block_size.unwrap_or(defaults.block_size),
            tensor_parallel_size: self.tp.unwrap_or(defaults.tensor_parallel_size),
            max_model_len: self.max_model_len.unwrap_or(defaults.max_model_len),
            max_num_batched_tokens: self
                .max_num_batched_tokens
                .unwrap_or(defaults.max_num_batched_tokens),
            max_num_seqs: self.max_num_seqs.unwrap_or(defaults.max_num_seqs),
            num_speculative_tokens: mode.num_tokens,
            speculative_mode: Some(mode.name.to_owned()),
            torch_profile: match &self.cmd {
                Cmd::Bench(args) => args.profile_configuration()?,
                _ => None,
            },
            ..defaults
        };
        model.validate_options(&config)?;
        Ok(config)
    }
}

fn validate_prompt_tokens(prompts: &[Vec<u32>], model: ModelPolicy) -> Result<()> {
    ensure!(
        prompts
            .iter()
            .flatten()
            .all(|&token| token < model.vocabulary_size()),
        "prompt token is outside the model vocabulary"
    );
    Ok(())
}

impl BenchArgs {
    fn profile_configuration(
        &self,
    ) -> Result<Option<oh_my_vllm_zmq_worker::protocol::BenchmarkProfile>> {
        let Some(path) = &self.torch_profile else {
            return Ok(None);
        };
        ensure!(
            (1..=4).contains(&self.batch_size)
                && self.input_len.saturating_add(self.output_len) <= 4096
                && self.batch_size.saturating_mul(self.input_len) <= 4096,
            "torch profiling is limited to small benchmarks (batch <= 4, context <= 4096)"
        );
        ensure!(
            (1..=8).contains(&self.torch_profile_steps) && self.torch_profile_skip_steps <= 10000,
            "torch profiling needs 1..8 active steps and at most 10000 skipped steps"
        );
        Ok(Some(oh_my_vllm_zmq_worker::protocol::BenchmarkProfile {
            path: path.to_string_lossy().into_owned(),
            skip_steps: self.torch_profile_skip_steps,
            steps: self.torch_profile_steps,
        }))
    }

    fn prompts(&self, max_model_len: usize, model: ModelPolicy) -> Result<Vec<Vec<u32>>> {
        ensure!(
            (1..=model.max_offline_batch_size()).contains(&self.batch_size)
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
            return self.parse_prompts(&std::fs::read_to_string(path)?, model);
        }
        let token_cycle = model.vocabulary_size().saturating_sub(1).min(32000) as usize;
        ensure!(
            token_cycle > 0,
            "synthetic prompts require a nonzero token range"
        );
        Ok((0..self.batch_size)
            .map(|request| {
                (0..self.input_len)
                    .map(|i| ((i + request * 997) % token_cycle + 1) as u32)
                    .collect()
            })
            .collect())
    }

    fn parse_prompts(&self, text: &str, model: ModelPolicy) -> Result<Vec<Vec<u32>>> {
        let prompts = text
            .lines()
            .map(|line| line.split_whitespace().map(str::parse::<u32>).collect())
            .collect::<std::result::Result<Vec<Vec<u32>>, _>>()?;
        ensure!(
            prompts.len() == self.batch_size
                && prompts.iter().all(|tokens| tokens.len() == self.input_len),
            "prompt file must match benchmark batch size and input length"
        );
        validate_prompt_tokens(&prompts, model)?;
        Ok(prompts)
    }
}

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
    let model = select_model(&cli.model)?;
    let config = cli.worker_config(model)?;
    let speculative_tokens = config.num_speculative_tokens;
    let run_prompts = cli.run_prompts(&config, model)?;
    let bench_prompts = if let Cmd::Bench(args) = &cli.cmd {
        Some(args.prompts(config.max_model_len as usize, model)?)
    } else {
        None
    };
    let client = WorkerClient::launch_with_model(config.clone(), model)
        .await
        .context("launch worker")?;
    let blocks = cli.scheduler_blocks.unwrap_or(client.logical_num_blocks);
    ensure!(
        blocks > 1 && blocks <= client.logical_num_blocks,
        "invalid scheduler pool capacity"
    );
    ensure!(client.mamba_blocks > 1, "invalid worker GDN pool capacity");
    if let Some(capacity) = config.mamba_blocks {
        ensure!(
            capacity > 1 && capacity <= client.mamba_blocks,
            "invalid scheduler GDN pool capacity"
        );
    }
    let mamba_blocks = config.mamba_blocks.unwrap_or(client.mamba_blocks);
    let mut kv = HybridCoordinator::new(blocks, config.block_size as usize, true, 0)
        .with_mamba_capacity(mamba_blocks);
    kv.set_speculative_blocks(speculative_tokens);
    let sched_max_num_seqs = config.max_num_seqs as usize;
    let sched_max_batched = config.max_num_batched_tokens as usize;
    info!(
        max_num_seqs = sched_max_num_seqs,
        max_num_batched_tokens = sched_max_batched,
        block_size = config.block_size,
        max_model_len = config.max_model_len,
        num_speculative_tokens = speculative_tokens,
        speculative_mode = config.speculative_mode.as_deref().unwrap_or("none"),
        worker_fa_pool_blocks = client.logical_num_blocks,
        worker_gdn_pool_blocks = client.mamba_blocks,
        fa_pool_blocks = blocks,
        gdn_pool_blocks = mamba_blocks,
        host_cache_bytes = client.host_cache_bytes,
        rust_sources_sha256 = env!("OH_MY_VLLM_RUST_SOURCES_SHA256"),
        "BENCH_CONFIG"
    );
    if let Some(dep4) = client.dep4_cache(kv.empty_like()) {
        run_scheduler(cli, client, dep4, run_prompts, bench_prompts).await
    } else {
        run_scheduler(cli, client, kv, run_prompts, bench_prompts).await
    }
}

impl Cli {
    fn run_prompts(
        &self,
        config: &WorkerConfig,
        model: ModelPolicy,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        if let Cmd::Run {
            tokens,
            prompt_file,
            max_tokens,
            ..
        } = &self.cmd
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
                    && prompts.len() <= model.max_offline_batch_size()
                    && prompts.iter().all(|tokens| !tokens.is_empty())
                    && *max_tokens > 0,
                "nonempty input and positive output required"
            );
            ensure!(
                prompts
                    .iter()
                    .all(|tokens| tokens.len().saturating_add(*max_tokens)
                        <= config.max_model_len as usize),
                "request exceeds context limit"
            );
            validate_prompt_tokens(&prompts, model)?;
            Ok(Some(prompts))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
fn check_benchmark_cli(model_path: &str, model: ModelPolicy) {
    let benchmark = Cli::try_parse_from([
        "omv",
        "--model",
        model_path,
        "bench",
        "--batch-size",
        "2",
        "--input-len",
        "2",
        "--output-len",
        "2",
    ])
    .unwrap();
    let Cmd::Bench(mut args) = benchmark.cmd else {
        panic!("expected benchmark arguments");
    };
    assert_eq!(
        args.prompts(262144, model).unwrap(),
        vec![vec![1, 2], vec![998, 999]]
    );
    assert!(args.parse_prompts("1 2\n3 248320", model).is_err());
    args.batch_size = 33;
    assert!(args.prompts(262144, model).is_err());
}

#[cfg(test)]
mod tests {
    use super::*;
    use oh_my_vllm_zmq_worker::models::select_model;

    fn cli(model_path: &str, flags: &[&str]) -> Cli {
        Cli::try_parse_from(
            ["omv", "--model", model_path]
                .into_iter()
                .chain(flags.iter().copied())
                .chain(["run", "--tokens", "1", "--max-tokens", "1"]),
        )
        .unwrap()
    }

    #[test]
    fn selected_model_preserves_cli_defaults_modes_and_rejections() {
        let directory = std::env::temp_dir().join(format!("omv-cli-model-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/qwen38/config.json",
        ))
        .strip_suffix(b"\n")
        .unwrap();
        std::fs::write(directory.join("config.json"), raw).unwrap();
        let model = select_model(&directory).unwrap();
        let model_path = directory.to_str().unwrap();
        for (flags, mode, count) in [
            (vec![], "none", 0),
            (vec!["--speculative-mode", "mtp"], "mtp", 4),
            (vec!["--num-speculative-tokens", "4"], "mtp", 4),
            (vec!["--speculative-mode", "none"], "none", 0),
        ] {
            let config = cli(model_path, &flags).worker_config(model).unwrap();
            assert_eq!(config.block_size, 784);
            assert_eq!(config.tensor_parallel_size, 1);
            assert_eq!(config.max_model_len, 262144);
            assert_eq!(config.max_num_batched_tokens, 32768);
            assert_eq!(config.max_num_seqs, 4);
            assert_eq!(config.mamba_blocks, Some(128));
            assert_eq!(config.num_gpu_blocks, None);
            assert_eq!(config.speculative_mode.as_deref(), Some(mode));
            assert_eq!(config.num_speculative_tokens, count);
        }
        for flags in [
            vec!["--max-model-len", "262145"],
            vec!["--block-size", "785"],
            vec!["--tp", "4"],
            vec!["--num-speculative-tokens", "1"],
            vec!["--speculative-mode", "unknown"],
            vec![
                "--speculative-mode",
                "none",
                "--num-speculative-tokens",
                "4",
            ],
        ] {
            assert!(
                cli(model_path, &flags).worker_config(model).is_err(),
                "{flags:?}"
            );
        }
        let config = cli(
            model_path,
            &[
                "--max-model-len",
                "128",
                "--max-num-seqs",
                "2",
                "--max-num-batched-tokens",
                "32",
                "--mamba-blocks",
                "16",
            ],
        )
        .worker_config(model)
        .unwrap();
        assert_eq!((config.max_model_len, config.max_num_seqs), (128, 2));
        assert_eq!(
            (config.max_num_batched_tokens, config.mamba_blocks),
            (32, Some(16))
        );
        assert!(validate_prompt_tokens(&[vec![248319]], model).is_ok());
        assert!(validate_prompt_tokens(&[vec![248320]], model).is_err());
        check_benchmark_cli(model_path, model);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
