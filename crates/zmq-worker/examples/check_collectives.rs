use std::ffi::OsString;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use oh_my_vllm_kv_cache::state::{AllocationRequest, StateArena};
use oh_my_vllm_model_contract::ModelDescriptor;
use oh_my_vllm_zmq_worker::group::GroupLimits;
use oh_my_vllm_zmq_worker::group::collectives::{
    CollectiveOperation, CollectivePlan, CollectiveType,
};
use oh_my_vllm_zmq_worker::group::protocol::{RequestWork, StepPlan};
use oh_my_vllm_zmq_worker::rank_client::{GroupClient, GroupConfig, GroupTransport};
use serde_json::json;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "none", value_parser = ["none", "exit", "hang"])]
    failure: String,
    #[arg(long)]
    sanitizer: Option<PathBuf>,
}

struct IpcDirectory(PathBuf);

impl Drop for IpcDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn descriptor(strategy: &str) -> Result<ModelDescriptor> {
    Ok(serde_json::from_value(json!({
        "schema_version": 1, "architecture": "SyntheticForGeneration", "model_type": "synthetic",
        "checkpoint_sha256": "a".repeat(64), "config_sha256": "b".repeat(64),
        "tokenizer_sha256": "c".repeat(64), "vocabulary_size": 101, "max_context_tokens": 97,
        "capabilities": ["text"], "weight_formats": ["bf16"], "speculation": [],
        "cache_schema": {"schema_version": 1, "groups": [
            {"name": "state", "kind": "model_checkpoint", "block_tokens": 1, "capacity_blocks": 4,
             "storage": {"type": "primitive", "format": "f32"},
             "layers": [0], "source": {"type": "owned"}}
        ]},
        "parallel_plan": {"strategy": strategy, "world_size": 4, "ranks": [0, 1, 2, 3],
            "request_placement": if strategy == "dep4" {"rank_partitioned"} else {"replicated"}}
    }))?)
}

fn communication() -> CollectivePlan {
    let mut operations = Vec::new();
    for dtype in [
        CollectiveType::F32,
        CollectiveType::Bf16,
        CollectiveType::F16,
        CollectiveType::I32,
        CollectiveType::I64,
        CollectiveType::U8,
    ] {
        operations.extend([
            CollectiveOperation::AllReduce { dtype, count: 8 },
            CollectiveOperation::AllGather { dtype, count: 3 },
            CollectiveOperation::ReduceScatter { dtype, count: 2 },
            CollectiveOperation::Route {
                dtype,
                counts: vec![
                    vec![1, 2, 0, 0],
                    vec![0, 0, 0, 1],
                    vec![3, 0, 0, 0],
                    vec![0; 4],
                ],
            },
            CollectiveOperation::AllReduce { dtype, count: 0 },
        ]);
    }
    CollectivePlan {
        version: 1,
        world_size: 4,
        operations,
    }
}

fn worker(arguments: &Arguments) -> Result<(PathBuf, Vec<OsString>)> {
    let python = std::env::var_os("OH_MY_VLLM_WORKER_PYTHON")
        .context("use scripts/with-env.sh for the worker interpreter")?;
    let mut options: Vec<OsString> = vec![
        "-m".into(),
        "development.operators.collective_peer".into(),
        "--failure".into(),
        arguments.failure.clone().into(),
    ];
    if let Some(sanitizer) = &arguments.sanitizer {
        let mut prefix: Vec<OsString> = [
            "--tool",
            "memcheck",
            "--error-exitcode",
            "77",
            "--target-processes",
            "all",
        ]
        .map(OsString::from)
        .into();
        prefix.push(python);
        prefix.append(&mut options);
        Ok((sanitizer.clone(), prefix))
    } else {
        Ok((python.into(), options))
    }
}

async fn check(arguments: &Arguments, strategy: &str) -> Result<()> {
    let directory = arguments.output.join(strategy);
    std::fs::create_dir(&directory)?;
    let devices: Vec<String> = std::env::var("CUDA_VISIBLE_DEVICES")?
        .split(',')
        .map(String::from)
        .collect();
    ensure!(devices.len() == 4, "use the four-GPU lease wrapper");
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
    let ipc =
        IpcDirectory(std::env::temp_dir().join(format!("omv-nccl-{}-{epoch}", std::process::id())));
    std::fs::create_dir(&ipc.0)?;
    let model = descriptor(strategy)?;
    let arena = StateArena::new(epoch, &model.cache_schema, &model.parallel_plan, 16)?;
    let (program, mut options) = worker(arguments)?;
    options.extend(["--output".into(), directory.clone().into_os_string()]);
    let mut client = GroupClient::launch(GroupConfig {
        epoch,
        descriptor: model,
        device_uuids: devices.clone(),
        transport: GroupTransport::Nccl,
        limits: GroupLimits {
            max_slots: 16,
            max_requests: 64,
            max_rows: 1024,
        },
        socket_directory: ipc.0.clone(),
        program,
        arguments: options,
        startup_timeout: Duration::from_secs(180),
        step_timeout: Duration::from_secs(if arguments.failure == "none" {
            60
        } else {
            5
        }),
    })
    .await?;
    let started = Instant::now();
    let run = async {
        for step_id in 1..=3 {
            let transaction = arena.reserve(
                step_id,
                10,
                &[AllocationRequest {
                    group: "state".into(),
                    rank: 0,
                    blocks: 3,
                }],
                &[],
            )?;
            let plan = StepPlan {
                step_id,
                collective_plan: communication(),
                requests: vec![RequestWork {
                    request_id: 10,
                    transaction_id: transaction.key().transaction_id,
                    owner_rank: 0,
                    rows: 3,
                    prefill: step_id == 1,
                }],
            };
            let result = client.execute(plan, vec![transaction]).await;
            if arguments.failure != "none" && step_id == 1 {
                ensure!(result.is_err(), "failed collective returned storage");
                let injected: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(directory.join("rank-2.json"))?)?;
                ensure!(
                    injected["injected_failure"] == arguments.failure,
                    "step failed before the intended fault injection"
                );
                ensure!(
                    arena.free_blocks("state", 0)? == 3,
                    "storage remains leased"
                );
                return Ok(());
            }
            drop(result?);
            ensure!(
                arena.free_blocks("state", 0)? == 3,
                "storage remains leased"
            );
        }
        Ok::<_, anyhow::Error>(())
    }
    .await;
    if run.is_ok() && arguments.failure == "none" {
        client.shutdown(Duration::from_secs(90)).await?;
    } else {
        client.close().await?;
    }
    run?;
    ensure!(
        !std::fs::read_dir(&ipc.0)?.any(|entry| {
            entry.is_ok_and(|entry| entry.path().extension().is_some_and(|ext| ext == "ipc"))
        }),
        "group IPC remains after closure"
    );
    let report = json!({"strategy": strategy, "devices": devices, "epoch": epoch,
        "failure": arguments.failure, "elapsed_seconds": started.elapsed().as_secs_f64(),
        "plan_sha256": communication().identity(4)?, "closed": true});
    std::fs::write(
        directory.join("controller.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{strategy}: native collective check complete");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    std::fs::create_dir(&arguments.output)?;
    let strategies: &[&str] = if arguments.failure == "none" {
        &["tp4", "tep4", "dep4"]
    } else {
        &["dep4"]
    };
    for strategy in strategies {
        check(&arguments, strategy).await?;
    }
    Ok(())
}
