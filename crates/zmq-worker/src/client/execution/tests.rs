use super::*;
use crate::client::WorkerConfig;
use crate::client::fixture::TestWorker;
use crate::models::{ModelFamily, ModelPolicy};
use crate::protocol::{EndpointRef, ExecuteResultMsg, RequestResultMsg, WorkerMessage};
use oh_my_vllm_kv_cache::HybridCoordinator;
use oh_my_vllm_scheduler::execution::{
    ExecutionKey, SamplingMode, WorkAction, WorkPurpose, WorkSpec,
};
use oh_my_vllm_scheduler::policy::resident::ResidentState;
use std::sync::{Arc, Mutex};
use zeromq::prelude::{SocketRecv, SocketSend};

#[derive(Clone, Copy)]
enum Fault {
    None,
    Identity,
    Resources,
    RowError,
}

async fn worker() -> TestWorker {
    TestWorker::launch(WorkerConfig {
        num_gpu_blocks: Some(4096),
        mamba_blocks: Some(64),
        ..WorkerConfig::for_model(
            "/synthetic-qwen".into(),
            ModelPolicy::for_family(ModelFamily::Qwen38),
        )
    })
    .await
    .unwrap()
}

fn row(id: u64, step: u64, sampling: SamplingMode) -> ScheduledRequest<ResidentState> {
    ScheduledRequest {
        request_id: id,
        execution: ExecutionKey {
            worker_epoch: 1,
            step_id: step,
            request_id: id,
            lease_id: 1,
        },
        work: WorkSpec {
            purpose: WorkPurpose::Advance,
            action: WorkAction::Forward { start: 0, end: 2 },
            sampling,
        },
        token_ids: vec![1, 2],
        prefill_token_ids: None,
        num_computed_tokens: 0,
        accepted_end: 4,
        state: ResidentState::default(),
    }
}

fn result(row: &ScheduledRequest<ResidentState>, count: usize) -> RequestResultMsg {
    RequestResultMsg {
        request_id: row.request_id,
        lease_id: row.execution.lease_id,
        purpose: row.work.purpose,
        action: codec::action(row.work.action),
        error: None,
        token_ids: if row.work.sampling == SamplingMode::Target {
            vec![42]
        } else {
            vec![]
        },
        num_accepted_draft_tokens: 0,
        new_draft_token_ids: vec![],
        text: String::new(),
        finish_reason: None,
        reasoning_tokens: count,
        computed_end: 2,
        accepted_end: row.accepted_end as u32,
        validated_exports: vec![],
        committed_page_valid_ends: vec![],
        mtp_accepted_endpoint: None,
    }
}

async fn exchange(
    worker: &mut TestWorker,
    rows: &[ScheduledRequest<ResidentState>],
    finished: &[u64],
    preempted: &[u64],
    count: usize,
    fault: Fault,
) -> Result<WorkerOutput> {
    let tokens = rows.iter().map(|row| row.token_ids.len()).sum();
    let execute = worker
        .client
        .execute_one_step::<HybridCoordinator>(rows, finished, preempted, tokens);
    let peer = async {
        let message = worker.socket.recv().await.unwrap();
        let RustMessage::Execute(sent) = rmp_serde::from_slice(message.get(0).unwrap()).unwrap()
        else {
            panic!("expected execute");
        };
        let mut outputs: Vec<_> = rows.iter().map(|row| result(row, count)).collect();
        match fault {
            Fault::None => {}
            Fault::Identity => outputs.last_mut().unwrap().lease_id += 1,
            Fault::Resources => outputs[0].validated_exports.push(EndpointRef {
                end: 2,
                slot: 1,
                version: 1,
            }),
            Fault::RowError => {
                outputs[0].error = Some("computation failed".into());
                outputs[0].token_ids.clear();
            }
        }
        let reply = WorkerMessage::ExecuteResult(ExecuteResultMsg {
            rpc_id: sent.rpc_id,
            step_id: sent.step_id,
            worker_epoch: sent.worker_epoch,
            outputs,
            rank_completions: vec![],
        });
        worker
            .socket
            .send(crate::protocol::encode(&reply).unwrap().into())
            .await
            .unwrap();
    };
    let (reply, ()) = tokio::join!(execute, peer);
    reply
}

#[tokio::test]
async fn cumulative_reasoning_survives_preemption_and_finalize_until_finished() {
    let mut worker = worker().await;
    let target = [row(7, 1, SamplingMode::Target)];
    exchange(&mut worker, &target, &[], &[], 1, Fault::None)
        .await
        .unwrap();
    assert_eq!(worker.client.verified_reasoning_tokens.get(&7), Some(&1));
    exchange(&mut worker, &[], &[], &[7], 0, Fault::None)
        .await
        .unwrap();
    assert_eq!(worker.client.verified_reasoning_tokens.get(&7), Some(&1));
    let replay = [row(7, 3, SamplingMode::None)];
    assert!(
        exchange(&mut worker, &replay, &[], &[], 1, Fault::None)
            .await
            .is_ok()
    );
    let mut finalize = row(7, 4, SamplingMode::None);
    finalize.work.purpose = WorkPurpose::Finalize;
    finalize.work.action = WorkAction::ExportOnly { endpoint: 2 };
    finalize.token_ids.clear();
    exchange(&mut worker, &[finalize], &[], &[], 0, Fault::None)
        .await
        .unwrap();
    assert_eq!(worker.client.verified_reasoning_tokens.get(&7), Some(&1));
    exchange(&mut worker, &[], &[7], &[], 0, Fault::None)
        .await
        .unwrap();
    assert!(worker.client.verified_reasoning_tokens.is_empty());
    worker.close().await;
}

#[tokio::test]
async fn rejected_batches_and_row_errors_preserve_verified_cumulative_reasoning() {
    let mut worker = worker().await;
    worker.client.verified_reasoning_tokens.insert(7, 1);
    for (step, fault) in [
        (1, Fault::Identity),
        (2, Fault::Resources),
        (3, Fault::RowError),
    ] {
        let rows = [
            row(7, step, SamplingMode::Target),
            row(8, step, SamplingMode::Target),
        ];
        assert!(
            exchange(&mut worker, &rows, &[], &[], 2, fault)
                .await
                .is_err()
        );
        assert_eq!(worker.client.verified_reasoning_tokens.get(&7), Some(&1));
        assert!(!worker.client.verified_reasoning_tokens.contains_key(&8));
    }
    worker.client.serving_enabled = true;
    let rows = [row(7, 4, SamplingMode::Target)];
    assert!(
        exchange(&mut worker, &rows, &[], &[], 2, Fault::RowError)
            .await
            .is_ok()
    );
    assert_eq!(worker.client.verified_reasoning_tokens.get(&7), Some(&1));
    worker.close().await;
}

#[tokio::test]
async fn missing_rank_ack_does_not_update_verified_cumulative_reasoning() {
    let mut worker = worker().await;
    let owners = Arc::new(Mutex::new(
        crate::models::deepseek_v41::ownership::RequestOwners::default(),
    ));
    owners.lock().unwrap().assign(7);
    worker.client.execution.owners = Some(owners);
    worker.client.verified_reasoning_tokens.insert(7, 1);
    let rows = [row(7, 1, SamplingMode::Target)];
    assert!(
        exchange(&mut worker, &rows, &[], &[], 2, Fault::None)
            .await
            .is_err()
    );
    assert_eq!(worker.client.verified_reasoning_tokens.get(&7), Some(&1));
    worker.close().await;
}

#[tokio::test]
async fn cancellation_clears_verified_reasoning_and_a_new_client_starts_empty() {
    let mut first = worker().await;
    first.client.verified_reasoning_tokens.insert(7, 1);
    let cancel = first.client.cancel_prepare(7);
    let peer = async {
        let message = first.socket.recv().await.unwrap();
        let message: RustMessage = rmp_serde::from_slice(message.get(0).unwrap()).unwrap();
        assert!(matches!(message, RustMessage::Abort(ref abort) if abort.request_id == 7));
    };
    let (result, ()) = tokio::join!(cancel, peer);
    result.unwrap();
    assert!(first.client.verified_reasoning_tokens.is_empty());
    first.close().await;
    let replacement = worker().await;
    assert!(replacement.client.verified_reasoning_tokens.is_empty());
    replacement.close().await;
}
