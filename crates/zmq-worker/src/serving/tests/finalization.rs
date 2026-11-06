use super::*;
use crate::protocol::{EndpointRef, ModelStateWire, PageValidEnd, Sampling};
use oh_my_vllm_scheduler::policy::qwen::QwenResidentCache;

async fn next_execute<F>(running: &mut Pin<Box<F>>, socket: &mut DealerSocket) -> ExecuteMsg
where
    F: Future,
    F::Output: Debug,
{
    loop {
        match next_message(running, socket).await {
            RustMessage::Stats { rpc_id } => {
                reply(
                    socket,
                    WorkerMessage::StatsResult {
                        rpc_id,
                        stats: json!({}),
                    },
                )
                .await
            }
            RustMessage::Prepare {
                rpc_id, request_id, ..
            } => prepared(socket, rpc_id, request_id).await,
            RustMessage::Execute(message) => return message,
            other => panic!("unexpected message: {other:?}"),
        }
    }
}

async fn completed(socket: &mut DealerSocket, message: &ExecuteMsg, valid: bool) {
    let mut outputs: Vec<RequestResultMsg> = message
        .scheduled
        .iter()
        .map(|row| {
            let ModelStateWire::QwenRadixV1(state) = &row.model_state else {
                panic!("Qwen state required")
            };
            let sampled = row.sampling == Sampling::Target;
            RequestResultMsg {
                request_id: row.request_id,
                lease_id: row.lease_id,
                purpose: row.purpose,
                action: row.action,
                error: None,
                token_ids: if sampled {
                    vec![42]
                } else {
                    vec![]
                },
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![],
                text: if sampled {
                    "x"
                } else {
                    ""
                }
                .into(),
                finish_reason: None,
                reasoning_tokens: 0,
                computed_end: row.forward_end,
                accepted_end: row.accepted_end + u32::from(sampled),
                validated_exports: state
                    .exports
                    .iter()
                    .map(|export| EndpointRef {
                        end: export.end,
                        slot: export.slot,
                        version: export.version,
                    })
                    .collect(),
                committed_page_valid_ends: state
                    .page_table
                    .iter()
                    .map(|page| PageValidEnd {
                        page_index: page.page_index,
                        slot: page.slot,
                        version: page.version,
                        valid_end: row
                            .forward_end
                            .min((page.page_index + 1) * 128)
                            .max(page.page_index * 128),
                    })
                    .collect(),
                mtp_accepted_endpoint: None,
            }
        })
        .collect();
    if !valid {
        outputs[0].lease_id += 1;
    }
    reply(
        socket,
        WorkerMessage::ExecuteResult(ExecuteResultMsg {
            rpc_id: message.rpc_id,
            step_id: message.step_id,
            worker_epoch: message.worker_epoch,
            outputs,
            rank_completions: vec![],
        }),
    )
    .await;
}

#[tokio::test]
async fn terminal_response_and_stored_success_wait_for_exact_finalize() {
    for valid in [false, true] {
        finalization_case(valid).await;
    }
}

async fn finalization_case(valid: bool) {
    let mut worker = TestWorker::launch(configuration()).await.unwrap();
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 16,
            ..Default::default()
        },
        QwenResidentCache::new(64, 4, 256, 16),
    );
    let context = context();
    let budget = Budget::new(1 << 28);
    let (mut request, mut response) = submission(&budget);
    request.request = request::normalize(
        json!({"model":"qwen3.8-27b-fp8", "input":"hello",
        "max_output_tokens":1,"reasoning":{"effort":"off"},"stream":true,
        "metadata":{"session_id":"session"},"prompt_cache_key":"namespace"}),
        true,
        "qwen3.8-27b-fp8",
        None,
    )
    .unwrap();
    let (submit, mut incoming) = mpsc::channel(4);
    submit.try_send(request).unwrap();
    let mut active = BTreeMap::new();
    let mut pending = None;
    let mut running = Box::pin(run_engine(
        &mut worker.client,
        &mut scheduler,
        &mut incoming,
        &context,
        1,
        &mut active,
        &mut pending,
    ));
    let advance = next_execute(&mut running, &mut worker.socket).await;
    assert_eq!(
        advance.scheduled[0].purpose,
        oh_my_vllm_scheduler::execution::WorkPurpose::Advance
    );
    completed(&mut worker.socket, &advance, true).await;
    let finalize = next_execute(&mut running, &mut worker.socket).await;
    assert_eq!(
        finalize.scheduled[0].purpose,
        oh_my_vllm_scheduler::execution::WorkPurpose::Finalize
    );
    assert_eq!(finalize.scheduled[0].accepted_end, 9);
    assert_pending(&mut response, &context);
    completed(&mut worker.socket, &finalize, valid).await;
    if !valid {
        assert!(running.await.is_err());
        assert_pending(&mut response, &context);
        worker.close().await;
        return;
    }
    let result = tokio::select! {
        result = &mut response._done => result.unwrap().unwrap(),
        stopped = &mut running => panic!("engine stopped: {stopped:?}"),
    };
    assert_eq!(result["status"], "incomplete");
    assert_eq!(result["usage"]["output_tokens"], 1);
    assert!(
        context
            .store
            .lock()
            .unwrap()
            .entries
            .contains_key(result["id"].as_str().unwrap())
    );
    drop(running);
    worker.close().await;
}

fn assert_pending(response: &mut Response, context: &EngineContext) {
    assert!(matches!(
        response._done.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert!(context.store.lock().unwrap().entries.is_empty());
    while let Ok(event) = response._events.try_recv() {
        assert_ne!(event["type"], "response.completed");
        assert_ne!(event["type"], "response.incomplete");
    }
}
