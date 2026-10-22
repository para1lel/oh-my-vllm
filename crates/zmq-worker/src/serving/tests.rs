use super::*;
use crate::client::{WorkerConfig, fixture::TestWorker};
use crate::protocol::{ExecuteMsg, ExecuteResultMsg, RequestResultMsg, RustMessage, WorkerMessage};
use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;
use oh_my_vllm_scheduler::SchedulerConfig;
use std::{fmt::Debug, future::Future, pin::Pin};
use zeromq::{DealerSocket, ZmqMessage, prelude::*};

type Ready = oneshot::Receiver<std::result::Result<(), (StatusCode, String)>>;

#[test]
fn ignore_eos_is_boolean_and_preserves_independent_stop_and_length_options() {
    for responses in [false, true] {
        let mut body = json!({
            "model": "qwen3.8-27b-fp8",
            "max_tokens": 17,
            "min_tokens": 2,
            "stop": ["end"],
        });
        if responses {
            body["input"] = json!("hello");
        } else {
            body["messages"] = json!([{"role": "user", "content": "hello"}]);
        }
        for option in [None, Some(json!(false)), Some(json!(true))] {
            let mut input = body.clone();
            if let Some(value) = &option {
                input["ignore_eos"] = value.clone();
            }
            let request = request::normalize(input, responses, "qwen3.8-27b-fp8", None).unwrap();
            assert_eq!(
                request.normalized["ignore_eos"],
                option.unwrap_or(json!(false))
            );
            assert_eq!(request.normalized["max_tokens"], 17);
            assert_eq!(request.normalized["min_tokens"], 2);
            assert_eq!(request.normalized["stop"], json!(["end"]));
        }
        for invalid in [Value::Null, json!(0), json!(1), json!("true"), json!([])] {
            body["ignore_eos"] = invalid;
            assert!(request::normalize(body.clone(), responses, "qwen3.8-27b-fp8", None).is_err());
        }
    }
}

struct Response {
    ready: Option<Ready>,
    _events: mpsc::Receiver<Value>,
    _done: oneshot::Receiver<std::result::Result<Value, String>>,
}

fn configuration() -> WorkerConfig {
    WorkerConfig {
        model_path: "stub".into(),
        num_gpu_blocks: Some(128),
        mamba_blocks: Some(64),
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
        ..WorkerConfig::default()
    }
}

fn scheduler() -> Scheduler {
    Scheduler::new(
        SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 16,
            ..SchedulerConfig::default()
        },
        HybridCoordinator::new(128, 784, true, 0).with_mamba_capacity(64),
    )
}

fn context() -> EngineContext {
    EngineContext {
        store: Arc::new(Mutex::new(Store {
            entries: BTreeMap::new(),
            ttl: Duration::from_secs(60),
            capacity: 16,
            max_bytes: 1 << 20,
            bytes: 0,
        })),
        health: Arc::new(Mutex::new(json!({"prefix_hit_tokens": 0}))),
        diagnostics_interval: Duration::from_secs(60),
        restart_attempts: 1,
        step_timeout: Duration::from_secs(10),
    }
}

fn submission(budget: &Budget) -> (Submission, Response) {
    let lease = budget.admit(Duration::from_secs(60), 1).unwrap();
    let request = request::normalize(
        json!({
            "model": "qwen3.8-27b-fp8",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "reasoning_effort": "off",
        }),
        false,
        "qwen3.8-27b-fp8",
        None,
    )
    .unwrap();
    let (events, received_events) = mpsc::channel(256);
    let (done, result) = oneshot::channel();
    let (ready, prepared) = oneshot::channel();
    (
        Submission {
            request,
            enqueued: lease.received(),
            events,
            done,
            ready,
            lease,
        },
        Response {
            ready: Some(prepared),
            _events: received_events,
            _done: result,
        },
    )
}

async fn receive(socket: &mut DealerSocket) -> RustMessage {
    let frames = socket.recv().await.unwrap().into_vec();
    assert_eq!(frames.len(), 1);
    rmp_serde::from_slice(&frames[0]).unwrap()
}

async fn next_message<F>(running: &mut Pin<Box<F>>, socket: &mut DealerSocket) -> RustMessage
where
    F: Future,
    F::Output: Debug,
{
    tokio::select! {
        message = receive(socket) => message,
        result = running.as_mut() => panic!("operation stopped before worker message: {result:?}"),
    }
}

async fn reply(socket: &mut DealerSocket, message: WorkerMessage) {
    socket
        .send(ZmqMessage::from(crate::protocol::encode(&message).unwrap()))
        .await
        .unwrap();
}

async fn prepared(socket: &mut DealerSocket, rpc_id: u64, request_id: u64) {
    reply(
        socket,
        WorkerMessage::Prepared {
            rpc_id,
            prompt_token_ids: (0..8)
                .map(|token| request_id as u32 * 100 + token)
                .collect(),
        },
    )
    .await;
}

async fn executed(socket: &mut DealerSocket, message: &ExecuteMsg) {
    let outputs = message
        .scheduled
        .iter()
        .map(|row| {
            let final_chunk = row.num_computed_tokens as usize + row.token_ids.len() >= 8;
            RequestResultMsg {
                request_id: row.request_id,
                error: None,
                token_ids: if final_chunk {
                    vec![42]
                } else {
                    vec![]
                },
                num_accepted_draft_tokens: 0,
                new_draft_token_ids: vec![],
                text: if final_chunk {
                    "x"
                } else {
                    ""
                }
                .to_owned(),
                finish_reason: None,
                reasoning_tokens: 0,
            }
        })
        .collect();
    reply(
        socket,
        WorkerMessage::ExecuteResult(ExecuteResultMsg {
            rpc_id: message.rpc_id,
            outputs,
        }),
    )
    .await;
}

#[tokio::test]
async fn idle_cohort_is_finite_and_active_decode_does_not_wait_for_prepare() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for initial in [2, 6] {
            let mut worker = TestWorker::launch(configuration()).await.unwrap();
            let mut scheduler = scheduler();
            let context = context();
            let budget = Budget::new(1 << 28);
            let (submit, mut incoming) = mpsc::channel(64);
            let mut responses = Vec::new();
            for _ in 0..initial {
                let (request, response) = submission(&budget);
                submit.try_send(request).unwrap();
                responses.push(response);
            }
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
            let mut cohort = Vec::new();
            let first = loop {
                match next_message(&mut running, &mut worker.socket).await {
                    RustMessage::Stats { rpc_id } => {
                        reply(
                            &mut worker.socket,
                            WorkerMessage::StatsResult {
                                rpc_id,
                                stats: json!({}),
                            },
                        )
                        .await;
                    }
                    RustMessage::Prepare {
                        rpc_id, request_id, ..
                    } => {
                        if cohort.len() == 1 {
                            // This Prepare is the barrier after the queue snapshot.
                            for _ in 0..2 {
                                let (request, response) = submission(&budget);
                                submit.try_send(request).unwrap();
                                responses.push(response);
                            }
                        }
                        cohort.push(request_id);
                        prepared(&mut worker.socket, rpc_id, request_id).await;
                    }
                    RustMessage::Execute(message) => break message,
                    other => panic!("unexpected initial message: {other:?}"),
                }
            };
            assert_eq!(cohort.len(), initial.min(4));
            assert_eq!(
                first
                    .scheduled
                    .iter()
                    .map(|row| row.request_id)
                    .collect::<Vec<_>>(),
                cohort,
            );
            assert!(
                first
                    .scheduled
                    .iter()
                    .all(|row| { row.num_computed_tokens == 0 && row.prefill_token_ids.is_some() })
            );
            executed(&mut worker.socket, &first).await;
            let mut withheld = None;
            loop {
                match next_message(&mut running, &mut worker.socket).await {
                    RustMessage::Prepare {
                        rpc_id, request_id, ..
                    } => {
                        assert!(withheld.replace((rpc_id, request_id)).is_none());
                    }
                    RustMessage::Execute(message) => {
                        assert!(withheld.is_some());
                        let decode = message
                            .scheduled
                            .iter()
                            .all(|row| row.num_computed_tokens >= 8 && row.token_ids.len() == 1);
                        executed(&mut worker.socket, &message).await;
                        if decode {
                            assert_eq!(message.scheduled.len(), cohort.len());
                            break;
                        }
                    }
                    other => panic!("unexpected active message: {other:?}"),
                }
            }
            for response in responses.iter_mut().take(cohort.len()) {
                response.ready.take().unwrap().await.unwrap().unwrap();
            }
            drop(running);
            active.clear();
            pending.take();
            drop(incoming);
            drop(submit);
            assert_eq!(budget.snapshot()["request_state_bytes"], 0);
            worker.close().await;
        }
    })
    .await
    .expect("admission protocol deadlocked");
}

#[tokio::test]
async fn cancelled_or_expired_prepare_releases_lease_and_discards_late_reply() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for expired in [false, true] {
            let mut worker = TestWorker::launch(configuration()).await.unwrap();
            let mut scheduler = scheduler();
            let mut active = BTreeMap::new();
            let budget = Budget::new(1 << 28);
            let (request, mut response) = submission(&budget);
            let mut pending = begin_admit(&mut worker.client, 0, request, 1)
                .await
                .unwrap();
            let RustMessage::Prepare {
                rpc_id, request_id, ..
            } = receive(&mut worker.socket).await
            else {
                panic!("expected preparation");
            };
            if expired {
                pending.as_mut().unwrap().deadline = Instant::now();
                wait_admit(
                    &mut worker.client,
                    &mut scheduler,
                    &mut active,
                    &mut pending,
                )
                .await
                .unwrap();
            } else {
                let mut waiting = Box::pin(wait_admit(
                    &mut worker.client,
                    &mut scheduler,
                    &mut active,
                    &mut pending,
                ));
                std::future::poll_fn(|context| {
                    assert!(waiting.as_mut().poll(context).is_pending());
                    std::task::Poll::Ready(())
                })
                .await;
                response.ready.take();
                waiting.await.unwrap();
            }
            let RustMessage::Abort(message) = receive(&mut worker.socket).await else {
                panic!("expected cancellation");
            };
            assert_eq!(message.request_id, request_id);
            if expired {
                let error = response.ready.take().unwrap().await.unwrap().unwrap_err();
                assert_eq!(error.0, StatusCode::SERVICE_UNAVAILABLE);
            }
            assert!(pending.is_none() && active.is_empty());
            assert_eq!(scheduler.num_waiting() + scheduler.num_running(), 0);
            assert_eq!(budget.snapshot()["live_reservations"], 0);
            assert_eq!(budget.snapshot()["request_state_bytes"], 0);
            prepared(&mut worker.socket, rpc_id, request_id).await;
            let (request, mut next_response) = submission(&budget);
            pending = begin_admit(&mut worker.client, 0, request, 1)
                .await
                .unwrap();
            let RustMessage::Prepare {
                rpc_id,
                request_id: next_id,
                ..
            } = receive(&mut worker.socket).await
            else {
                panic!("expected next preparation");
            };
            assert_ne!(request_id, next_id);
            prepared(&mut worker.socket, rpc_id, next_id).await;
            wait_admit(
                &mut worker.client,
                &mut scheduler,
                &mut active,
                &mut pending,
            )
            .await
            .unwrap();
            next_response.ready.take().unwrap().await.unwrap().unwrap();
            assert!(!active.contains_key(&request_id));
            for tail in [false, true] {
                let batch = scheduler.schedule();
                let mut execute = Box::pin(worker.client.execute_one_step(
                    &batch.scheduled,
                    &batch.finished_request_ids,
                    &batch.preempted_request_ids,
                    batch.num_batched_tokens,
                ));
                let RustMessage::Execute(message) =
                    next_message(&mut execute, &mut worker.socket).await
                else {
                    panic!("expected execution");
                };
                assert_eq!(message.scheduled.len(), 1);
                assert_eq!(message.scheduled[0].request_id, next_id);
                executed(&mut worker.socket, &message).await;
                let output = scheduler.update(execute.await.unwrap()).unwrap();
                assert_eq!(
                    output.outputs[0].token_ids,
                    if tail {
                        vec![42]
                    } else {
                        vec![]
                    },
                );
            }
            active.clear();
            assert_eq!(budget.snapshot()["request_state_bytes"], 0);
            worker.close().await;
        }
    })
    .await
    .expect("cancelled preparation blocked a subsequent RPC");
}
