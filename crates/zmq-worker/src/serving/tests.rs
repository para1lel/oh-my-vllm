mod cancellation;
mod cohort;

use super::*;
use crate::client::{WorkerConfig, fixture::TestWorker};
use crate::models::{ModelFamily, ModelPolicy};
use crate::protocol::{ExecuteMsg, ExecuteResultMsg, RequestResultMsg, RustMessage, WorkerMessage};
use crate::serving::admission::{begin_admit, wait_admit};
use crate::serving::engine::run_engine;
use oh_my_vllm_kv_cache::coordinator::HybridCoordinator;
use oh_my_vllm_scheduler::{Scheduler, SchedulerConfig};
use std::{fmt::Debug, future::Future, pin::Pin};
use zeromq::{DealerSocket, ZmqMessage, prelude::*};

type Ready = oneshot::Receiver<std::result::Result<(), (StatusCode, String)>>;

#[test]
fn qwen_output_preserves_incremental_text_reasoning_and_tools() {
    use crate::models::{ModelFamily, output::Delta, output::Parser};

    let family = ModelFamily::Qwen38;
    let tools = vec![json!({"function":{"name":"echo"}})];
    let mut parser = Parser::new(family, true, tools.clone());
    let arguments = json!({"text":"</tool_call>"});
    let raw = format!(
        "reason</think>answer<tool_call>{}</tool_call>",
        json!({"name":"echo", "arguments":arguments})
    );
    let mut events = Vec::new();
    for character in raw.chars() {
        events.extend(parser.feed(&character.to_string(), false).unwrap());
    }
    events.extend(parser.feed("", true).unwrap());
    let (mut reasoning, mut text, mut calls) = (String::new(), String::new(), Vec::new());
    for event in events {
        match event {
            Delta::Reasoning(value) => reasoning.push_str(&value),
            Delta::Text(value) => text.push_str(&value),
            Delta::Tool {
                index,
                name,
                arguments,
            } => {
                calls.push((
                    index,
                    name,
                    serde_json::from_str::<Value>(&arguments).unwrap(),
                ));
            }
        }
    }
    assert_eq!(reasoning, "reason");
    assert_eq!(text, "answer");
    assert_eq!(calls, vec![(0, "echo".to_owned(), arguments)]);
    let mut parser = Parser::new(family, false, Vec::new());
    assert!(matches!(parser.feed_length("tail").unwrap().as_slice(),
        [Delta::Text(text)] if text == "tail"));
    let incomplete = "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"unfinished";
    let mut parser = Parser::new(family, false, tools.clone());
    assert!(parser.feed(incomplete, true).is_err());
    let mut parser = Parser::new(family, false, tools);
    assert!(parser.feed_length(incomplete).unwrap().is_empty());
}

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
        num_gpu_blocks: Some(128),
        mamba_blocks: Some(64),
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
        ..WorkerConfig::for_model(
            "/synthetic-qwen".into(),
            ModelPolicy::for_family(ModelFamily::Qwen38),
        )
    }
}

fn scheduler() -> Scheduler<HybridCoordinator> {
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
            rank_completions: Vec::new(),
        }),
    )
    .await;
}

#[tokio::test]
async fn admission_respects_configured_sequence_capacity() {
    let mut worker = TestWorker::launch(configuration()).await.unwrap();
    let budget = Budget::new(1 << 28);
    let (request, mut response) = submission(&budget);
    let pending = begin_admit(&mut worker.client, 1, 1, request, 1)
        .await
        .unwrap();

    assert!(pending.is_none());
    let error = response.ready.take().unwrap().await.unwrap().unwrap_err();
    assert_eq!(error.0, StatusCode::SERVICE_UNAVAILABLE);
    worker.close().await;
}
