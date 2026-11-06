use super::*;

#[tokio::test]
async fn idle_cohort_is_finite_and_active_decode_does_not_wait_for_prepare() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for initial in [2, 6] {
            cohort_case(initial).await;
        }
    })
    .await
    .expect("admission protocol deadlocked");
}

async fn cohort_case(initial: usize) {
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
    let (cohort, first) = first_cohort(
        &mut running,
        &mut worker.socket,
        &submit,
        &mut responses,
        &budget,
    )
    .await;
    assert_eq!(cohort.len(), initial.min(4));
    assert!(!first.scheduled.is_empty());
    assert!(
        first
            .scheduled
            .iter()
            .all(|row| { row.forward_start == 0 && row.prefill_token_ids.is_some() })
    );
    verify_active_decode(&mut running, &mut worker.socket, &cohort, first).await;
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

async fn first_cohort<F>(
    running: &mut Pin<Box<F>>,
    socket: &mut DealerSocket,
    submit: &mpsc::Sender<Submission>,
    responses: &mut Vec<Response>,
    budget: &Budget,
) -> (Vec<u64>, ExecuteMsg)
where
    F: Future,
    F::Output: Debug,
{
    let mut cohort = Vec::new();
    let first = loop {
        match next_message(running, socket).await {
            RustMessage::Stats { rpc_id } => {
                reply(
                    socket,
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
                        let (request, response) = submission(budget);
                        submit.try_send(request).unwrap();
                        responses.push(response);
                    }
                }
                cohort.push(request_id);
                prepared(socket, rpc_id, request_id).await;
            }
            RustMessage::Execute(message) => break message,
            other => panic!("unexpected initial message: {other:?}"),
        }
    };
    (cohort, first)
}

async fn verify_active_decode<F>(
    running: &mut Pin<Box<F>>,
    socket: &mut DealerSocket,
    cohort: &[u64],
    first: ExecuteMsg,
) where
    F: Future,
    F::Output: Debug,
{
    let mut positions: BTreeMap<_, _> = cohort.iter().map(|id| (*id, 0)).collect();
    let mut decoded = Vec::new();
    let mut withheld = None;
    let mut batch = first;
    loop {
        for row in &batch.scheduled {
            let position = positions
                .get_mut(&row.request_id)
                .expect("only the finite cohort can execute before Prepare completes");
            assert_eq!(row.forward_start, *position);
            assert!(!row.token_ids.is_empty());
            *position += row.token_ids.len() as u32;
            if row.forward_start >= 8 {
                assert!(
                    withheld.is_some(),
                    "decode must run with Prepare unanswered"
                );
                assert_eq!(row.token_ids.len(), 1);
                if !decoded.contains(&row.request_id) {
                    decoded.push(row.request_id);
                }
            } else {
                assert!(*position <= 8, "prefill must stop at the prompt endpoint");
            }
        }
        executed(socket, &batch).await;
        if decoded.len() == cohort.len() {
            break;
        }
        batch = loop {
            match next_message(running, socket).await {
                RustMessage::Prepare {
                    rpc_id, request_id, ..
                } => {
                    assert!(!cohort.contains(&request_id));
                    assert!(withheld.replace((rpc_id, request_id)).is_none());
                }
                RustMessage::Execute(message) => {
                    assert!(withheld.is_some());
                    break message;
                }
                other => panic!("unexpected active message: {other:?}"),
            }
        };
    }
}
