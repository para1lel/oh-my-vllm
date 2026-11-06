use super::*;

#[tokio::test]
async fn cancelled_or_expired_prepare_releases_lease_and_discards_late_reply() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for expired in [false, true] {
            cancellation_case(expired).await;
        }
    })
    .await
    .expect("cancelled preparation blocked a subsequent RPC");
}

async fn cancellation_case(expired: bool) {
    let mut worker = TestWorker::launch(configuration()).await.unwrap();
    let mut scheduler = scheduler();
    let mut active = BTreeMap::new();
    let budget = Budget::new(1 << 28);
    let (request, mut response) = submission(&budget);
    let mut pending = begin_admit(&mut worker.client, &mut scheduler, 0, 4, request, 1)
        .await
        .unwrap();
    let RustMessage::Prepare {
        rpc_id, request_id, ..
    } = receive(&mut worker.socket).await
    else {
        panic!("expected preparation");
    };
    cancel_waiter(
        &mut worker,
        &mut scheduler,
        &mut active,
        &mut pending,
        &mut response,
        expired,
    )
    .await;
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
    accept_after_late_reply(
        &mut worker,
        &mut scheduler,
        &mut active,
        &mut pending,
        &budget,
        request_id,
    )
    .await;
    active.clear();
    assert_eq!(budget.snapshot()["request_state_bytes"], 0);
    worker.close().await;
}

async fn cancel_waiter(
    worker: &mut TestWorker,
    scheduler: &mut Scheduler<HybridCoordinator>,
    active: &mut BTreeMap<u64, Active>,
    pending: &mut Option<PendingAdmission>,
    response: &mut Response,
    expired: bool,
) {
    if expired {
        pending.as_mut().unwrap().deadline = Instant::now();
        wait_admit(&mut worker.client, scheduler, active, pending)
            .await
            .unwrap();
    } else {
        let mut waiting = Box::pin(wait_admit(&mut worker.client, scheduler, active, pending));
        std::future::poll_fn(|context| {
            assert!(waiting.as_mut().poll(context).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        response.ready.take();
        waiting.await.unwrap();
    }
}

async fn accept_after_late_reply(
    worker: &mut TestWorker,
    scheduler: &mut Scheduler<HybridCoordinator>,
    active: &mut BTreeMap<u64, Active>,
    pending: &mut Option<PendingAdmission>,
    budget: &Budget,
    request_id: u64,
) {
    let (request, mut next_response) = submission(budget);
    *pending = begin_admit(&mut worker.client, scheduler, 0, 4, request, 1)
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
    wait_admit(&mut worker.client, scheduler, active, pending)
        .await
        .unwrap();
    next_response.ready.take().unwrap().await.unwrap().unwrap();
    assert!(!active.contains_key(&request_id));
    execute_admitted(worker, scheduler, next_id).await;
}

async fn execute_admitted(
    worker: &mut TestWorker,
    scheduler: &mut Scheduler<HybridCoordinator>,
    next_id: u64,
) {
    for tail in [false, true] {
        let batch = scheduler.schedule();
        scheduler.mark_submitted(&batch).unwrap();
        let mut execute = Box::pin(worker.client.execute_one_step::<HybridCoordinator>(
            &batch.scheduled,
            &batch.finished_request_ids,
            &batch.preempted_request_ids,
            batch.num_batched_tokens,
        ));
        let RustMessage::Execute(message) = next_message(&mut execute, &mut worker.socket).await
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
}
