use super::*;

#[cfg(target_os = "linux")]
#[tokio::test]
async fn terminate_reports_unconfirmed_cleanup_for_an_unowned_process_group() {
    let config = WorkerConfig {
        num_gpu_blocks: Some(128),
        mamba_blocks: Some(64),
        ..WorkerConfig::for_model(
            "/synthetic-qwen".into(),
            ModelPolicy::for_family(ModelFamily::Qwen38),
        )
    };
    let mut worker = TestWorker::launch(config).await.unwrap();
    let mut leader = independent_group();
    let leader_pid = leader.child().id();
    let leader_group = i32::try_from(leader_pid).unwrap();

    // The replacement child joins the independent leader group, not the test group.
    let mut command = Command::new("/bin/sh");
    command
        .args(["-c", "exec sleep 300"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: setpgid is async-signal-safe; the closure captures only a Copy integer.
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, leader_group) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command.spawn().unwrap();
    let pid = child.id();
    let stat = crate::client::lifecycle::process_stat(pid)
        .unwrap()
        .unwrap();
    assert_eq!(stat.pgrp, leader_group);
    assert_ne!(stat.pgrp, i32::try_from(pid).unwrap());
    worker.client._child = ManagedChild::new(child);

    let Error::WorkerCleanupUnconfirmed {
        pid: failed,
        reason,
    } = worker.client.terminate().await.unwrap_err()
    else {
        panic!("cleanup failure must block worker replacement");
    };
    assert_eq!(failed, pid);
    assert!(reason.contains("not owned"), "{reason}");
    assert!(reason.contains("start_time"), "{reason}");
    assert!(reason.contains("direct_child="), "{reason}");
    assert!(
        crate::client::lifecycle::process_stat(pid)
            .unwrap()
            .is_none()
    );
    assert!(
        crate::client::lifecycle::process_stat(leader_pid)
            .unwrap()
            .is_some()
    );

    leader.kill_group().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while leader.child().try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "leader did not exit");
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(leader);
    worker.close().await;
}

#[cfg(target_os = "linux")]
fn independent_group() -> ManagedChild {
    let mut leader_command = Command::new("/bin/sh");
    leader_command
        .args(["-c", "exec sleep 300"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut leader = crate::client::lifecycle::spawn(&mut leader_command).unwrap();
    let leader_pid = leader.child().id();
    let leader_group = i32::try_from(leader_pid).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(Some(stat)) = crate::client::lifecycle::process_stat(leader_pid)
            && stat.pgrp == leader_group
        {
            break;
        }
        assert!(Instant::now() < deadline, "leader did not own its group");
        std::thread::sleep(Duration::from_millis(10));
    }
    leader
}
