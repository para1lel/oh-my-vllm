use crate::client::lifecycle::diagnostics::cleanup_diagnostics;
#[cfg(target_os = "linux")]
use crate::client::lifecycle::parse_process_stat;
use crate::client::lifecycle::{ManagedChild, process_stat, spawn, worker_group_alive};
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::process::Stdio;
use std::process::{Child, Command};
use std::time::Duration;
use std::time::Instant;

fn silent(command: &mut Command) {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
}

fn wait_for_owned_group(pid: u32, limit: Duration) {
    let group = i32::try_from(pid).unwrap();
    let deadline = Instant::now() + limit;
    loop {
        if let Ok(Some(stat)) = process_stat(pid)
            && stat.pgrp == group
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "worker child did not own its process group"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn spawn_in_group(script: &str, group: u32) -> Child {
    let group = i32::try_from(group).unwrap();
    let mut command = Command::new("/bin/sh");
    command.args(["-c", script]);
    silent(&mut command);
    // SAFETY: setpgid is async-signal-safe; the closure captures only Copy integers.
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, group) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command.spawn().unwrap()
}

fn wait_for_exit(managed: &mut ManagedChild, limit: Duration) -> std::process::ExitStatus {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(status) = managed.child().try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "worker direct child did not exit"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_group_exit(pid: u32, limit: Duration) {
    let deadline = Instant::now() + limit;
    while worker_group_alive(pid).unwrap() {
        assert!(
            Instant::now() < deadline,
            "worker process group did not exit"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn owned_group_cleanup_reaps_direct_child_and_group() {
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "exec sleep 300"]);
    silent(&mut command);
    let mut managed = spawn(&mut command).unwrap();
    let pid = managed.child().id();
    let stat = process_stat(pid).unwrap().unwrap();
    assert_eq!(stat.pgrp, i32::try_from(pid).unwrap());

    managed.kill_group().unwrap();
    let status = wait_for_exit(&mut managed, Duration::from_secs(5));
    assert_eq!(status.signal(), Some(libc::SIGKILL));
    wait_for_group_exit(pid, Duration::from_secs(5));

    let report = cleanup_diagnostics(pid);
    assert!(
        report.contains(&format!("direct_child={pid} absent")),
        "{report}"
    );
    assert!(report.contains("group_live=0"), "{report}");
}

#[test]
fn unowned_group_is_not_signalled_but_direct_child_is_killed() {
    let mut leader_command = Command::new("/bin/sh");
    leader_command.args(["-c", "exec sleep 300"]);
    silent(&mut leader_command);
    let mut leader = spawn(&mut leader_command).unwrap();
    let leader_pid = leader.child().id();
    wait_for_owned_group(leader_pid, Duration::from_secs(5));

    let child = spawn_in_group("exec sleep 300", leader_pid);
    let pid = child.id();
    let observed = process_stat(pid).unwrap().unwrap();
    assert_eq!(observed.pgrp, i32::try_from(leader_pid).unwrap());
    assert_ne!(observed.pgrp, i32::try_from(pid).unwrap());

    let mut managed = ManagedChild::new(child);
    let error = managed.kill_group().unwrap_err();
    let reason = error.to_string();
    assert!(reason.contains("not owned"), "{reason}");
    assert!(reason.contains("start_time"), "{reason}");
    assert!(!managed.group_killed);
    let status = wait_for_exit(&mut managed, Duration::from_secs(5));
    assert_eq!(status.signal(), Some(libc::SIGKILL));
    // The independently owned leader group must survive the unowned cleanup.
    assert!(process_stat(leader_pid).unwrap().is_some());
    // A retry after the direct child is reaped must not signal the group either.
    managed.kill_group().unwrap();
    assert!(managed.group_killed);
    assert!(process_stat(leader_pid).unwrap().is_some());

    leader.kill_group().unwrap();
    wait_for_exit(&mut leader, Duration::from_secs(5));
    wait_for_group_exit(leader_pid, Duration::from_secs(5));
}

#[test]
fn reaped_child_cleanup_ignores_the_missing_pid_and_group() {
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "exit 0"]);
    silent(&mut command);
    let mut managed = spawn(&mut command).unwrap();
    let pid = managed.child().id();
    assert!(wait_for_exit(&mut managed, Duration::from_secs(5)).success());
    assert!(process_stat(pid).unwrap().is_none());

    managed.kill_group().unwrap();
    assert!(managed.group_killed);
    assert!(!worker_group_alive(pid).unwrap());
}

#[test]
fn reaped_leader_does_not_signal_its_former_group() {
    let directory = std::env::temp_dir().join(format!("omv-reaped-group-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let pid_file = directory.join("member.pid");
    let script = format!(
        "sleep 300 & echo $! > {}; exec sleep 300",
        pid_file.display()
    );
    let mut command = Command::new("/bin/sh");
    command.args(["-c", &script]);
    silent(&mut command);
    let mut managed = spawn(&mut command).unwrap();
    let leader_pid = managed.child().id();
    let group = i32::try_from(leader_pid).unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let member = loop {
        if let Ok(text) = std::fs::read_to_string(&pid_file)
            && let Ok(member) = text.trim().parse::<u32>()
            && let Ok(Some(stat)) = process_stat(member)
            && stat.pgrp == group
        {
            break member;
        }
        assert!(
            Instant::now() < deadline,
            "worker group member did not start"
        );
        std::thread::sleep(Duration::from_millis(10));
    };

    // Kill and reap only the direct leader; the background member stays in its group.
    managed.child().kill().unwrap();
    let status = wait_for_exit(&mut managed, Duration::from_secs(5));
    assert_eq!(status.signal(), Some(libc::SIGKILL));
    assert!(process_stat(leader_pid).unwrap().is_none());
    assert!(process_stat(member).unwrap().is_some());

    // The recorded group is still populated, but its leader is reaped.
    managed.kill_group().unwrap();
    assert!(managed.group_killed);
    assert!(process_stat(member).unwrap().is_some());

    // SAFETY: kill only signals the orphan member started by this test.
    let member_pid = i32::try_from(member).unwrap();
    assert_eq!(unsafe { libc::kill(member_pid, libc::SIGKILL) }, 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match process_stat(member) {
            Ok(None) | Err(_) => break,
            Ok(Some(stat)) if matches!(stat.state, 'Z' | 'X') => break,
            Ok(Some(_)) => {}
        }
        assert!(
            Instant::now() < deadline,
            "worker group member did not exit"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = std::fs::remove_dir_all(&directory);
}

#[cfg(target_os = "linux")]
#[test]
fn malformed_process_stat_is_rejected() {
    assert!(parse_process_stat("garbage").is_err());
    assert!(parse_process_stat("1 (sh) S 2").is_err());
    let first = process_stat(std::process::id()).unwrap().unwrap();
    let second = process_stat(std::process::id()).unwrap().unwrap();
    assert_eq!(first.start_time, second.start_time);
}
