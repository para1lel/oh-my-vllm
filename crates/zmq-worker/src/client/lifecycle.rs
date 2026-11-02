#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;
use tracing::{debug, warn};
use zeromq::DealerSocket;
use zeromq::prelude::Socket;

#[cfg(not(unix))]
compile_error!("oh-my-vllm worker lifecycle requires Unix process groups");

const MAX_REPORTED_PROCESSES: usize = 8;

pub(crate) struct ManagedChild {
    child: Option<Child>,
    group_killed: bool,
    start_time: Option<u64>,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProcessStat {
    pub(crate) state: char,
    pub(crate) ppid: i32,
    pub(crate) pgrp: i32,
    pub(crate) start_time: u64,
}

pub(crate) static PENDING_REAPS: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct IpcPath {
    pub(crate) path: PathBuf,
    pub(crate) identity: std::fs::Metadata,
}

pub(crate) struct Binding {
    pub(crate) socket: Option<DealerSocket>,
    pub(crate) ipc: Option<IpcPath>,
    pub(crate) runtime: tokio::runtime::Handle,
}

impl Binding {
    pub(crate) fn stop(&mut self) -> Option<JoinHandle<()>> {
        let mut socket = self.socket.take()?;
        let ipc = self.ipc.take();
        Some(self.runtime.spawn(async move {
            for error in socket.unbind_all().await {
                warn!(%error, "IPC unbind failed");
            }
            drop(ipc);
        }))
    }

    pub(crate) async fn close(&mut self) {
        if let Some(cleanup) = self.stop() {
            let _ = cleanup.await;
        }
    }
}

impl Drop for Binding {
    fn drop(&mut self) {
        drop(self.stop());
    }
}

impl Drop for IpcPath {
    fn drop(&mut self) {
        if let Ok(current) = std::fs::symlink_metadata(&self.path) {
            #[cfg(unix)]
            if (current.dev(), current.ino()) == (self.identity.dev(), self.identity.ino()) {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }
}

impl ManagedChild {
    pub(crate) fn new(child: Child) -> Self {
        #[cfg(target_os = "linux")]
        let start_time = process_stat(child.id())
            .ok()
            .flatten()
            .map(|stat| stat.start_time);
        #[cfg(not(target_os = "linux"))]
        let start_time = None;
        Self {
            child: Some(child),
            group_killed: false,
            start_time,
        }
    }

    pub(crate) fn child(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("worker child remains owned until drop")
    }

    pub(crate) fn kill_group(&mut self) -> std::io::Result<()> {
        if self.group_killed {
            return Ok(());
        }
        let start_time = self.start_time;
        let result = kill_worker_group(self.child(), start_time);
        if result.is_ok() {
            self.group_killed = true;
        }
        result
    }

    pub(crate) fn has_exited(&mut self) -> std::io::Result<bool> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: waitid writes to initialized storage for the owned, unreaped child.
            unsafe {
                let mut info: libc::siginfo_t = std::mem::zeroed();
                if libc::waitid(
                    libc::P_PID,
                    self.child().id(),
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                ) == -1
                {
                    return Err(std::io::Error::last_os_error());
                }
                let exited = info.si_pid() != 0;
                if exited && (info.si_code != libc::CLD_EXITED || info.si_status() != 0) {
                    warn!(
                        worker_pid = self.child().id(),
                        wait_code = info.si_code,
                        wait_status = info.si_status(),
                        "worker exit observed before process-group cleanup"
                    );
                }
                Ok(exited)
            }
        }
        #[cfg(not(target_os = "linux"))]
        Ok(self.child().try_wait()?.is_some())
    }
}

fn kill_worker_group(child: &mut Child, start_time: Option<u64>) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let pid = child.id();
        let group = i32::try_from(pid).map_err(std::io::Error::other)?;
        let parent = i32::try_from(std::process::id()).map_err(std::io::Error::other)?;
        let mut failure = None;
        match process_stat(pid) {
            Err(error) => failure = Some(error),
            Ok(None) => {
                debug!(
                    worker_pid = pid,
                    process_group = group,
                    "worker direct child is absent; the recorded process group is not signalled"
                );
            }
            Ok(Some(stat))
                if stat.pgrp != group
                    || stat.ppid != parent
                    || start_time.is_some_and(|token| token != stat.start_time) =>
            {
                let expected_start_time =
                    start_time.map_or_else(|| "unknown".to_owned(), |value| value.to_string());
                warn!(
                    worker_pid = pid,
                    process_group = stat.pgrp,
                    observed_parent = stat.ppid,
                    expected_parent = parent,
                    observed_start_time = stat.start_time,
                    ?start_time,
                    state = %stat.state,
                    "worker direct child is not owned by this process"
                );
                failure = Some(std::io::Error::other(format!(
                    "direct child {pid} is not owned by this process \
                     (parent {}, expected {parent}; group {}, expected {group}; \
                      start_time {}, expected {expected_start_time})",
                    stat.ppid, stat.pgrp, stat.start_time
                )));
            }
            Ok(Some(_)) => match kill_process_group(group) {
                Ok(true) => {
                    debug!(
                        worker_pid = pid,
                        process_group = group,
                        "worker process group signalled"
                    );
                }
                Ok(false) => {
                    debug!(
                        worker_pid = pid,
                        process_group = group,
                        "worker process group already absent"
                    );
                    match process_stat(pid) {
                        Ok(Some(current)) if !matches!(current.state, 'Z' | 'X') => {
                            failure = Some(std::io::Error::other(format!(
                                "group {group} absent while direct child {pid} alive in group {}",
                                current.pgrp
                            )));
                        }
                        Err(error) => {
                            failure.get_or_insert(error);
                        }
                        _ => {}
                    }
                }
                Err(error) => failure = Some(error),
            },
        }
        // Signal the direct PID even if the recorded group is absent or unowned.
        if let Err(error) = signal_child(child) {
            failure.get_or_insert(error);
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let _ = start_time;
        let pid = child.id();
        let group = i32::try_from(pid).map_err(std::io::Error::other)?;
        let mut failure = kill_process_group(group).err();
        if let Err(error) = signal_child(child) {
            failure.get_or_insert(error);
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = start_time;
        child.kill()
    }
}

#[cfg(unix)]
fn kill_process_group(group: i32) -> std::io::Result<bool> {
    // SAFETY: kill does not access Rust memory.
    if unsafe { libc::kill(-group, libc::SIGKILL) } == -1 {
        let error = std::io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ESRCH) => Ok(false),
            _ => Err(error),
        };
    }
    Ok(true)
}

#[cfg(unix)]
fn child_already_gone(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::InvalidInput || error.raw_os_error() == Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn child_already_gone(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::InvalidInput
}

fn signal_child(child: &mut Child) -> std::io::Result<()> {
    match child.kill() {
        Ok(()) => Ok(()),
        Err(error) if child_already_gone(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "linux")]
fn parse_process_stat(stat: &str) -> std::io::Result<ProcessStat> {
    let Some((_, fields)) = stat.rsplit_once(") ") else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "process stat has no command delimiter",
        ));
    };
    let mut fields = fields.split_whitespace();
    let state = fields.next().and_then(|value| value.chars().next());
    let ppid = fields.next().and_then(|value| value.parse().ok());
    let pgrp = fields.next().and_then(|value| value.parse().ok());
    let start_time = fields.nth(16).and_then(|value| value.parse().ok());
    match (state, ppid, pgrp, start_time) {
        (Some(state), Some(ppid), Some(pgrp), Some(start_time)) => Ok(ProcessStat {
            state,
            ppid,
            pgrp,
            start_time,
        }),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "process stat has invalid fields",
        )),
    }
}

pub(crate) fn process_stat(pid: u32) -> std::io::Result<Option<ProcessStat>> {
    #[cfg(target_os = "linux")]
    {
        let stat = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        parse_process_stat(&stat).map(Some)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "process identity inspection requires Linux /proc",
        ))
    }
}

pub(crate) fn cleanup_diagnostics(pid: u32) -> String {
    #[cfg(target_os = "linux")]
    {
        let mut report = Vec::new();
        match process_stat(pid) {
            Ok(Some(stat)) => report.push(format!(
                "direct_child={pid} state={} ppid={} pgid={}",
                stat.state, stat.ppid, stat.pgrp
            )),
            Ok(None) => report.push(format!("direct_child={pid} absent")),
            Err(error) => report.push(format!("direct_child={pid} stat_error={error}")),
        }
        report.push(group_members_diagnostic(pid));
        report.push(threads_diagnostic(pid));
        report.join(" ")
    }
    #[cfg(not(target_os = "linux"))]
    {
        format!("direct_child={pid} state=unknown")
    }
}

#[cfg(target_os = "linux")]
fn group_members_diagnostic(pid: u32) -> String {
    let group = i32::try_from(pid).ok();
    let mut members = Vec::new();
    let mut count = 0usize;
    match std::fs::read_dir("/proc") {
        Ok(entries) => {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Ok(member) = name.to_string_lossy().parse::<u32>() else {
                    continue;
                };
                let Ok(Some(stat)) = process_stat(member) else {
                    continue;
                };
                if Some(stat.pgrp) == group && !matches!(stat.state, 'Z' | 'X') {
                    count += 1;
                    if members.len() < MAX_REPORTED_PROCESSES {
                        members.push(format!("{member}:{}", stat.state));
                    }
                }
            }
        }
        Err(error) => return format!("group_scan_error={error}"),
    }
    let suffix = if count > members.len() {
        format!(" +{} more", count - members.len())
    } else {
        String::new()
    };
    format!("group_live={count} [{}]{suffix}", members.join(","))
}

#[cfg(target_os = "linux")]
fn threads_diagnostic(pid: u32) -> String {
    let Ok(entries) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return "threads=absent".to_owned();
    };
    let mut states = std::collections::BTreeMap::new();
    let mut blocked = Vec::new();
    let mut count = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Ok(tid) = name.to_string_lossy().parse::<u32>() else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/stat")) else {
            continue;
        };
        let Some((_, fields)) = stat.rsplit_once(") ") else {
            continue;
        };
        let Some(state) = fields
            .split_whitespace()
            .next()
            .and_then(|value| value.chars().next())
        else {
            continue;
        };
        count += 1;
        *states.entry(state).or_insert(0usize) += 1;
        if !matches!(state, 'R' | 'S' | 'I') && blocked.len() < MAX_REPORTED_PROCESSES {
            let wait_channel = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/wchan"))
                .map(|value| value.trim().to_owned())
                .unwrap_or_default();
            blocked.push(if wait_channel.is_empty() {
                format!("{tid}:{state}")
            } else {
                format!("{tid}:{state}:{wait_channel}")
            });
        }
    }
    let summary = states
        .into_iter()
        .map(|(state, count)| format!("{state}={count}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "threads={count} [{summary}] blocked=[{}]",
        blocked.join(",")
    )
}

pub(crate) fn worker_group_alive(pid: u32) -> std::io::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        let group = i32::try_from(pid).ok();
        for entry in std::fs::read_dir("/proc")? {
            let name = entry?.file_name();
            let Ok(member) = name.to_string_lossy().parse::<u32>() else {
                continue;
            };
            match process_stat(member) {
                Ok(Some(stat)) => {
                    if Some(stat.pgrp) == group && !matches!(stat.state, 'Z' | 'X') {
                        return Ok(true);
                    }
                }
                Ok(None) => {}
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                    return Err(error);
                }
                Err(_) => {}
            }
        }
        Ok(false)
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let group = i32::try_from(pid).map_err(std::io::Error::other)?;
        // SAFETY: signal 0 only checks process-group existence and does not access Rust memory.
        if unsafe { libc::kill(-group, 0) } == 0 {
            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ESRCH) => Ok(false),
            Some(libc::EPERM) => Ok(true),
            _ => Err(error),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "worker process-group inspection requires Unix",
        ))
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.kill_group();
        if let Some(child) = self.child.take() {
            PENDING_REAPS.fetch_add(1, Ordering::SeqCst);
            let child = Arc::new(Mutex::new(child));
            let waiter = Arc::clone(&child);
            if std::thread::Builder::new()
                .name("oh-my-vllm-worker-reap".to_owned())
                .spawn(move || {
                    reap_worker_group(&mut waiter.lock().unwrap());
                    PENDING_REAPS.fetch_sub(1, Ordering::SeqCst);
                })
                .is_err()
            {
                reap_worker_group(&mut child.lock().unwrap());
                PENDING_REAPS.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

pub(crate) fn reap_worker_group(child: &mut Child) {
    let pid = child.id();
    let _ = child.wait();
    while worker_group_alive(pid).unwrap_or(true) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

pub(crate) fn spawn(cmd: &mut Command) -> std::io::Result<ManagedChild> {
    #[cfg(unix)]
    cmd.process_group(0);
    // SAFETY: prctl is async-signal-safe; no Rust objects are touched.
    #[cfg(target_os = "linux")]
    unsafe {
        let parent_pid = libc::getpid();
        cmd.pre_exec(move || {
            if libc::prctl(
                libc::PR_SET_PDEATHSIG,
                libc::SIGKILL as libc::c_ulong,
                0,
                0,
                0,
            ) == -1
            {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent_pid {
                return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
            }
            Ok(())
        });
    }
    cmd.spawn().map(ManagedChild::new)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::os::unix::process::ExitStatusExt;
    use std::process::Stdio;
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
        let directory =
            std::env::temp_dir().join(format!("omv-reaped-group-{}", std::process::id()));
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
}
