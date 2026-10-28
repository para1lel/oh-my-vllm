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
use tracing::warn;
use zeromq::DealerSocket;
use zeromq::prelude::Socket;

pub(crate) struct ManagedChild {
    child: Option<Child>,
    group_killed: bool,
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
        Self {
            child: Some(child),
            group_killed: false,
        }
    }

    pub(crate) fn child(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("worker child remains owned until drop")
    }

    pub(crate) fn kill_group(&mut self) -> std::io::Result<()> {
        if !self.group_killed {
            kill_worker_group(self.child())?;
            self.group_killed = true;
        }
        Ok(())
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

fn kill_worker_group(child: &mut Child) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let group = i32::try_from(child.id()).map_err(std::io::Error::other)?;
        // SAFETY: the unreaped child owns this process group; kill does not access Rust memory.
        if unsafe { libc::kill(-group, libc::SIGKILL) } == -1 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        Ok(())
    }
    #[cfg(not(unix))]
    child.kill()
}

pub(crate) fn worker_group_alive(pid: u32) -> std::io::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        let group = pid.to_string();
        for entry in std::fs::read_dir("/proc")? {
            let entry = entry?;
            if !entry
                .file_name()
                .to_string_lossy()
                .bytes()
                .all(|b| b.is_ascii_digit())
            {
                continue;
            }
            let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
                continue;
            };
            let Some((_, fields)) = stat.rsplit_once(") ") else {
                continue;
            };
            let mut fields = fields.split_whitespace();
            let state = fields.next();
            let process_group = fields.nth(1);
            if process_group == Some(group.as_str()) && !matches!(state, Some("Z" | "X")) {
                return Ok(true);
            }
        }
        Ok(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        Ok(false)
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
