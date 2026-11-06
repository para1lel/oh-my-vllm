use crate::client::lifecycle::{PENDING_REAPS, cleanup_diagnostics, worker_group_alive};
use crate::client::{WORKER_CLEANUP_TIMEOUT, WorkerClient};
use crate::error::{Error, Result};
use crate::protocol::RustMessage;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

impl WorkerClient {
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("sending worker shutdown");
        Self::send_raw(&mut self.sock, &RustMessage::Shutdown).await?;
        let deadline = Instant::now() + WORKER_CLEANUP_TIMEOUT;
        loop {
            if self._child.has_exited()? {
                self._child.kill_group()?;
                let status = self._child.child().try_wait()?.ok_or(Error::WorkerDied)?;
                self.wait_for_worker_group().await?;
                self.binding.close().await;
                self.reader.abort();
                let _ = (&mut self.reader).await;
                return if status.success() {
                    Ok(())
                } else {
                    Err(Error::WorkerDied)
                };
            }
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    pub fn process_id(&mut self) -> u32 {
        self._child.child().id()
    }

    pub async fn terminate(&mut self) -> Result<()> {
        self.reader.abort();
        let pid = self._child.child().id();
        let kill_error = self._child.kill_group().err();
        let deadline = Instant::now() + WORKER_CLEANUP_TIMEOUT;
        let mut status = None;
        let mut wait_error = None;
        loop {
            match self._child.child().try_wait() {
                Ok(Some(exit)) => {
                    status = Some(exit);
                    break;
                }
                Ok(None) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Ok(None) => break,
                Err(error) => {
                    wait_error = Some(error);
                    break;
                }
            }
        }
        let group_error = if status.is_some() {
            self.wait_for_worker_group_until(deadline, pid).await.err()
        } else {
            None
        };
        self.binding.close().await;
        let _ = (&mut self.reader).await;
        if let Some(exit) = &status {
            debug!(worker_pid = pid, ?exit, "worker direct child reaped");
        }

        let mut reasons = Vec::new();
        if let Some(error) = wait_error {
            reasons.push(format!("direct child wait failed: {error}"));
        } else if status.is_none() {
            reasons.push(format!(
                "direct child did not exit within {}s",
                WORKER_CLEANUP_TIMEOUT.as_secs()
            ));
        }
        if let Some(error) = kill_error {
            reasons.push(format!("process-group cleanup failed: {error}"));
        }
        if let Some(error) = group_error {
            reasons.push(format!("process group wait failed: {error}"));
        }
        if !reasons.is_empty() {
            let reason = format!("{}; {}", reasons.join("; "), cleanup_diagnostics(pid));
            warn!(worker_pid = pid, "{reason}");
            return Err(Error::WorkerCleanupUnconfirmed { pid, reason });
        }
        Ok(())
    }

    async fn wait_for_worker_group(&mut self) -> Result<()> {
        let pid = self._child.child().id();
        let deadline = Instant::now() + WORKER_CLEANUP_TIMEOUT;
        self.wait_for_worker_group_until(deadline, pid)
            .await
            .map_err(|reason| {
                let reason = format!("{reason}; {}", cleanup_diagnostics(pid));
                warn!(worker_pid = pid, "{reason}");
                Error::WorkerCleanupUnconfirmed { pid, reason }
            })
    }

    async fn wait_for_worker_group_until(
        &mut self,
        deadline: Instant,
        pid: u32,
    ) -> std::result::Result<(), String> {
        loop {
            let alive = worker_group_alive(pid)
                .map_err(|error| format!("worker process group scan failed: {error}"))?;
            if !alive {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!("worker process group {pid} is still alive"));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    pub async fn wait_for_reapers() -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(30);
        while PENDING_REAPS.load(Ordering::SeqCst) != 0 {
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok(())
    }
}
