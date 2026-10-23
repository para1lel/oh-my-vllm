use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use zeromq::{SocketSend, ZmqMessage};

use super::{GroupClient, ManagedChild, deadline, protocol};

fn remaining(end: Instant) -> Result<Duration> {
    end.checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .context("rank shutdown deadline expired")
}

fn successful_exit(child: &mut ManagedChild) -> Result<Option<bool>> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: the owned child stays unreaped, so its process-group ID cannot be reused.
        unsafe {
            let mut info: libc::siginfo_t = std::mem::zeroed();
            if libc::waitid(
                libc::P_PID,
                child.child().id(),
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            ) == -1
            {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok((info.si_pid() != 0)
                .then(|| info.si_code == libc::CLD_EXITED && info.si_status() == 0))
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = child;
        bail!("graceful rank shutdown requires Linux exit observation")
    }
}

impl GroupClient {
    pub async fn shutdown(&mut self, timeout: Duration) -> Result<()> {
        let end = deadline(timeout)?;
        ensure!(
            !self.closed && self.gate.is_ready() && self.pending.is_empty(),
            "graceful shutdown requires an idle, ready group"
        );
        self.gate.fail_rank(0);
        let result = self.shutdown_exchange(end).await;
        self.close().await.context("rank shutdown cleanup failed")?;
        result
    }

    async fn shutdown_exchange(&mut self, end: Instant) -> Result<()> {
        for rank in 0..self.senders.len() {
            let bytes = rmp_serde::to_vec_named(&protocol::Command::Shutdown {
                header: self.header(rank as u32),
            })?;
            tokio::time::timeout(
                remaining(end)?,
                self.senders[rank].send(ZmqMessage::from(bytes)),
            )
            .await??;
        }
        let mut acknowledged = vec![false; self.senders.len()];
        for _ in 0..self.senders.len() {
            let (rank, reply) = tokio::time::timeout(remaining(end)?, self.inbox.recv())
                .await?
                .context("rank shutdown inbox closed")?;
            let protocol::Reply::Closed { header } = reply? else {
                bail!("unexpected reply during rank shutdown");
            };
            ensure!(header == self.header(rank), "rank shutdown header differs");
            ensure!(
                !acknowledged[rank as usize],
                "duplicate rank shutdown reply"
            );
            acknowledged[rank as usize] = true;
        }
        loop {
            remaining(end)?;
            ensure!(self.inbox.try_recv().is_err(), "extra rank shutdown reply");
            let mut all_exited = true;
            for child in &mut self.children {
                match successful_exit(child)? {
                    Some(true) => {}
                    Some(false) => bail!("rank did not exit successfully after shutdown"),
                    None => all_exited = false,
                }
            }
            if all_exited {
                remaining(end)?;
                ensure!(self.inbox.try_recv().is_err(), "extra rank shutdown reply");
                return Ok(());
            }
            tokio::select! {
                _ = self.inbox.recv() => bail!("extra rank shutdown reply"),
                _ = tokio::time::sleep(Duration::from_millis(10).min(remaining(end)?)) => {}
            }
        }
    }
}
