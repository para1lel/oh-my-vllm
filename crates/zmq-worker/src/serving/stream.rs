use crate::serving::resources::Lease;
use crate::serving::{
    ClientDisconnected, MAX_PENDING_STREAM_BYTES, MAX_PENDING_STREAM_EVENTS,
    STREAM_BACKPRESSURE_GRACE, STREAM_LOW_WATER,
};
use anyhow::Result;
use serde_json::Value;
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::warn;

#[derive(Default)]
pub(super) struct StreamBuffer {
    pending: VecDeque<(Value, usize)>,
    pending_bytes: usize,
    blocked_since: Option<Instant>,
}

impl StreamBuffer {
    pub(super) fn is_blocked(&self) -> bool {
        self.blocked_since.is_some()
    }

    pub(super) fn queue(&mut self, event: Value) -> Result<()> {
        let bytes = event.to_string().len();
        anyhow::ensure!(
            self.pending.len() < MAX_PENDING_STREAM_EVENTS
                && self.pending_bytes.saturating_add(bytes) <= MAX_PENDING_STREAM_BYTES,
            "stream pending buffer exceeds its memory limit"
        );
        self.blocked_since.get_or_insert_with(Instant::now);
        self.pending.push_back((event, bytes));
        self.pending_bytes += bytes;
        Ok(())
    }

    pub(super) fn flush(&mut self, sender: &mpsc::Sender<Value>) -> Result<()> {
        while let Some((event, bytes)) = self.pending.pop_front() {
            match sender.try_send(event) {
                Ok(()) => self.pending_bytes -= bytes,
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.pending.push_front((event, bytes));
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(ClientDisconnected.into());
                }
            }
        }
        if self.pending.is_empty() && sender.capacity() >= STREAM_LOW_WATER {
            self.blocked_since = None;
        }
        anyhow::ensure!(
            self.blocked_since
                .is_none_or(|since| since.elapsed() < STREAM_BACKPRESSURE_GRACE),
            "stream backpressure exceeded 30-second grace"
        );
        Ok(())
    }

    pub(super) fn send(&mut self, sender: &mpsc::Sender<Value>, events: Vec<Value>) -> Result<()> {
        self.flush(sender)?;
        for event in events {
            if self.pending.is_empty() {
                match sender.try_send(event) {
                    Ok(()) => continue,
                    Err(mpsc::error::TrySendError::Full(event)) => self.queue(event)?,
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        return Err(ClientDisconnected.into());
                    }
                }
            } else {
                self.queue(event)?;
            }
        }
        self.flush(sender)
    }

    pub(super) fn drain_after_completion(
        self,
        sender: mpsc::Sender<Value>,
        lease: Option<Lease>,
    ) -> Option<tokio::task::JoinHandle<()>> {
        if self.pending.is_empty() {
            return None;
        }
        let deadline = tokio::time::Instant::from_std(
            self.blocked_since
                .expect("pending stream must have a deadline")
                + STREAM_BACKPRESSURE_GRACE,
        );
        Some(tokio::spawn(async move {
            let _drain = lease.map(|lease| lease.drain());
            for (event, _) in self.pending {
                match tokio::time::timeout_at(deadline, sender.send(event)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => return,
                    Err(_) => {
                        warn!("completed stream exceeded backpressure grace");
                        return;
                    }
                }
            }
        }))
    }
}
