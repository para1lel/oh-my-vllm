use crate::client::lifecycle::ManagedChild;
use crate::client::{MAX_READY_MESSAGE_BYTES, WorkerClient};
use crate::error::{Error, Result};
use crate::protocol::{RustMessage, decode, encode};
use std::time::Duration;
use tracing::{debug, warn};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, ZmqMessage};

impl WorkerClient {
    pub(super) async fn recv_for(
        &mut self,
        expected: u64,
    ) -> Result<crate::protocol::WorkerMessage> {
        if self.pending_prepare.is_some_and(|(_, id)| id == expected)
            && let Some(reply) = self.prepared_reply.take()
        {
            return Ok(reply);
        }
        loop {
            let reply = self.recv_reply().await?;
            let received = reply_rpc_id(&reply)?;
            if received == expected {
                let valid = match &reply {
                    crate::protocol::WorkerMessage::Prepared { .. } => {
                        self.pending_prepare.is_some_and(|(_, id)| id == expected)
                    }
                    crate::protocol::WorkerMessage::ExecuteResult(_) => {
                        self.pending_prepare.is_none_or(|(_, id)| id != expected)
                    }
                    crate::protocol::WorkerMessage::Error(_) => true,
                    crate::protocol::WorkerMessage::StatsResult { .. } => true,
                    crate::protocol::WorkerMessage::CacheResult { .. } => true,
                    crate::protocol::WorkerMessage::AbortResult { .. } => true,
                    _ => false,
                };
                if !valid {
                    return Err(Error::UnexpectedMessageType(format!(
                        "wrong reply type for RPC {expected}: {reply:?}"
                    )));
                }
                return Ok(reply);
            }
            if received > expected {
                return Err(Error::UnexpectedMessageType(format!(
                    "future RPC reply {received} while waiting for {expected}"
                )));
            }
            if self.pending_prepare.is_some_and(|(_, id)| id == received) {
                if self.prepared_reply.is_some() {
                    return Err(Error::UnexpectedMessageType(format!(
                        "duplicate preparation reply for RPC {received}"
                    )));
                }
                if !matches!(
                    reply,
                    crate::protocol::WorkerMessage::Prepared { .. }
                        | crate::protocol::WorkerMessage::Error(_)
                ) {
                    return Err(Error::UnexpectedMessageType(format!(
                        "wrong preparation reply for RPC {received}: {reply:?}"
                    )));
                }
                self.prepared_reply = Some(reply);
                debug!(
                    received,
                    expected, "prepare reply buffered before execute reply"
                );
            } else if matches!(
                reply,
                crate::protocol::WorkerMessage::Prepared { .. }
                    | crate::protocol::WorkerMessage::Error(_)
            ) && self.cancelled_prepares.remove(&received)
            {
                warn!(received, expected, "discarding reply to cancelled RPC");
            } else {
                return Err(Error::UnexpectedMessageType(format!(
                    "unsolicited stale RPC reply {received} while waiting for {expected}"
                )));
            }
        }
    }

    pub(super) async fn send_raw<S: SocketSend + Send>(
        sock: &mut S,
        msg: &RustMessage,
    ) -> Result<()> {
        let bytes = encode(msg)?;
        let zmq_msg = ZmqMessage::from(bytes);
        sock.send(zmq_msg).await?;
        Ok(())
    }

    pub(super) async fn recv_with_child(
        sock: &mut DealerSocket,
        child: &mut ManagedChild,
    ) -> Result<crate::protocol::WorkerMessage> {
        let receive = Self::recv_raw_with_limit(sock, Some(MAX_READY_MESSAGE_BYTES));
        tokio::pin!(receive);
        let mut heartbeat = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                reply = &mut receive => return reply,
                _ = heartbeat.tick() => {
                    if child.has_exited()? { return Err(Error::WorkerDied); }
                }
            }
        }
    }

    async fn recv_reply(&mut self) -> Result<crate::protocol::WorkerMessage> {
        let mut heartbeat = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                reply = self.inbox.recv() => return reply.ok_or(Error::WorkerDied)?,
                _ = heartbeat.tick() => {
                    if self._child.has_exited()? {
                        return Err(Error::WorkerDied);
                    }
                }
            }
        }
    }

    pub(super) async fn recv_raw<S: SocketRecv + Send>(
        sock: &mut S,
    ) -> Result<crate::protocol::WorkerMessage> {
        Self::recv_raw_with_limit(sock, None).await
    }

    async fn recv_raw_with_limit<S: SocketRecv + Send>(
        sock: &mut S,
        byte_limit: Option<usize>,
    ) -> Result<crate::protocol::WorkerMessage> {
        let raw: ZmqMessage = sock.recv().await?;
        if raw.len() != 1 {
            return Err(Error::UnexpectedMessageType(
                "expected one msgpack frame".to_owned(),
            ));
        }
        let bytes = raw.into_vec().into_iter().next().unwrap_or_default();
        if byte_limit.is_some_and(|limit| bytes.len() > limit) {
            return Err(Error::UnexpectedMessageType(
                "oversized worker Ready message".to_owned(),
            ));
        }
        Ok(decode(&bytes)?)
    }
}

fn reply_rpc_id(reply: &crate::protocol::WorkerMessage) -> Result<u64> {
    match reply {
        crate::protocol::WorkerMessage::Prepared { rpc_id, .. } => Ok(*rpc_id),
        crate::protocol::WorkerMessage::ExecuteResult(result) => Ok(result.rpc_id),
        crate::protocol::WorkerMessage::Error(error) => Ok(error.rpc_id),
        crate::protocol::WorkerMessage::StatsResult { rpc_id, .. } => Ok(*rpc_id),
        crate::protocol::WorkerMessage::CacheResult { rpc_id, .. } => Ok(*rpc_id),
        crate::protocol::WorkerMessage::AbortResult { rpc_id, .. } => Ok(*rpc_id),
        other => Err(Error::UnexpectedMessageType(format!("{other:?}"))),
    }
}
