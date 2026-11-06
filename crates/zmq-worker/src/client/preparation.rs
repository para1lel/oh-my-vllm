use crate::client::{WorkerClient, worker_error};
use crate::error::{Error, Result};
use crate::protocol::{AbortMsg, RegisterMsg, RustMessage};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, warn};

pub(super) mod trace;

impl WorkerClient {
    pub async fn register_request(&mut self, request_id: u64, prompt_token_ids: Vec<u32>) {
        self.send_registration(request_id, prompt_token_ids, None)
            .await;
    }

    pub async fn register_request_with_limit(
        &mut self,
        request_id: u64,
        prompt_token_ids: Vec<u32>,
        max_tokens: usize,
    ) {
        self.send_registration(request_id, prompt_token_ids, Some(max_tokens))
            .await;
    }

    async fn send_registration(
        &mut self,
        request_id: u64,
        prompt_token_ids: Vec<u32>,
        max_tokens: Option<usize>,
    ) {
        debug!(
            request_id,
            input_tokens = prompt_token_ids.len(),
            "register_request"
        );
        let msg = RustMessage::Register(RegisterMsg {
            request_id,
            owner_rank: self.execution.owner(request_id),
            prompt_token_ids,
            max_tokens,
        });
        if let Err(e) = Self::send_raw(&mut self.sock, &msg).await {
            warn!(request_id, "register send failed: {e}");
        }
    }

    pub async fn cancel_scheduled_request(&mut self, request_id: u64) -> Result<()> {
        if self.execution.needs_rank_ack() {
            // The scheduler's finished notification still owns DEP request release.
            Ok(())
        } else {
            self.cancel_prepare(request_id).await
        }
    }

    pub async fn cancel_prepare(&mut self, request_id: u64) -> Result<()> {
        let abort_rpc = if self.execution.needs_rank_ack() {
            self.rpc_id += 1;
            Some(self.rpc_id)
        } else {
            None
        };
        let msg = RustMessage::Abort(AbortMsg {
            request_id,
            rpc_id: abort_rpc,
        });
        timeout(Duration::from_secs(1), Self::send_raw(&mut self.sock, &msg))
            .await
            .map_err(|_| Error::Timeout)??;
        if let Some((id, rpc_id)) = self.last_prepare
            && id == request_id
        {
            if self.prepared_reply.is_none() {
                self.cancelled_prepares.insert(rpc_id);
                if self.cancelled_prepares.len() > 1024 {
                    self.cancelled_prepares.pop_first();
                }
            }
            self.last_prepare = None;
        }
        if self.pending_prepare.is_some_and(|(id, _)| id == request_id) {
            self.pending_prepare = None;
            self.prepared_reply = None;
        }
        if let Some(rpc_id) = abort_rpc {
            let reply = timeout(Duration::from_secs(1), self.recv_for(rpc_id))
                .await
                .map_err(|_| Error::Timeout)??;
            match reply {
                crate::protocol::WorkerMessage::AbortResult {
                    request_id: id,
                    ranks,
                    ..
                } if id == request_id => {
                    self.execution.abort(id, &ranks)?;
                }
                reply => {
                    return Err(Error::UnexpectedMessageType(format!(
                        "wrong abort reply: {reply:?}"
                    )));
                }
            }
        }
        self.verified_reasoning_tokens.remove(&request_id);
        self.preparation_trace.cancel(request_id);
        Ok(())
    }

    pub async fn begin_prepare_request(
        &mut self,
        request_id: u64,
        request: serde_json::Value,
    ) -> Result<()> {
        if self.pending_prepare.is_some() {
            return Err(Error::UnexpectedMessageType(
                "a preparation is already in flight".to_owned(),
            ));
        }
        self.serving_enabled = true;
        self.rpc_id += 1;
        let rpc_id = self.rpc_id;
        self.last_prepare = Some((request_id, rpc_id));
        self.preparation_trace
            .begin(self.config.worker_epoch, rpc_id, request_id, &request);
        Self::send_raw(
            &mut self.sock,
            &RustMessage::Prepare {
                rpc_id,
                request_id,
                owner_rank: self.execution.owner(request_id),
                request,
            },
        )
        .await?;
        self.pending_prepare = Some((request_id, rpc_id));
        Ok(())
    }

    pub fn poll_prepare_request(&mut self) -> Result<Option<Result<Vec<u32>>>> {
        let Some((_, rpc_id)) = self.pending_prepare else {
            return Ok(None);
        };
        let reply = if let Some(reply) = self.prepared_reply.take() {
            reply
        } else {
            match self.inbox.try_recv() {
                Ok(reply) => {
                    let reply = reply?;
                    let received = match &reply {
                        crate::protocol::WorkerMessage::Prepared { rpc_id, .. } => *rpc_id,
                        crate::protocol::WorkerMessage::ExecuteResult(result) => result.rpc_id,
                        crate::protocol::WorkerMessage::Error(error) => error.rpc_id,
                        other => {
                            return Err(Error::UnexpectedMessageType(format!("{other:?}")));
                        }
                    };
                    if received < rpc_id {
                        if matches!(
                            reply,
                            crate::protocol::WorkerMessage::Prepared { .. }
                                | crate::protocol::WorkerMessage::Error(_)
                        ) && self.cancelled_prepares.remove(&received)
                        {
                            warn!(received, rpc_id, "discarding reply to cancelled RPC");
                            return Ok(None);
                        }
                        return Err(Error::UnexpectedMessageType(format!(
                            "unsolicited stale RPC reply {received} while waiting for {rpc_id}"
                        )));
                    }
                    if received != rpc_id {
                        return Err(Error::UnexpectedMessageType(format!(
                            "future RPC reply {received} while waiting for {rpc_id}"
                        )));
                    }
                    reply
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    if self._child.has_exited()? {
                        return Err(Error::WorkerDied);
                    }
                    return Ok(None);
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return Err(Error::WorkerDied),
            }
        };
        self.finish_prepare_reply(reply).map(Some)
    }

    pub async fn wait_prepare_request(&mut self) -> Result<Result<Vec<u32>>> {
        let (_, rpc_id) = self.pending_prepare.ok_or_else(|| {
            Error::UnexpectedMessageType("no preparation is in flight".to_owned())
        })?;
        let reply = self.recv_for(rpc_id).await?;
        self.finish_prepare_reply(reply)
    }

    fn finish_prepare_reply(
        &mut self,
        reply: crate::protocol::WorkerMessage,
    ) -> Result<Result<Vec<u32>>> {
        self.preparation_trace.finish(self.pending_prepare, &reply);
        self.pending_prepare = None;
        self.last_prepare = None;
        let result = match reply {
            crate::protocol::WorkerMessage::Prepared {
                prompt_token_ids, ..
            } => Ok(prompt_token_ids),
            crate::protocol::WorkerMessage::Error(e) => Err(worker_error(e)),
            other => {
                return Err(Error::UnexpectedMessageType(format!(
                    "wrong preparation reply type: {other:?}"
                )));
            }
        };
        Ok(result)
    }
}
