use crate::models::ServiceCache;
use crate::serving::engine::EngineLoop;
use crate::serving::{Active, Store, Stored, send_events};
use anyhow::{Result, ensure};
use oh_my_vllm_scheduler::execution::WorkPurpose;
use serde_json::{Value, json};
use std::sync::Mutex;
use std::time::Instant;
use tracing::info;

pub(in crate::serving) struct Completion {
    response: Value,
    events: Vec<Value>,
    stored: Option<Stored>,
    reason: String,
}

impl<P: ServiceCache> EngineLoop<'_, P> {
    pub(super) fn process_results(
        &mut self,
        batch: &oh_my_vllm_scheduler::SchedulerOutput<P::State>,
        worker: oh_my_vllm_scheduler::WorkerOutput,
    ) {
        for result in worker.outputs {
            let id = result.request_id;
            let Some(request) = self.active.get_mut(&id) else {
                continue;
            };
            let output = self.client.serving_outputs.remove(&id).unwrap_or_default();
            if request.completion.is_some() || result.work.purpose == WorkPurpose::Finalize {
                continue;
            }
            let (text, mut reason, reasoning_tokens) = output;
            request.record_result(&result, batch.scheduled.iter().find(|s| s.request_id == id));
            if request.output.output_tokens
                >= request.output.request.normalized["max_tokens"]
                    .as_u64()
                    .unwrap() as usize
            {
                reason.get_or_insert("length".to_owned());
            }
            let preparation = (|| -> Result<()> {
                request.lease.grow(
                    text.len()
                        .saturating_mul(8)
                        .saturating_add(result.token_ids.len().saturating_mul(16)),
                )?;
                request.output.reasoning_tokens = reasoning_tokens;
                request.parse_result(&text, &reason)?;
                if let Some(reason) = reason {
                    request.completion =
                        Some(Completion::prepare(request, &reason, &self.context.store)?);
                    self.trace.accepted(
                        id,
                        request.output.input_tokens + request.output.output_tokens,
                    );
                    self.scheduler.finalize_request(
                        id,
                        request.output.input_tokens + request.output.output_tokens,
                    )?;
                    self.scheduler.resume(id);
                } else if request.stream_buffer.is_blocked() {
                    self.scheduler.pause(id);
                }
                Ok(())
            })();
            if let Err(error) = preparation {
                self.scheduler.abort(id);
                self.active.remove(&id).unwrap().fail_delivery(&error);
            }
        }
        self.finish_finalized();
    }

    pub(super) fn finish_finalized(&mut self) {
        let ready: Vec<_> = self
            .active
            .iter()
            .filter(|(id, request)| {
                request.completion.is_some() && self.scheduler.is_finalized(**id)
            })
            .map(|(&id, _)| id)
            .collect();
        for id in ready {
            let request = &self.active[&id];
            if request.events.is_closed() || Instant::now() >= request.lease.deadline() {
                self.scheduler.abort(id);
                self.active
                    .remove(&id)
                    .unwrap()
                    .fail_with_status("request cancelled or timed out", "cancelled");
                continue;
            }
            if let Err(error) = self.scheduler.finish_success(id) {
                self.scheduler.abort(id);
                self.active.remove(&id).unwrap().fail(&error.to_string());
                continue;
            }
            self.trace.complete(id);
            finish_request(self.active.remove(&id).unwrap(), id, &self.context.store);
        }
    }
}

impl Active {
    fn record_result<S>(
        &mut self,
        result: &oh_my_vllm_scheduler::RequestOutput,
        scheduled: Option<&oh_my_vllm_scheduler::ScheduledRequest<S>>,
    ) {
        if let Some(scheduled) = scheduled {
            if self.steps == 0 {
                self.output.cached_tokens =
                    scheduled.num_computed_tokens.min(self.output.input_tokens);
            }
            self.proposed += (scheduled.num_computed_tokens + scheduled.token_ids.len())
                .saturating_sub(self.output.input_tokens + self.output.output_tokens);
        }
        self.steps += 1;
        self.accepted += result.num_accepted_draft_tokens;
        self.output.output_tokens += result.token_ids.len();
        if !result.token_ids.is_empty() && self.first_token.is_none() {
            self.first_token = Some(Instant::now());
        }
    }

    fn parse_result(&mut self, text: &str, reason: &Option<String>) -> Result<()> {
        let parsed = if reason.as_deref() == Some("length") {
            self.parser.feed_length(text)
        } else {
            self.parser.feed(text, reason.is_some())
        };
        let events = parsed.map(|deltas| {
            deltas
                .into_iter()
                .flat_map(|d| self.output.delta(d))
                .collect()
        });
        events.and_then(|events| send_events(self, events))
    }
}

impl Completion {
    fn prepare(request: &mut Active, reason: &str, store: &Mutex<Store>) -> Result<Self> {
        let (response, events) = request.output.finish(reason);
        let stored = if request.output.request.store {
            let history = request.output.history();
            let bytes = serde_json::to_vec(&response)?.len()
                + serde_json::to_vec(&history)?.len()
                + request.output.request.identity.retained_bytes();
            ensure!(
                bytes <= store.lock().unwrap().max_bytes,
                "response exceeds configured storage capacity"
            );
            request.lease.grow(bytes)?;
            Some(Stored {
                response: response.clone(),
                history,
                identity: request.output.request.identity.clone(),
                saved: Instant::now(),
                bytes,
            })
        } else {
            None
        };
        Ok(Self {
            response,
            events,
            stored,
            reason: reason.to_owned(),
        })
    }
}

fn finish_request(mut request: Active, id: u64, store: &Mutex<Store>) {
    let Completion {
        response,
        events,
        stored,
        reason,
    } = request.completion.take().unwrap();
    if let Some(stored) = stored {
        store
            .lock()
            .unwrap()
            .insert(request.output.id.clone(), stored);
    }
    if let Err(error) = send_events(&mut request, events) {
        request.fail_delivery(&error);
        return;
    }
    if !request.output.request.responses
        && request.output.request.stream
        && let Err(error) = send_events(&mut request, vec![json!("[DONE]")])
    {
        request.fail_delivery(&error);
        return;
    }
    let pending = std::mem::take(&mut request.stream_buffer);
    let _ = pending.drain_after_completion(request.events.clone(), Some(request.lease.clone()));
    let elapsed = request.started.elapsed().as_secs_f64();
    info!(request_id = id, output_tokens = request.output.output_tokens,
        input_tokens = request.output.input_tokens, cached_tokens = request.output.cached_tokens,
        reasoning_tokens = request.output.reasoning_tokens, elapsed_s = elapsed,
        steps = request.steps, proposed_draft_tokens = request.proposed,
        accepted_draft_tokens = request.accepted, finish_reason = %reason, "request completed");
    request.metric("completed");
    let _ = request.done.send(Ok(response));
}
