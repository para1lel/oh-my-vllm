use crate::models::ServiceCache;
use crate::serving::engine::EngineLoop;
use crate::serving::{Active, Store, send_events};
use anyhow::Result;
use serde_json::json;
use std::sync::Mutex;
use std::time::Instant;
use tracing::info;

impl<P: ServiceCache> EngineLoop<'_, P> {
    pub(super) fn process_results(
        &mut self,
        batch: &oh_my_vllm_scheduler::SchedulerOutput,
        worker: oh_my_vllm_scheduler::WorkerOutput,
    ) {
        for result in worker.outputs {
            let id = result.request_id;
            let Some(request) = self.active.get_mut(&id) else {
                continue;
            };
            request.record_result(&result, batch.scheduled.iter().find(|s| s.request_id == id));
            let (text, mut reason, reasoning_tokens) =
                self.client.serving_outputs.remove(&id).unwrap_or_default();
            if let Err(error) = request.lease.grow(
                text.len()
                    .saturating_mul(8)
                    .saturating_add(result.token_ids.len().saturating_mul(16)),
            ) {
                self.scheduler.abort(id);
                self.active.remove(&id).unwrap().fail(&error.to_string());
                continue;
            }
            request.output.reasoning_tokens = reasoning_tokens;
            if request.output.output_tokens
                >= request.output.request.normalized["max_tokens"]
                    .as_u64()
                    .unwrap() as usize
            {
                reason.get_or_insert("length".to_owned());
            }
            if let Err(error) = request.parse_result(&text, &reason) {
                self.scheduler.abort(id);
                self.active.remove(&id).unwrap().fail_delivery(&error);
                info!(request_id = id, error = %error, "request aborted");
                continue;
            }
            if request.stream_buffer.is_blocked() {
                self.scheduler.pause(id);
            }
            if let Some(reason) = reason {
                self.scheduler.abort(id);
                let request = self.active.remove(&id).unwrap();
                finish_request(request, id, &reason, &self.context.store);
            }
        }
    }
}

impl Active {
    fn record_result(
        &mut self,
        result: &oh_my_vllm_scheduler::RequestOutput,
        scheduled: Option<&oh_my_vllm_scheduler::ScheduledRequest>,
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

fn finish_request(mut request: Active, id: u64, reason: &str, store: &Mutex<Store>) {
    let (response, events) = request.output.finish(reason);
    if request.output.request.store
        && !store.lock().unwrap().insert(
            request.output.id.clone(),
            response.clone(),
            request.output.history(),
        )
    {
        request.fail("response exceeds configured storage capacity");
        return;
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
    let ttft_ms = request
        .first_token
        .map(|t| t.duration_since(request.started).as_millis());
    info!(request_id = id,
        output_tokens = request.output.output_tokens,
        input_tokens = request.output.input_tokens,
        cached_tokens = request.output.cached_tokens,
        reasoning_tokens = request.output.reasoning_tokens,
        thinking_token_budget = request.output.request.normalized
            ["thinking_token_budget"].as_i64(),
        elapsed_s = elapsed,
        ttft_ms,
        output_tps = request.output.output_tokens as f64 / elapsed,
        steps = request.steps,
        proposed_draft_tokens = request.proposed,
        verified_draft_tokens = request.proposed,
        accepted_draft_tokens = request.accepted,
        finish_reason = %reason, "request completed");
    request.metric("completed");
    let _ = request.done.send(Ok(response));
}
