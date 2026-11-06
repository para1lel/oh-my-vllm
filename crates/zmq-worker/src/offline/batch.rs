use anyhow::{Context, Result, ensure};
use oh_my_vllm_scheduler::{Request, Scheduler};
use oh_my_vllm_zmq_worker::client::WorkerClient;
use oh_my_vllm_zmq_worker::models::ServiceCache;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;
use tracing::info;

pub(super) struct BatchResult {
    pub(super) outputs: BTreeMap<u64, Vec<u32>>,
    pub(super) elapsed: f64,
    pub(super) ttft_s: Vec<f64>,
    pub(super) request_phases: Vec<RequestPhases>,
    pub(super) cleanup_s: f64,
    pub(super) steps: usize,
    pub(super) prefix_hit_tokens: usize,
    pub(super) initial_prefix_hit_tokens: usize,
    pub(super) preemptions: usize,
    pub(super) proposed_draft_tokens: usize,
    pub(super) accepted_draft_tokens: usize,
    pub(super) verified_draft_tokens: usize,
}

#[derive(serde::Serialize)]
pub(super) struct RequestPhases {
    request_id: u64,
    submitted_s: f64,
    first_token_s: f64,
    last_token_s: f64,
    first_step: usize,
    last_step: usize,
}

pub(super) async fn execute_batch<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
    next_id: &mut u64,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    arrival_interval: usize,
) -> Result<BatchResult> {
    let mut progress = BatchProgress::new();
    let cleanup_s;
    let mut pending = prompts.iter().peekable();
    loop {
        cache_step(client, scheduler).await?;
        let idle = scheduler.num_running() + scheduler.num_waiting() == 0;
        if arrival_interval == 0 || idle || progress.steps.is_multiple_of(arrival_interval) {
            progress
                .admit(
                    client,
                    scheduler,
                    next_id,
                    &mut pending,
                    max_tokens,
                    arrival_interval,
                )
                .await?;
        }
        let step = scheduler.schedule();
        cache_step(client, scheduler).await?;
        progress.record_schedule(&step);
        if step.scheduled.is_empty() && !scheduler.cache_pending() && client.host_cache_bytes == 0 {
            ensure!(
                scheduler.num_running() + scheduler.num_waiting() == 0,
                "scheduler stalled with unfinished requests"
            );
        }
        let result = client
            .execute_one_step(
                &step.scheduled,
                &step.finished_request_ids,
                &step.preempted_request_ids,
                step.num_batched_tokens,
            )
            .await?;
        if !step.scheduled.is_empty() {
            progress.apply_output(scheduler, result)?;
        }
        if step.scheduled.is_empty() && client.host_cache_bytes > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        if scheduler.num_running() + scheduler.num_waiting() == 0
            && pending.peek().is_none()
            && !scheduler.cache_pending()
        {
            let cleanup = scheduler.schedule();
            let cleanup_started = Instant::now();
            client
                .execute_one_step(&[], &cleanup.finished_request_ids, &[], 0)
                .await?;
            cleanup_s = cleanup_started.elapsed().as_secs_f64();
            break;
        }
    }
    progress.finish(prompts.len(), max_tokens, cleanup_s)
}

struct BatchProgress {
    started: Instant,
    outputs: BTreeMap<u64, Vec<u32>>,
    first_tokens: BTreeMap<u64, f64>,
    submitted: BTreeMap<u64, f64>,
    last_tokens: BTreeMap<u64, f64>,
    first_steps: BTreeMap<u64, usize>,
    last_steps: BTreeMap<u64, usize>,
    steps: usize,
    prefix_hit_tokens: usize,
    initial_prefix_hit_tokens: usize,
    awaiting_first_schedule: BTreeSet<u64>,
    preemptions: usize,
    proposed_draft_tokens: usize,
    accepted_draft_tokens: usize,
    verified_draft_tokens: usize,
    prompt_lengths: BTreeMap<u64, usize>,
}

impl BatchProgress {
    fn new() -> Self {
        let started = Instant::now();
        let outputs = BTreeMap::<u64, Vec<u32>>::new();
        let first_tokens = BTreeMap::<u64, f64>::new();
        let submitted = BTreeMap::<u64, f64>::new();
        let last_tokens = BTreeMap::<u64, f64>::new();
        let first_steps = BTreeMap::new();
        let last_steps = BTreeMap::new();
        let steps: usize = 0;
        let prefix_hit_tokens = 0;
        let initial_prefix_hit_tokens = 0;
        let awaiting_first_schedule = BTreeSet::new();
        let preemptions = 0;
        let proposed_draft_tokens = 0;
        let accepted_draft_tokens = 0;
        let verified_draft_tokens = 0;
        let prompt_lengths = BTreeMap::new();
        Self {
            started,
            outputs,
            first_tokens,
            submitted,
            last_tokens,
            first_steps,
            last_steps,
            steps,
            prefix_hit_tokens,
            initial_prefix_hit_tokens,
            awaiting_first_schedule,
            preemptions,
            proposed_draft_tokens,
            accepted_draft_tokens,
            verified_draft_tokens,
            prompt_lengths,
        }
    }

    async fn admit<P: ServiceCache>(
        &mut self,
        client: &mut WorkerClient,
        scheduler: &mut Scheduler<P>,
        next_id: &mut u64,
        pending: &mut std::iter::Peekable<std::slice::Iter<'_, Vec<u32>>>,
        max_tokens: usize,
        arrival_interval: usize,
    ) -> Result<()> {
        for prompt in pending.by_ref() {
            let id = *next_id;
            *next_id += 1;
            self.submitted.insert(
                id,
                if arrival_interval == 0 {
                    0.0
                } else {
                    self.started.elapsed().as_secs_f64()
                },
            );
            let mut request = Request::new(id, prompt.clone(), max_tokens, vec![]);
            client.bind_request_cache(&mut request);
            if !scheduler.add_request(request) {
                tracing::error!(id, "prompt rejected: exceeds pool capacity");
                continue;
            }
            client
                .register_request_with_limit(id, prompt.clone(), max_tokens)
                .await;
            self.outputs.insert(id, vec![]);
            self.prompt_lengths.insert(id, prompt.len());
            self.awaiting_first_schedule.insert(id);
            if arrival_interval > 0 {
                break;
            }
        }
        Ok(())
    }

    fn record_schedule(&mut self, step: &oh_my_vllm_scheduler::SchedulerOutput) {
        for request in &step.scheduled {
            let known =
                self.prompt_lengths[&request.request_id] + self.outputs[&request.request_id].len();
            self.verified_draft_tokens +=
                (request.num_computed_tokens + request.token_ids.len()).saturating_sub(known);
        }
        self.prefix_hit_tokens += step.cache_hit_tokens;
        if !self.awaiting_first_schedule.is_empty() {
            for request in &step.scheduled {
                if self.awaiting_first_schedule.remove(&request.request_id) {
                    self.initial_prefix_hit_tokens += request.num_computed_tokens;
                }
            }
        }
        self.preemptions += step.preempted_request_ids.len();
    }

    fn apply_output<P: ServiceCache>(
        &mut self,
        scheduler: &mut Scheduler<P>,
        result: oh_my_vllm_scheduler::WorkerOutput,
    ) -> Result<()> {
        self.steps += 1;
        for output in &result.outputs {
            self.proposed_draft_tokens += output.new_draft_token_ids.len();
            self.accepted_draft_tokens += output.num_accepted_draft_tokens;
        }
        let result = scheduler.update(result)?;
        for output in result.outputs {
            if !output.token_ids.is_empty() {
                let received = self.started.elapsed().as_secs_f64();
                self.first_tokens
                    .entry(output.request_id)
                    .or_insert(received);
                self.first_steps
                    .entry(output.request_id)
                    .or_insert(self.steps);
                self.last_tokens.insert(output.request_id, received);
                self.last_steps.insert(output.request_id, self.steps);
            }
            self.outputs
                .get_mut(&output.request_id)
                .context("unexpected request ID")?
                .extend(output.token_ids);
        }
        Ok(())
    }

    fn finish(self, prompt_count: usize, max_tokens: usize, cleanup_s: f64) -> Result<BatchResult> {
        ensure!(
            self.outputs.len() == prompt_count
                && self
                    .outputs
                    .values()
                    .all(|tokens| tokens.len() == max_tokens),
            "incomplete output"
        );
        let elapsed = self.started.elapsed().as_secs_f64();
        info!(
            steps = self.steps,
            elapsed,
            prefix_hit_tokens = self.prefix_hit_tokens,
            initial_prefix_hit_tokens = self.initial_prefix_hit_tokens,
            preemptions = self.preemptions,
            "batch_complete"
        );
        ensure!(
            self.first_tokens.len() == self.outputs.len(),
            "missing first-token timestamp"
        );
        Ok(BatchResult {
            ttft_s: self
                .first_tokens
                .iter()
                .map(|(id, first)| first - self.submitted[id])
                .collect(),
            request_phases: self
                .first_tokens
                .iter()
                .map(|(id, first)| RequestPhases {
                    request_id: *id,
                    submitted_s: self.submitted[id],
                    first_token_s: *first,
                    last_token_s: self.last_tokens[id],
                    first_step: self.first_steps[id],
                    last_step: self.last_steps[id],
                })
                .collect(),
            cleanup_s,
            outputs: self.outputs,
            elapsed,
            steps: self.steps,
            prefix_hit_tokens: self.prefix_hit_tokens,
            initial_prefix_hit_tokens: self.initial_prefix_hit_tokens,
            preemptions: self.preemptions,
            proposed_draft_tokens: self.proposed_draft_tokens,
            accepted_draft_tokens: self.accepted_draft_tokens,
            verified_draft_tokens: self.verified_draft_tokens,
        })
    }
}

async fn cache_step<P: ServiceCache>(
    client: &mut WorkerClient,
    scheduler: &mut Scheduler<P>,
) -> Result<()> {
    client.cache_step(scheduler).await?;
    if let Some((id, message)) = scheduler.take_cache_failures().into_iter().next() {
        anyhow::bail!("request {id}: {message}");
    }
    Ok(())
}
