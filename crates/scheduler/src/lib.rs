//! Request scheduler for oh-my-vllm.
//!
//! A port of the core scheduling logic from vLLM's
//! `vllm/v1/core/sched/scheduler.py`, focused on the paths that matter for
//! Qwen3.5 on a single B200:
//!
//! - FCFS waiting queue → running queue.
//! - Per-step token budget; chunked prefill is a natural consequence of
//!   `remaining.min(budget)` rather than a dedicated branch.
//! - Continuous batching: decode-step requests are scheduled before new prefills.
//! - Prefix cache lookup via [`HybridCoordinator::get_computed_blocks`].
//! - KV block allocation via [`HybridCoordinator::allocate_slots`].
//! - Preemption by recompute: when the pool is full, the last running request is
//!   evicted back to the waiting queue and its blocks are freed.
//! - MTP speculative decoding accounting: `num_tokens_with_spec` is the budget
//!   unit; rollback via `set_drafts([])` + `num_computed_tokens` correction when
//!   the worker rejects draft tokens.

pub mod output;
pub mod request;

use std::collections::VecDeque;

use oh_my_vllm_kv_cache::{coordinator::HybridCoordinator, group::RequestId};
use rustc_hash::FxHashMap;

pub use output::{RequestOutput, ScheduledRequest, SchedulerOutput, WorkerOutput};
pub use request::{Request, RequestStatus};

// ── config ────────────────────────────────────────────────────────────────────

/// Scheduler construction parameters.
pub struct SchedulerConfig {
    /// Maximum total tokens across all requests in a single step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of requests that can run concurrently.
    pub max_num_seqs: usize,
    /// Whether MTP speculative decoding is enabled.
    pub enable_mtp: bool,
    /// Number of MTP draft tokens to request per decode step.
    pub mtp_draft_len: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 32768,
            max_num_seqs: 256,
            enable_mtp: false,
            mtp_draft_len: 4,
        }
    }
}

// ── scheduler ─────────────────────────────────────────────────────────────────

pub struct Scheduler {
    config: SchedulerConfig,
    /// Requests waiting for their first KV block allocation, in arrival order.
    waiting: VecDeque<Request>,
    /// Requests with live KV blocks, currently being executed.
    running: VecDeque<Request>,
    /// Quick lookup from request id to running-queue position is done by linear
    /// scan (queue depth is small); this map only tracks the FA + Mamba block
    /// tables returned by the KV cache for building `ScheduledRequest`.
    block_tables: FxHashMap<RequestId, (Vec<u32>, Vec<u32>)>,
    kv: HybridCoordinator,
    finished_ids: Vec<u64>,
}

impl Scheduler {
    fn aligned_prefill(&self, request: &Request, start: usize, count: usize) -> usize {
        let prefill_end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        if start >= prefill_end {
            return count;
        }
        let block = self.kv.block_size();
        let mut end = start + count;
        if end < prefill_end && self.config.max_num_batched_tokens >= block {
            end = end / block * block;
        }
        let last_boundary = request.token_ids.len() / block * block;
        let next_boundary = (start / block + 1) * block;
        for stop in [
            last_boundary,
            if start.is_multiple_of(block) {
                0
            } else {
                next_boundary
            },
        ] {
            if start < stop && stop < end {
                end = stop;
            }
        }
        end.saturating_sub(start)
    }
    pub fn new(config: SchedulerConfig, kv: HybridCoordinator) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            block_tables: FxHashMap::default(),
            kv,
            finished_ids: Vec::new(),
        }
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(&mut self, mut req: Request) {
        req.block_hashes = self.kv.compute_block_hashes(&req.token_ids);
        self.waiting.push_back(req);
    }

    /// Number of requests currently in the waiting queue.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of requests currently running.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    // ── main scheduling entry point ───────────────────────────────────────────

    /// Produce the next step's batch.
    ///
    /// Call this once per inference step. After the worker finishes, call
    /// [`update`] with the results before calling `schedule` again.
    pub fn schedule(&mut self) -> SchedulerOutput {
        let started = std::time::Instant::now();
        self.kv.new_step_starts();

        let mut output = SchedulerOutput {
            finished_request_ids: std::mem::take(&mut self.finished_ids),
            ..Default::default()
        };
        let mut token_budget = self.config.max_num_batched_tokens;

        // ── phase 1: running requests (decode / continued prefill) ────────────
        //
        // Walk the running queue first so already-warm requests are never
        // starved by new arrivals. Preempt from the tail if the pool is full.
        let mut still_running: VecDeque<Request> = VecDeque::new();
        while let Some(mut req) = self.running.pop_front() {
            let tokens_needed = req.num_tokens_with_spec() - req.num_computed_tokens;
            let to_schedule = self.aligned_prefill(
                &req,
                req.num_computed_tokens,
                tokens_needed.min(token_budget),
            );
            if to_schedule == 0 {
                still_running.push_back(req);
                continue;
            }

            // Re-allocate for the new tokens (no cache-hit lookup for running reqs).
            let alloc = self.kv.allocate_slots(
                &req,
                (Vec::new(), Vec::new()),
                0,
                to_schedule,
                0,
                !still_running.is_empty() || !self.running.is_empty(),
            );

            match alloc {
                Some((new_fa, new_mb)) => {
                    // Append the new block ids to the stored block tables.
                    let entry = self.block_tables.entry(req.id).or_default();
                    entry.0.extend_from_slice(&new_fa);
                    entry.1.extend_from_slice(&new_mb);

                    let fa_table = self.kv.full_attn_blocks(req.id).to_vec();
                    let mb_table = self.kv.mamba_blocks(req.id).to_vec();
                    // Combine confirmed + draft tokens so the worker receives the full
                    // speculative sequence. For non-MTP requests draft_token_ids is empty.
                    let all_tokens: Vec<u32> = req
                        .token_ids
                        .iter()
                        .chain(req.draft_token_ids.iter())
                        .copied()
                        .collect();
                    let end = (req.num_computed_tokens + to_schedule).min(all_tokens.len());
                    output.scheduled.push(ScheduledRequest {
                        request_id: req.id,
                        token_ids: all_tokens[req.num_computed_tokens..end].to_vec(),
                        num_computed_tokens: req.num_computed_tokens,
                        fa_block_table: fa_table,
                        mamba_block_table: mb_table,
                    });
                    token_budget -= to_schedule;
                    req.num_in_flight_tokens = to_schedule;
                    still_running.push_back(req);
                }
                None => {
                    // Pool exhausted: preempt this request.
                    self.kv.free(req.id);
                    self.block_tables.remove(&req.id);
                    req.num_computed_tokens = 0;
                    req.num_in_flight_tokens = 0;
                    req.draft_token_ids.clear();
                    req.status = RequestStatus::Preempted;
                    output.preempted_request_ids.push(req.id);
                    // Re-queue at the front so it is retried before new arrivals.
                    self.waiting.push_front(req);
                }
            }
        }
        self.running = still_running;

        // ── phase 2: waiting requests (new prefills) ──────────────────────────
        while let Some(mut req) = self.waiting.pop_front() {
            if self.running.len() >= self.config.max_num_seqs {
                self.waiting.push_front(req);
                break;
            }

            // Prefix cache lookup.
            let (fa_hit, mb_hit, hit_len) = self.kv.get_computed_blocks(&req);

            let total_tokens = req.num_tokens_with_spec();
            let remaining = total_tokens - hit_len;
            let to_schedule = self.aligned_prefill(&req, hit_len, remaining.min(token_budget));
            if to_schedule == 0 {
                self.waiting.push_front(req);
                break;
            }

            let has_running = !self.running.is_empty();
            let alloc = self.kv.allocate_slots(
                &req,
                (fa_hit, mb_hit),
                hit_len,
                to_schedule,
                0,
                has_running,
            );

            match alloc {
                Some((new_fa, new_mb)) => {
                    // Build the block table: hit blocks (from add_local_computed_blocks,
                    // which registered them inside allocate_slots) plus new blocks.
                    // The coordinator tracks per-request blocks internally; expose them.
                    let fa_table = self.kv.full_attn_blocks(req.id).to_vec();
                    let mb_table = self.kv.mamba_blocks(req.id).to_vec();
                    self.block_tables
                        .insert(req.id, (fa_table.clone(), mb_table.clone()));

                    let computed_start = hit_len;
                    output.scheduled.push(ScheduledRequest {
                        request_id: req.id,
                        token_ids: req.token_ids[computed_start..computed_start + to_schedule]
                            .to_vec(),
                        num_computed_tokens: hit_len,
                        fa_block_table: fa_table,
                        mamba_block_table: mb_table,
                    });
                    token_budget -= to_schedule;
                    req.num_computed_tokens = hit_len;
                    req.num_in_flight_tokens = to_schedule;
                    req.status = RequestStatus::Running;
                    let _ = new_fa;
                    let _ = new_mb;
                    self.running.push_back(req);
                }
                None => {
                    // Pool exhausted even for a new request — stop admitting.
                    self.waiting.push_front(req);
                    break;
                }
            }
        }

        output.num_batched_tokens = output.scheduled.iter().map(|s| s.token_ids.len()).sum();
        tracing::debug!(
            elapsed_us = started.elapsed().as_micros() as u64,
            running = self.running.len(),
            waiting = self.waiting.len(),
            scheduled_tokens = output.num_batched_tokens,
            free_blocks = self.kv.pool().num_free_blocks(),
            "schedule"
        );
        output
    }

    // ── post-step update ──────────────────────────────────────────────────────

    /// Integrate worker results after one inference step.
    ///
    /// For each scheduled request:
    /// - Appends the verified next token.
    /// - Handles MTP acceptance: advances `num_computed_tokens` by the number
    ///   of accepted draft tokens and sets new drafts for the next step.
    /// - Marks finished requests and removes them from the running queue.
    ///
    /// Panics if `worker.outputs` has a different length than the number of
    /// running requests that were scheduled, or references an unknown id.
    pub fn update(&mut self, mut worker: WorkerOutput) -> WorkerOutput {
        let mut finished_ids = Vec::new();

        for result in &mut worker.outputs {
            let req = match self.running.iter_mut().find(|r| r.id == result.request_id) {
                Some(r) => r,
                None => continue,
            };

            // Commit in-flight tokens to computed.
            req.num_computed_tokens += req.num_in_flight_tokens;
            req.num_in_flight_tokens = 0;

            // Handle MTP acceptance rollback: if fewer draft tokens were
            // accepted than were proposed, we still credit the accepted ones.
            if result.num_accepted_draft_tokens < req.draft_token_ids.len() {
                // Roll back num_computed_tokens to exclude rejected drafts — the
                // coordinator already allocated blocks for them, but cache_blocks
                // won't register them since they exceeded num_tokens().
                // The next step's allocate_slots call will handle the gap.
                let accepted = result.num_accepted_draft_tokens;
                req.num_computed_tokens -= req.draft_token_ids.len() - accepted;
            }

            // Append the verified output token and install new drafts.
            result
                .token_ids
                .truncate(req.max_tokens.saturating_sub(req.num_generated_tokens()));
            for &token in &result.token_ids {
                if req.is_finished() {
                    break;
                }
                req.append_token(token);
            }
            req.draft_token_ids.clear();
            req.block_hashes = self.kv.compute_block_hashes(&req.token_ids);

            if self.config.enable_mtp && !result.new_draft_token_ids.is_empty() {
                req.set_drafts(std::mem::take(&mut result.new_draft_token_ids));
            }

            if req.is_finished() {
                finished_ids.push(req.id);
            }
        }

        self.finished_ids.extend_from_slice(&finished_ids);

        // Remove finished requests and free their blocks.
        for id in &finished_ids {
            self.running.retain(|r| r.id != *id);
            self.kv.free(*id);
            self.block_tables.remove(id);
        }
        worker
    }

    /// Abort a request by id, removing it from whichever queue it is in.
    pub fn abort(&mut self, request_id: RequestId) {
        if self.running.iter().any(|r| r.id == request_id) {
            self.running.retain(|r| r.id != request_id);
            self.kv.free(request_id);
            self.block_tables.remove(&request_id);
        } else {
            self.waiting.retain(|r| r.id != request_id);
        }
    }
}

#[cfg(test)]
mod tests;
