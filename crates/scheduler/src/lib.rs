pub mod cache;
pub mod output;
mod planning;
pub mod policy;
pub mod request;
mod scheduling;
pub mod transaction;
mod update;

use std::collections::VecDeque;

use oh_my_vllm_kv_cache::group::RequestId;

use policy::CachePolicy;

pub use output::{RequestOutput, ScheduledRequest, SchedulerOutput, WorkerOutput};
pub use request::{Request, RequestStatus};

#[derive(Debug, thiserror::Error)]
#[error("invalid worker output for request {request_id}: {reason}")]
pub struct UpdateError {
    pub request_id: RequestId,
    pub reason: &'static str,
}

#[derive(Clone)]
pub struct SchedulerConfig {
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub enable_mtp: bool,
    pub mtp_draft_len: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 32768,
            max_num_seqs: 4,
            enable_mtp: false,
            mtp_draft_len: 4,
        }
    }
}

pub struct Scheduler<P: CachePolicy> {
    config: SchedulerConfig,
    waiting: VecDeque<Request>,
    running: VecDeque<Request>,
    paused: rustc_hash::FxHashSet<RequestId>,
    kv: P,
    finished_ids: Vec<u64>,
    prefill_cursor: Option<RequestId>,
    host: cache::HostCache<P>,
    detached_state: bool,
}

impl<P: CachePolicy> Scheduler<P> {
    pub fn empty_like(&self) -> Self {
        let mut scheduler = Self::new(self.config.clone(), self.kv.empty_like());
        scheduler.enable_host_cache(self.host.capacity);
        scheduler
    }

    pub fn cache_policy(&self) -> &P {
        &self.kv
    }

    pub fn cache_resource_counts(&self) -> P::Resources {
        self.kv.resource_counts()
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        if self.num_running() + self.num_waiting() != 0 || !self.host.pending.is_empty() {
            return false;
        }
        self.clear_host_prefixes();
        self.kv.reset_prefix_cache()
    }

    fn aligned_prefill(&self, request: &Request, start: usize, count: usize) -> usize {
        self.kv
            .align_prefill(request, start, count, self.config.max_num_batched_tokens)
    }

    fn decode_tokens(request: &Request) -> usize {
        let end = request
            .prompt_len
            .max(request.token_ids.len().saturating_sub(1));
        if request.num_computed_tokens < end {
            0
        } else {
            request
                .num_tokens_with_spec()
                .saturating_sub(request.num_computed_tokens)
        }
    }

    pub fn new(config: SchedulerConfig, kv: P) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            paused: rustc_hash::FxHashSet::default(),
            kv,
            finished_ids: Vec::new(),
            prefill_cursor: None,
            host: cache::HostCache::default(),
            detached_state: false,
        }
    }

    fn lookahead(&self, request: &Request, start: usize, count: usize) -> usize {
        if self.config.enable_mtp {
            self.config.mtp_draft_len.min(
                request
                    .prompt_len
                    .saturating_add(request.max_tokens.saturating_sub(1))
                    .saturating_sub(start.saturating_add(count)),
            )
        } else {
            0
        }
    }

    pub fn add_request(&mut self, mut req: Request) -> bool {
        if !self.kv.accepts(&req, &self.config, self.host.capacity > 0) {
            return false;
        }
        self.kv.initialize_hashes(&mut req);
        self.waiting.push_back(req);
        true
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len() + self.host.requests.len() + self.host.request_transfers()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn pause(&mut self, request_id: RequestId) {
        self.paused.insert(request_id);
    }

    pub fn resume(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
    }

    fn preempt_request(&mut self, mut req: Request, output: &mut SchedulerOutput<P::State>) {
        self.kv.free(req.id);
        req.num_computed_tokens = 0;
        req.prefill_start = 0;
        req.num_in_flight_tokens = 0;
        req.draft_token_ids.clear();
        req.status = RequestStatus::Preempted;
        output.preempted_request_ids.push(req.id);
        self.waiting.push_front(req);
    }

    pub fn abort(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
        self.host.prefix_hits.remove(&request_id);
        if self.abort_host_request(request_id) {
            return;
        }
        if self.running.iter().any(|r| r.id == request_id) {
            self.running.retain(|r| r.id != request_id);
            self.kv.free(request_id);
            self.finished_ids.push(request_id);
        } else {
            let before = self.waiting.len();
            self.waiting.retain(|r| r.id != request_id);
            if self.waiting.len() != before {
                self.finished_ids.push(request_id);
            }
        }
    }
}

#[cfg(test)]
mod tests;
