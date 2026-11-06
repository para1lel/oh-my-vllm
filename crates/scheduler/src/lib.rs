pub mod execution;
mod lifecycle;
pub mod output;
mod planning;
pub mod policy;
pub mod request;
mod scheduling;
pub mod session;
pub mod transaction;
mod update;

use std::collections::VecDeque;

use oh_my_vllm_kv_cache::group::RequestId;
use policy::CachePolicy;
use session::{RequestIdentity, RequestTicket, SessionRegistry};

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
    sessions: SessionRegistry,
    worker_epoch: u64,
    step_id: u64,
    next_lease: u64,
    detached_state: bool,
}

impl<P: CachePolicy> Scheduler<P> {
    pub fn empty_like(&self) -> Self {
        let mut replacement = Self::new(self.config.clone(), self.kv.empty_like());
        replacement.sessions = self.sessions.empty_like();
        replacement.worker_epoch = self.worker_epoch;
        replacement
    }

    pub fn cache_policy(&self) -> &P {
        &self.kv
    }

    pub fn cache_resource_counts(&self) -> P::Resources {
        self.kv.resource_counts()
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        self.num_running() + self.num_waiting() == 0 && self.kv.reset_prefix_cache()
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
            kv,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            paused: Default::default(),
            finished_ids: Vec::new(),
            prefill_cursor: None,
            sessions: SessionRegistry::default(),
            worker_epoch: 1,
            step_id: 0,
            next_lease: 0,
            detached_state: false,
        }
    }

    fn lookahead(&self, request: &Request, start: usize, count: usize) -> usize {
        if self.config.enable_mtp && request.status != RequestStatus::Finalizing {
            self.config.mtp_draft_len.min(
                request
                    .prompt_len
                    .saturating_add(request.max_tokens)
                    .saturating_sub(start.saturating_add(count)),
            )
        } else {
            0
        }
    }

    pub fn reserve_admission(
        &mut self,
        identity: RequestIdentity,
        request_id: u64,
    ) -> RequestTicket {
        self.sessions.reserve(identity, request_id)
    }

    pub fn release_admission(&mut self, ticket: &RequestTicket) -> bool {
        self.sessions.release_pending(ticket)
    }

    pub fn add_request(&mut self, mut req: Request) -> bool {
        req.enable_mtp = self.config.enable_mtp;
        if self
            .running
            .iter()
            .chain(&self.waiting)
            .any(|row| row.id == req.id)
        {
            if let Some(ticket) = &req.ticket {
                self.sessions.release_pending(ticket);
            }
            return false;
        }
        if !self.kv.accepts(&req, &self.config) {
            if let Some(ticket) = &req.ticket {
                self.sessions.release_pending(ticket);
            }
            return false;
        }
        let ticket = req.ticket.take().unwrap_or_else(|| {
            let mut identity = RequestIdentity::anonymous(req.id);
            identity.namespace.clone_from(&req.cache_domain.namespace);
            self.reserve_admission(identity, req.id)
        });
        if ticket.request != req.id || !self.sessions.consume(&ticket) {
            return false;
        }
        req.cache_domain
            .namespace
            .clone_from(&ticket.identity.namespace);
        req.cache_domain.salt = req.cache_salt;
        req.ticket = Some(ticket);
        self.kv.initialize_hashes(&mut req);
        self.waiting.push_back(req);
        true
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    fn resident_count(&self) -> usize {
        self.running.len()
    }

    pub fn pause(&mut self, request_id: RequestId) {
        self.paused.insert(request_id);
    }

    pub fn resume(&mut self, request_id: RequestId) {
        self.paused.remove(&request_id);
    }

    fn preempt_request(&mut self, mut req: Request, output: &mut SchedulerOutput<P::State>) {
        assert!(
            req.in_flight.is_none(),
            "cannot reclaim submitted execution"
        );
        self.kv.free(req.id);
        req.num_computed_tokens = 0;
        req.prefill_start = 0;
        req.num_in_flight_tokens = 0;
        req.draft_token_ids.clear();
        req.work = None;
        req.status = RequestStatus::Preempted;
        output.preempted_request_ids.push(req.id);
        self.waiting.push_front(req);
    }
}

#[cfg(test)]
mod tests;
