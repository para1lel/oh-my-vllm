mod completion;
mod prefix;

use std::collections::{BTreeMap, VecDeque};

use oh_my_vllm_kv_cache::hash::BlockHash;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::{
    Request, Scheduler,
    policy::{CachePolicy, qwen::QwenTransferPlan},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    Store,
    Load,
    Drop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOperation<T = QwenTransferPlan> {
    pub transfer_id: u64,
    pub snapshot_id: u64,
    pub kind: CacheKind,
    pub request_id: Option<u64>,
    #[serde(flatten)]
    pub state: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    Complete,
    Deferred,
    Capacity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheCompletion {
    pub transfer_id: u64,
    pub status: CacheStatus,
    pub bytes: u64,
}

#[derive(Debug, thiserror::Error)]
#[error("invalid cache completion: {0}")]
pub struct CacheError(&'static str);

#[derive(Clone)]
struct Prefix {
    length: usize,
    hash: BlockHash,
    salt: Option<BlockHash>,
}

struct HostEntry<S> {
    snapshot: u64,
    tables: S,
    prefix: Prefix,
}

pub(crate) struct HostRequest<S> {
    request: Request,
    snapshot: u64,
    tables: S,
}

pub(crate) struct Pending<P: CachePolicy> {
    kind: CacheKind,
    snapshot: u64,
    request: Option<Request>,
    transfer: Option<P::Transfer>,
    source: Option<P::Snapshot>,
    prefix: Option<Prefix>,
    cancelled: bool,
}

pub(crate) struct HostCache<P: CachePolicy> {
    pub capacity: u64,
    next_id: u64,
    pub pending: BTreeMap<u64, Pending<P>>,
    operations: Vec<CacheOperation<P::TransferPlan>>,
    pub requests: VecDeque<HostRequest<P::Snapshot>>,
    prefixes: VecDeque<HostEntry<P::Snapshot>>,
    pub blocked: FxHashSet<u64>,
    capacity_blocked: FxHashSet<u64>,
    retry_at: Option<std::time::Instant>,
    failures: Vec<(u64, &'static str)>,
    pub prefix_hits: BTreeMap<u64, usize>,
    candidates: VecDeque<Request>,
}

impl<P: CachePolicy> Default for HostCache<P> {
    fn default() -> Self {
        Self {
            capacity: 0,
            next_id: 0,
            pending: BTreeMap::new(),
            operations: Vec::new(),
            requests: VecDeque::new(),
            prefixes: VecDeque::new(),
            blocked: FxHashSet::default(),
            capacity_blocked: FxHashSet::default(),
            retry_at: None,
            failures: Vec::new(),
            prefix_hits: BTreeMap::new(),
            candidates: VecDeque::new(),
        }
    }
}

impl<P: CachePolicy> HostCache<P> {
    fn id(&mut self) -> u64 {
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("cache identity exhausted");
        self.next_id
    }

    fn submit(&mut self, job: Pending<P>) {
        let transfer_id = self.id();
        self.operations.push(CacheOperation {
            transfer_id,
            snapshot_id: job.snapshot,
            kind: job.kind,
            request_id: job
                .request
                .as_ref()
                .filter(|_| job.prefix.is_none())
                .map(|r| r.id),
            state: P::transfer_plan(job.transfer.as_ref()),
        });
        self.pending.insert(transfer_id, job);
    }

    fn drop_snapshot(&mut self, snapshot: u64) {
        self.submit(Pending {
            snapshot,
            kind: CacheKind::Drop,
            request: None,
            transfer: None,
            source: None,
            prefix: None,
            cancelled: false,
        });
    }

    pub fn request_transfers(&self) -> usize {
        self.pending
            .values()
            .filter(|p| p.request.is_some() && !p.cancelled)
            .count()
    }

    pub fn has_request_store(&self) -> bool {
        self.pending
            .values()
            .any(|p| p.kind == CacheKind::Store && p.request.is_some())
    }

    pub fn has_prefix_store(&self) -> bool {
        self.pending
            .values()
            .any(|p| p.kind == CacheKind::Store && p.prefix.is_some())
    }

    pub fn has_prefix_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }
}

impl<P: CachePolicy> Scheduler<P> {
    pub fn enable_host_cache(&mut self, capacity: u64) {
        assert!(self.host.pending.is_empty() && self.host.requests.is_empty());
        self.host.capacity = capacity;
    }

    pub fn cache_pending(&self) -> bool {
        !self.host.pending.is_empty()
    }

    pub fn cache_operations(&mut self) -> Vec<CacheOperation<P::TransferPlan>> {
        std::mem::take(&mut self.host.operations)
    }

    pub fn take_cache_failures(&mut self) -> Vec<(u64, &'static str)> {
        std::mem::take(&mut self.host.failures)
    }

    pub(crate) fn retry_cache_stores(&mut self) {
        if self
            .host
            .retry_at
            .is_some_and(|at| std::time::Instant::now() >= at)
        {
            self.host
                .blocked
                .retain(|id| self.host.capacity_blocked.contains(id));
            self.host.retry_at = None;
        }
    }

    pub(crate) fn cache_progress(&mut self) {
        self.host.blocked.clear();
        self.host.capacity_blocked.clear();
        self.host.retry_at = None;
    }

    pub(crate) fn fail_cache_deadlock(&mut self) {
        if self.cache_pending() {
            return;
        }
        let victim = self
            .running
            .iter()
            .find(|r| self.host.capacity_blocked.contains(&r.id))
            .map(|r| r.id);
        if let Some(id) = victim {
            self.abort(id);
            self.host.failures.push((
                id,
                concat!(
                    "host cache capacity cannot preserve a preempted request; ",
                    "increase --host-cache-bytes or reduce concurrent requests",
                ),
            ));
            self.cache_progress();
        }
    }

    pub(crate) fn resident_count(&self) -> usize {
        self.running.len()
            + self
                .host
                .pending
                .values()
                .filter(|p| p.request.is_some())
                .count()
    }

    pub(crate) fn begin_swap(&mut self, request: Request) {
        if self.host.blocked.contains(&request.id) || self.host.has_request_store() {
            self.running.push_back(request);
            return;
        }
        let snapshot = self.host.id();
        let transfer = self.kv.store_request(request.id);
        self.host.submit(Pending {
            kind: CacheKind::Store,
            snapshot,
            request: Some(request),
            transfer: Some(transfer),
            source: None,
            prefix: None,
            cancelled: false,
        });
    }

    pub(crate) fn swap_out(&mut self, id: u64) -> bool {
        if self.host.capacity == 0
            || self.host.has_request_store()
            || self.host.blocked.contains(&id)
        {
            return false;
        }
        let Some(index) = self
            .running
            .iter()
            .position(|r| r.id == id && r.num_in_flight_tokens == 0)
        else {
            return false;
        };
        let request = self.running.remove(index).expect("index exists");
        self.begin_swap(request);
        true
    }

    pub(super) fn reserve_load(
        &mut self,
        request: &Request,
        source: &P::Snapshot,
        prefix: bool,
        budget: usize,
    ) -> Option<P::Transfer> {
        let count = self.aligned_prefill(
            request,
            request.num_computed_tokens,
            request
                .num_tokens_with_spec()
                .saturating_sub(request.num_computed_tokens)
                .min(budget),
        );
        if count == 0 {
            return None;
        }
        self.kv.load_snapshot(
            request,
            source,
            prefix,
            count,
            self.lookahead(request, request.num_computed_tokens, count),
        )
    }

    pub(crate) fn restore_host_requests(&mut self, budget: usize) {
        let count = self.host.requests.len();
        for _ in 0..count {
            let resident = self.resident_count();
            let saved = self.host.requests.pop_front().expect("count checked");
            if resident >= self.config.max_num_seqs || self.paused.contains(&saved.request.id) {
                self.host.requests.push_back(saved);
                continue;
            }
            let Some(transfer) = self.reserve_load(&saved.request, &saved.tables, false, budget)
            else {
                self.host.requests.push_back(saved);
                continue;
            };
            self.host.submit(Pending {
                kind: CacheKind::Load,
                snapshot: saved.snapshot,
                request: Some(saved.request),
                transfer: Some(transfer),
                source: Some(saved.tables),
                prefix: None,
                cancelled: false,
            });
        }
    }

    pub(crate) fn abort_host_request(&mut self, id: u64) -> bool {
        if let Some(job) = self
            .host
            .pending
            .values_mut()
            .find(|p| p.request.as_ref().is_some_and(|r| r.id == id))
        {
            job.cancelled = true;
            return true;
        }
        if let Some(index) = self.host.requests.iter().position(|r| r.request.id == id) {
            let saved = self.host.requests.remove(index).expect("index exists");
            self.host.drop_snapshot(saved.snapshot);
            self.finished_ids.push(id);
            return true;
        }
        false
    }
}
