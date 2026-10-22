use std::collections::{BTreeMap, VecDeque};

use oh_my_vllm_kv_cache::{hash::BlockHash, snapshot::BlockTables};
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::{Request, RequestStatus, Scheduler};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    Store,
    Load,
    Drop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOperation {
    pub transfer_id: u64,
    pub snapshot_id: u64,
    pub kind: CacheKind,
    pub request_id: Option<u64>,
    pub fa_blocks: Vec<u32>,
    pub mamba_blocks: Vec<u32>,
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

struct HostEntry {
    snapshot: u64,
    tables: BlockTables,
    prefix: Prefix,
}

pub(crate) struct HostRequest {
    request: Request,
    snapshot: u64,
    tables: BlockTables,
}

pub(crate) struct Pending {
    kind: CacheKind,
    snapshot: u64,
    request: Option<Request>,
    tables: BlockTables,
    source: Option<BlockTables>,
    prefix: Option<Prefix>,
    cancelled: bool,
}

#[derive(Default)]
pub(crate) struct HostCache {
    pub capacity: u64,
    next_id: u64,
    pub pending: BTreeMap<u64, Pending>,
    operations: Vec<CacheOperation>,
    pub requests: VecDeque<HostRequest>,
    prefixes: VecDeque<HostEntry>,
    pub blocked: FxHashSet<u64>,
    capacity_blocked: FxHashSet<u64>,
    retry_at: Option<std::time::Instant>,
    failures: Vec<(u64, &'static str)>,
    pub prefix_hits: BTreeMap<u64, usize>,
    candidates: VecDeque<Request>,
}

impl HostCache {
    fn id(&mut self) -> u64 {
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("cache identity exhausted");
        self.next_id
    }

    fn submit(&mut self, job: Pending) {
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
            fa_blocks: job.tables.fa_sources(),
            mamba_blocks: job.tables.mamba_sources(),
        });
        self.pending.insert(transfer_id, job);
    }

    fn drop_snapshot(&mut self, snapshot: u64) {
        self.submit(Pending {
            snapshot,
            kind: CacheKind::Drop,
            request: None,
            tables: BlockTables {
                fa: vec![],
                mamba: vec![],
                mamba_last_state: None,
                mamba_read_source: None,
            },
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

impl Scheduler {
    pub fn enable_host_cache(&mut self, capacity: u64) {
        assert!(self.host.pending.is_empty() && self.host.requests.is_empty());
        self.host.capacity = capacity;
    }

    pub fn cache_pending(&self) -> bool {
        !self.host.pending.is_empty()
    }

    pub fn cache_operations(&mut self) -> Vec<CacheOperation> {
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
        let tables = self.kv.request_tables(request.id);
        self.host.submit(Pending {
            kind: CacheKind::Store,
            snapshot,
            request: Some(request),
            tables,
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

    fn reserve_load(
        &mut self,
        request: &Request,
        source: &BlockTables,
        prefix: bool,
        budget: usize,
    ) -> Option<BlockTables> {
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
        let tables = self.kv.reserve_tables(source)?;
        // A reservation can retire old GDN checkpoints; DMA still owns their destinations.
        self.kv.pin_tables(&tables);
        self.kv.install_tables(
            request.id,
            tables.clone(),
            prefix,
            request.num_computed_tokens,
        );
        if self
            .kv
            .allocate_slots(
                request,
                (vec![], vec![]),
                0,
                count,
                self.lookahead(request, request.num_computed_tokens, count),
                true,
            )
            .is_none()
        {
            self.kv.free(request.id);
            self.kv.release_tables(&tables);
            return None;
        }
        Some(tables)
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
            let Some(tables) = self.reserve_load(&saved.request, &saved.tables, false, budget)
            else {
                self.host.requests.push_back(saved);
                continue;
            };
            self.host.submit(Pending {
                kind: CacheKind::Load,
                snapshot: saved.snapshot,
                request: Some(saved.request),
                tables,
                source: Some(saved.tables),
                prefix: None,
                cancelled: false,
            });
        }
    }

    pub(crate) fn remember_host_prefix(&mut self, request: &Request) {
        if self.host.capacity == 0 {
            return;
        }
        self.host.candidates.push_back(request.clone());
        while self.host.candidates.len() > self.config.max_num_seqs.max(1) {
            self.host.candidates.pop_front();
        }
    }

    pub(crate) fn writeback_host_prefix(&mut self) -> bool {
        if self.host.capacity == 0
            || self
                .host
                .pending
                .values()
                .any(|p| p.kind != CacheKind::Drop)
        {
            return false;
        }
        while let Some(request) = self.host.candidates.pop_front() {
            if self.retain_host_prefix(&request) {
                return true;
            }
        }
        false
    }

    fn retain_host_prefix(&mut self, request: &Request) -> bool {
        if self.host.capacity == 0 {
            return false;
        }
        let (fa, mamba, end) = self.kv.get_computed_blocks(request);
        if end == 0 {
            return false;
        }
        let hash = self.kv.prefix_hash(request, end);
        if self.host.prefixes.iter().any(|p| {
            p.prefix.salt == request.cache_salt && p.prefix.length == end && p.prefix.hash == hash
        }) {
            return false;
        }
        let tables = BlockTables {
            fa,
            mamba,
            mamba_last_state: None,
            mamba_read_source: None,
        };
        self.kv.pin_tables(&tables);
        let snapshot = self.host.id();
        self.host.submit(Pending {
            kind: CacheKind::Store,
            snapshot,
            request: None,
            tables,
            source: None,
            prefix: Some(Prefix {
                length: end,
                hash,
                salt: request.cache_salt,
            }),
            cancelled: false,
        });
        true
    }

    pub(crate) fn begin_prefix_load(
        &mut self,
        request: &Request,
        gpu_hit: usize,
        budget: usize,
    ) -> Option<bool> {
        let best = self
            .host
            .prefixes
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.prefix.salt == request.cache_salt
                    && entry.prefix.length > gpu_hit
                    && entry.prefix.length < request.token_ids.len()
                    && self.kv.prefix_hash(request, entry.prefix.length) == entry.prefix.hash
            })
            .max_by_key(|(_, entry)| entry.prefix.length)
            .map(|(i, _)| i);
        let index = best?;
        let mut request = request.clone();
        request.num_computed_tokens = self.host.prefixes[index].prefix.length;
        request.prefill_start = request.num_computed_tokens;
        let source = self.host.prefixes[index].tables.clone();
        let Some(tables) = self.reserve_load(&request, &source, true, budget) else {
            return Some(false);
        };
        let entry = self.host.prefixes.remove(index).expect("index checked");
        self.host.submit(Pending {
            kind: CacheKind::Load,
            snapshot: entry.snapshot,
            request: Some(request),
            tables,
            source: Some(entry.tables),
            prefix: Some(entry.prefix),
            cancelled: false,
        });
        Some(true)
    }

    pub(crate) fn clear_host_prefixes(&mut self) {
        self.host.candidates.clear();
        while let Some(entry) = self.host.prefixes.pop_front() {
            self.host.drop_snapshot(entry.snapshot);
        }
    }

    fn evict_host_prefix(&mut self) {
        if let Some(entry) = self.host.prefixes.pop_front() {
            self.host.drop_snapshot(entry.snapshot);
            self.cache_progress();
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

    pub fn complete_cache(&mut self, completions: &[CacheCompletion]) -> Result<(), CacheError> {
        let mut seen = FxHashSet::default();
        for result in completions {
            if !seen.insert(result.transfer_id)
                || !self.host.pending.contains_key(&result.transfer_id)
            {
                return Err(CacheError("unknown or duplicate transfer"));
            }
            if self.host.pending[&result.transfer_id].kind == CacheKind::Drop
                && result.status != CacheStatus::Complete
            {
                return Err(CacheError("drop was not completed"));
            }
            if self.host.pending[&result.transfer_id].kind == CacheKind::Load
                && result.status == CacheStatus::Capacity
            {
                return Err(CacheError("load reported a store capacity failure"));
            }
        }
        for result in completions {
            let job = self
                .host
                .pending
                .remove(&result.transfer_id)
                .expect("validated transfer");
            self.finish_cache(job, result.status);
        }
        Ok(())
    }

    fn finish_cache(&mut self, mut job: Pending, status: CacheStatus) {
        if job.kind == CacheKind::Drop {
            return;
        }
        let complete = status == CacheStatus::Complete;
        if job.kind == CacheKind::Store {
            if let Some(request) = job.request {
                if complete || job.cancelled {
                    self.kv.free(request.id);
                }
                if job.cancelled {
                    self.finished_ids.push(request.id);
                    if complete {
                        self.host.drop_snapshot(job.snapshot);
                    }
                } else if complete {
                    self.host.requests.push_back(HostRequest {
                        request,
                        snapshot: job.snapshot,
                        tables: job.tables,
                    });
                } else {
                    self.host.blocked.insert(request.id);
                    if status == CacheStatus::Capacity {
                        self.host.capacity_blocked.insert(request.id);
                    } else {
                        self.host.retry_at =
                            Some(std::time::Instant::now() + std::time::Duration::from_millis(1));
                    }
                    self.running.push_back(request);
                    if status == CacheStatus::Capacity {
                        self.evict_host_prefix();
                    }
                }
            } else {
                self.kv.release_tables(&job.tables);
                if complete {
                    self.host.prefixes.push_back(HostEntry {
                        snapshot: job.snapshot,
                        tables: job.tables,
                        prefix: job.prefix.expect("prefix store"),
                    });
                } else if status == CacheStatus::Capacity {
                    self.evict_host_prefix();
                }
            }
            return;
        }
        let mut request = job.request.take().expect("load request");
        self.kv.release_tables(&job.tables);
        if complete && !job.cancelled {
            if let Some(prefix) = &job.prefix {
                request.num_computed_tokens = prefix.length;
                request.prefill_start = prefix.length;
                self.host.prefix_hits.insert(request.id, prefix.length);
            }
            request.status = RequestStatus::Running;
            self.running.push_back(request);
        } else {
            self.kv.free(request.id);
            if job.cancelled {
                self.finished_ids.push(request.id);
            } else if job.prefix.is_some() {
                request.num_computed_tokens = 0;
                request.prefill_start = 0;
                request.status = RequestStatus::Waiting;
                self.waiting.push_back(request);
            } else {
                self.host.requests.push_back(HostRequest {
                    request,
                    snapshot: job.snapshot,
                    tables: job.source.take().expect("load source"),
                });
            }
        }
        if let Some(prefix) = job.prefix {
            self.host.prefixes.push_back(HostEntry {
                snapshot: job.snapshot,
                tables: job.source.expect("prefix source"),
                prefix,
            });
        } else if complete || job.cancelled {
            self.host.drop_snapshot(job.snapshot);
        }
    }
}
