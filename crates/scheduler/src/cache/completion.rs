use crate::cache::{
    CacheCompletion, CacheError, CacheKind, CacheStatus, HostEntry, HostRequest, Pending,
};
use crate::policy::CachePolicy;
use crate::{RequestStatus, Scheduler};
use rustc_hash::FxHashSet;

impl<P: CachePolicy> Scheduler<P> {
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

    fn finish_cache(&mut self, mut job: Pending<P>, status: CacheStatus) {
        if job.kind == CacheKind::Drop {
            return;
        }
        let tables = self
            .kv
            .finish_transfer(job.transfer.take().expect("store or load owner"));
        let complete = status == CacheStatus::Complete;
        if job.kind == CacheKind::Store {
            self.finish_store(job, tables, status);
            return;
        }
        let mut request = job.request.take().expect("load request");
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

    fn finish_store(&mut self, mut job: Pending<P>, tables: P::Snapshot, status: CacheStatus) {
        let complete = status == CacheStatus::Complete;
        if let Some(request) = job.request.take() {
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
                    tables,
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
            if complete {
                self.host.prefixes.push_back(HostEntry {
                    snapshot: job.snapshot,
                    tables,
                    prefix: job.prefix.expect("prefix store"),
                });
            } else if status == CacheStatus::Capacity {
                self.evict_host_prefix();
            }
        }
    }
}
