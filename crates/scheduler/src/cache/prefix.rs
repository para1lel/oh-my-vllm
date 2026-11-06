use crate::cache::{CacheKind, Pending, Prefix};
use crate::policy::CachePolicy;
use crate::{Request, Scheduler};

impl<P: CachePolicy> Scheduler<P> {
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
        let hit = self.kv.find_prefix(request);
        let end = hit.length;
        if end == 0 {
            return false;
        }
        let hash = self.kv.prefix_hash(request, end);
        if self.host.prefixes.iter().any(|p| {
            p.prefix.salt == request.cache_salt && p.prefix.length == end && p.prefix.hash == hash
        }) {
            return false;
        }
        let transfer = self.kv.store_prefix(hit);
        let snapshot = self.host.id();
        self.host.submit(Pending {
            kind: CacheKind::Store,
            snapshot,
            request: None,
            transfer: Some(transfer),
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
        let Some(transfer) = self.reserve_load(&request, &source, true, budget) else {
            return Some(false);
        };
        let entry = self.host.prefixes.remove(index).expect("index checked");
        self.host.submit(Pending {
            kind: CacheKind::Load,
            snapshot: entry.snapshot,
            request: Some(request),
            transfer: Some(transfer),
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

    pub(super) fn evict_host_prefix(&mut self) {
        if let Some(entry) = self.host.prefixes.pop_front() {
            self.host.drop_snapshot(entry.snapshot);
            self.cache_progress();
        }
    }
}
