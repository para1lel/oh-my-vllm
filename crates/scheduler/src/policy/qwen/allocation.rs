use oh_my_vllm_kv_cache::radix::{GDN_PERIOD_TOKENS, KV_PAGE_TOKENS};

use super::*;
use crate::{
    Request,
    execution::{SamplingMode, WorkAction, WorkPurpose},
};

struct Selected {
    active: Active,
    pages: Vec<PageId>,
    checkpoints: Vec<CheckpointId>,
}

impl QwenResidentCache {
    fn select_active(&mut self, request: &Request, hit: QwenHit) -> Option<Selected> {
        let mut selected = Selected {
            active: self.active.get(&request.id).cloned().unwrap_or_default(),
            pages: vec![],
            checkpoints: vec![],
        };
        if self.active.contains_key(&request.id) {
            let start = request.work?.range().0;
            let read = if let Some(&id) = selected.active.endpoints.get(&start) {
                CheckpointRead::Cached { id, end: start }
            } else if start == 0 {
                CheckpointRead::Zero
            } else {
                let hit = self.hit(request);
                if hit.resume.end() != start {
                    return None;
                }
                if let CheckpointRead::Cached { id, .. } = hit.resume
                    && !selected.active.checkpoints.contains(&id)
                {
                    if !self.checkpoints.pin(id.0) {
                        return None;
                    }
                    selected.checkpoints.push(id);
                    selected.active.checkpoints.push(id);
                }
                hit.resume
            };
            selected.active.assignment.resume = read;
            selected.active.assignment.mtp_endpoint = match read {
                CheckpointRead::Cached { id, .. } => {
                    self.mtp_contexts.get(&id).map(|&end| (end, id))
                }
                CheckpointRead::Zero => None,
            };
            return Some(selected);
        }
        selected.active.assignment.evidence = hit.evidence;
        selected.active.assignment.resume = hit.resume;
        selected.active.assignment.mtp_endpoint = hit.mtp_endpoint;
        if let CheckpointRead::Cached { id, .. } = hit.resume {
            if !self.checkpoints.pin(id.0) {
                return None;
            }
            selected.checkpoints.push(id);
            selected.active.checkpoints.push(id);
        }
        for page in hit.pages {
            if !self.pages.pin(page.id.0) {
                self.rollback_pins(&selected.pages, &selected.checkpoints);
                return None;
            }
            selected.pages.push(page.id);
            selected.active.pages.insert(page.page_index, page);
        }
        Some(selected)
    }

    pub(super) fn allocate_work(
        &mut self,
        request: &Request,
        hit: QwenHit,
        lookahead: usize,
    ) -> bool {
        let Some(work) = request.work else {
            return false;
        };
        let (start, end) = work.range();
        let Some(mut selected) = self.select_active(request, hit) else {
            return false;
        };
        let active = &mut selected.active;
        let retired = retire_future_pages(active, request);
        let reclaimable = retired
            .iter()
            .filter(|id| self.pages.pins(id.0) == 1)
            .count();
        let exports = export_ends(request, start, end);
        let needs = page_needs(
            active,
            start,
            reserved_end(request, end, lookahead),
            (work.purpose == WorkPurpose::Finalize).then_some(end),
        );
        let private_count = if active.private.is_empty() {
            6
        } else {
            0
        };
        if needs.len() > self.pages.available() + reclaimable
            || exports.len() > self.checkpoints.available()
            || private_count > self.private.available()
        {
            self.rollback_pins(&selected.pages, &selected.checkpoints);
            return false;
        }
        for id in retired {
            self.pages.unpin(id.0);
        }
        let page_reservation = self.pages.reserve(needs.len()).expect("all pools checked");
        let checkpoint_reservation = self
            .checkpoints
            .reserve(exports.len())
            .expect("all pools checked");
        let private_reservation = self
            .private
            .reserve(private_count)
            .expect("all pools checked");
        for (node, key) in &page_reservation.detached {
            self.tree.detach_page(*node, PageId(*key));
        }
        for (node, key) in &checkpoint_reservation.detached {
            self.tree.detach_checkpoint(*node, CheckpointId(*key));
        }
        active
            .private
            .extend(private_reservation.keys().iter().copied().map(CheckpointId));
        active.assignment.page_copies.clear();
        for (index, key) in needs
            .into_iter()
            .zip(page_reservation.keys().iter().copied())
        {
            self.bind_page(
                active,
                index,
                PageId(key),
                (work.purpose == WorkPurpose::Finalize).then_some(end),
            );
        }
        bind_exports(active, exports, checkpoint_reservation.keys());
        self.bind_endpoint(active, request, start, end);
        self.bind_assignment(active, request, start, end);
        self.active.insert(request.id, selected.active);
        self.clean_tails();
        true
    }

    fn bind_assignment(&self, active: &mut Active, request: &Request, start: usize, end: usize) {
        active
            .assignment
            .private_checkpoints
            .clone_from(&active.private);
        active.assignment.page_table = active.pages.values().cloned().collect();
        active.assignment.kv_spans = kv_spans(active, start, end);
        if request.work.unwrap().purpose == WorkPurpose::Finalize {
            for page in &mut active.assignment.page_table {
                page.valid_end = page
                    .valid_end
                    .min(end)
                    .max(page.page_index * KV_PAGE_TOKENS);
            }
        }
        let mut contiguous = 0;
        for page in active.pages.values() {
            if page.page_index * KV_PAGE_TOKENS != contiguous {
                break;
            }
            contiguous = page.valid_end;
            if contiguous < (page.page_index + 1) * KV_PAGE_TOKENS {
                break;
            }
        }
        active.assignment.evidence.kv_contiguous_end = contiguous;
        let private = active
            .endpoints
            .range(..=request.token_ids.len())
            .next_back()
            .map_or(0, |(&end, _)| end);
        active.assignment.evidence.checkpoint_hit_end =
            self.hit(request).evidence.checkpoint_hit_end.max(private);
    }

    fn rollback_pins(&mut self, pages: &[PageId], checkpoints: &[CheckpointId]) {
        for id in pages {
            self.pages.unpin(id.0);
        }
        for id in checkpoints {
            self.checkpoints.unpin(id.0);
        }
    }

    fn bind_page(&mut self, active: &mut Active, index: usize, id: PageId, trim: Option<usize>) {
        let mut valid_end = index * KV_PAGE_TOKENS;
        if let Some(source) = active.pages.remove(&index) {
            valid_end = trim.map_or(source.valid_end, |end| source.valid_end.min(end));
            let valid_tokens = valid_end
                .saturating_sub(index * KV_PAGE_TOKENS)
                .min(KV_PAGE_TOKENS);
            if valid_tokens > 0 {
                active.assignment.page_copies.push(PageCopy {
                    source: source.id,
                    destination: id,
                    valid_tokens: valid_tokens as u16,
                });
            }
            // The old page stays pinned until the validated copy completes.
            active.copy_sources.push(source.id);
        }
        active.pages.insert(
            index,
            PageBinding {
                page_index: index,
                id,
                valid_end,
                access: PageAccess::UniqueWrite,
            },
        );
    }

    fn bind_endpoint(&self, active: &mut Active, request: &Request, start: usize, end: usize) {
        let work = request.work.unwrap();
        let saved = active
            .endpoints
            .get(&end)
            .copied()
            .or(match active.assignment.resume {
                CheckpointRead::Cached { id, end: at } if at == end => Some(id),
                _ => None,
            });
        active.assignment.endpoint_source = if matches!(work.action, WorkAction::Forward { .. }) {
            EndpointSource::Forward
        } else {
            saved.map_or(EndpointSource::Forward, |id| EndpointSource::SavedHidden {
                id,
                end,
            })
        };
        if let Some(id) = saved {
            active.assignment.mtp_endpoint = self.mtp_contexts.get(&id).map(|&end| (end, id));
        }
        if let Some(id) = active.endpoints.get(&start).copied() {
            active.assignment.resume = CheckpointRead::Cached { id, end: start };
            active.assignment.mtp_endpoint = self.mtp_contexts.get(&id).map(|&end| (end, id));
        }
        if matches!(work.action, WorkAction::ExportOnly { .. }) {
            return;
        }

        let read = match active.assignment.resume {
            CheckpointRead::Cached { id, .. } => Some(id),
            _ => None,
        };
        let candidates = active
            .private
            .iter()
            .copied()
            .filter(|id| Some(*id) != read)
            .collect::<Vec<_>>();
        let first = match work.sampling {
            SamplingMode::Verify { drafts } => end - drafts,
            _ => end,
        };
        for (at, id) in (first..=end).zip(candidates) {
            active.assignment.exports.push(CheckpointExport {
                end: at,
                destination: id,
                role: ExportRole::PrivateEndpoint,
            });
        }
    }
}

fn export_ends(request: &Request, start: usize, end: usize) -> Vec<(usize, ExportRole)> {
    let mut exports = ((start / GDN_PERIOD_TOKENS + 1)..=end / GDN_PERIOD_TOKENS)
        .map(|index| (index * GDN_PERIOD_TOKENS, ExportRole::Periodic))
        .collect::<Vec<_>>();
    if request.work.unwrap().purpose == WorkPurpose::Finalize && request.accepted_end == Some(end) {
        if let Some(last) = exports.last_mut().filter(|(at, _)| *at == end) {
            last.1 = ExportRole::CandidateTail;
        } else {
            exports.push((end, ExportRole::CandidateTail));
        }
    }
    exports
}

fn page_needs(
    active: &Active,
    start: usize,
    reserved_end: usize,
    trim: Option<usize>,
) -> Vec<usize> {
    (0..reserved_end.div_ceil(KV_PAGE_TOKENS))
        .filter(|index| match active.pages.get(index) {
            None => true,
            Some(page) => {
                page.access == PageAccess::ReadOnly
                    && ((page.valid_end < reserved_end.min((index + 1) * KV_PAGE_TOKENS)
                        && (index + 1) * KV_PAGE_TOKENS > start)
                        || trim.is_some_and(|end| {
                            end / KV_PAGE_TOKENS == *index
                                && !end.is_multiple_of(KV_PAGE_TOKENS)
                                && page.valid_end > end
                        }))
            }
        })
        .collect()
}

fn kv_spans(active: &Active, start: usize, end: usize) -> Vec<KvSpan> {
    let mut spans: Vec<KvSpan> = Vec::new();
    let mut position = start;
    while position < end {
        let index = position / KV_PAGE_TOKENS;
        let page = &active.pages[&index];
        let boundary = ((index + 1) * KV_PAGE_TOKENS).min(end);
        let reuse_end = page.valid_end.min(boundary);
        let (next, mode) = if position < reuse_end {
            (reuse_end, KvSpanMode::Reuse)
        } else {
            (boundary, KvSpanMode::Write)
        };
        if let Some(last) = spans.last_mut().filter(|last| last.mode == mode) {
            last.end = next;
        } else {
            spans.push(KvSpan {
                start: position,
                end: next,
                mode,
            });
        }
        position = next;
    }
    spans
}

fn bind_exports(active: &mut Active, exports: Vec<(usize, ExportRole)>, keys: &[SlotKey]) {
    active.assignment.exports.clear();
    for ((at, role), key) in exports.into_iter().zip(keys.iter().copied()) {
        let export = CheckpointExport {
            end: at,
            destination: CheckpointId(key),
            role,
        };
        active.checkpoints.push(export.destination);
        active.assignment.exports.push(export);
    }
}

fn reserved_end(request: &Request, end: usize, lookahead: usize) -> usize {
    let extra = usize::from(
        request.work.unwrap().sampling != SamplingMode::None
            && end < request.prompt_len.saturating_add(request.max_tokens),
    );
    end.saturating_add(lookahead.max(extra))
}

fn retire_future_pages(active: &mut Active, request: &Request) -> Vec<PageId> {
    if request.work.unwrap().purpose != WorkPurpose::Finalize {
        return vec![];
    }
    let end = request
        .accepted_end
        .expect("finalize requires accepted endpoint");
    active
        .pages
        .split_off(&end.div_ceil(KV_PAGE_TOKENS))
        .into_values()
        .map(|page| page.id)
        .collect()
}
