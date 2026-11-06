use super::*;
use crate::{
    Request,
    execution::{SamplingMode, WorkAction, WorkPurpose, WorkSpec},
    policy::CachePolicy,
};

fn cache() -> QwenResidentCache {
    QwenResidentCache::new(4, 2, 32768, 8192)
}

fn request(end: usize, forward_start: usize, forward_end: usize) -> Request {
    let mut request = Request::new(1, (0..end as u32).collect(), 1, vec![]);
    request.work = Some(WorkSpec {
        purpose: WorkPurpose::Advance,
        action: WorkAction::Forward {
            start: forward_start,
            end: forward_end,
        },
        sampling: SamplingMode::Target,
    });
    request
}

pub(super) fn checkpoint(
    cache: &mut QwenResidentCache,
    request: &Request,
    end: usize,
) -> CheckpointId {
    let reservation = cache.checkpoints.reserve(1).unwrap();
    for (node, key) in &reservation.detached {
        cache.tree.detach_checkpoint(*node, CheckpointId(*key));
    }
    let id = CheckpointId(reservation.keys()[0]);
    let node = cache
        .tree
        .ensure(&request.cache_domain, &request.token_ids[..end]);
    cache.tree.attach_checkpoint(node, id);
    cache.checkpoints.attach(id.0, node);
    cache.checkpoints.unpin(id.0);
    id
}

pub(super) fn page(cache: &mut QwenResidentCache, request: &Request, end: usize) -> PageId {
    let reservation = cache.pages.reserve(1).unwrap();
    for (node, key) in &reservation.detached {
        cache.tree.detach_page(*node, PageId(*key));
    }
    let id = PageId(reservation.keys()[0]);
    let node = cache
        .tree
        .ensure(&request.cache_domain, &request.token_ids[..end]);
    cache.tree.attach_page(node, id);
    cache.pages.attach(id.0, node);
    cache.pages.unpin(id.0);
    id
}

#[test]
fn independent_prefix_lengths_restore_only_an_existing_checkpoint_and_reuse_after_hole() {
    let mut cache = cache();
    let request = request(512, 128, 512);
    checkpoint(&mut cache, &request, 128);
    checkpoint(&mut cache, &request, 384);
    page(&mut cache, &request, 128);
    page(&mut cache, &request, 256);
    page(&mut cache, &request, 512);
    let hit = cache.find_prefix(&request);
    assert_eq!(
        hit.blocks.evidence,
        PrefixEvidence {
            checkpoint_hit_end: 384,
            kv_contiguous_end: 256
        }
    );
    assert_eq!(hit.length, 128);
    assert!(cache.allocate(&request, hit, 384, 0, false));
    let state = cache.assignment(1);
    assert_eq!(
        state.kv_spans,
        [
            KvSpan {
                start: 128,
                end: 256,
                mode: KvSpanMode::Reuse
            },
            KvSpan {
                start: 256,
                end: 384,
                mode: KvSpanMode::Write
            },
            KvSpan {
                start: 384,
                end: 512,
                mode: KvSpanMode::Reuse
            }
        ]
    );
    cache.free(1);
}

#[test]
fn checkpoint_before_longer_kv_prefix_keeps_both_evidence_values() {
    let mut cache = cache();
    let request = request(384, 128, 384);
    checkpoint(&mut cache, &request, 128);
    for end in [128, 256, 384] {
        page(&mut cache, &request, end);
    }
    let hit = cache.find_prefix(&request);
    assert_eq!(hit.blocks.evidence.checkpoint_hit_end, 128);
    assert_eq!(hit.blocks.evidence.kv_contiguous_end, 384);
    assert_eq!(hit.length, 128);
}

#[test]
fn one_pool_failure_releases_selected_pins_without_eviction_or_partial_allocation() {
    let mut cache = cache();
    let mut request = request(129, 128, 129);
    let source = checkpoint(&mut cache, &request, 128);
    let existing = page(&mut cache, &request, 128);
    let occupied = cache.private.reserve(cache.private.capacity()).unwrap();
    let before = cache.resource_counts();
    let hit = cache.find_prefix(&request);
    assert!(!cache.allocate(&request, hit, 1, 0, false));
    assert_eq!(cache.resource_counts(), before);
    assert_eq!(cache.checkpoints.pins(source.0), 0);
    assert_eq!(cache.pages.pins(existing.0), 0);
    for id in occupied.keys() {
        cache.private.unpin(*id);
    }
    request.id = 2;
    assert!(cache.allocate(&request, cache.find_prefix(&request), 1, 0, false));
}

#[test]
fn partial_shared_page_copies_valid_codes_and_scales_and_holds_source_until_ack() {
    let mut cache = cache();
    let request = request(132, 131, 132);
    checkpoint(&mut cache, &request, 131);
    page(&mut cache, &request, 128);
    let source = page(&mut cache, &request, 131);
    let hit = cache.find_prefix(&request);
    assert_eq!(hit.length, 131);
    assert!(cache.allocate(&request, hit, 1, 0, false));
    let state = cache.assignment(1);
    assert_eq!(state.page_copies.len(), 1);
    assert_eq!(state.page_copies[0].source, source);
    assert_eq!(state.page_copies[0].valid_tokens, 3);
    assert_ne!(state.page_copies[0].destination, source);
    assert_eq!(cache.pages.pins(source.0), 1);
    cache.free(1);
    assert_eq!(cache.pages.pins(source.0), 0);
    assert_eq!(cache.resource_counts().available_private_checkpoints, 12);
}

#[test]
fn evicting_one_attachment_preserves_other_attachment_and_descendants() {
    let mut cache = cache();
    let request = request(256, 0, 256);
    let cp = checkpoint(&mut cache, &request, 128);
    let p1 = page(&mut cache, &request, 128);
    let p2 = page(&mut cache, &request, 256);
    let node = cache.checkpoints.detach(cp.0).unwrap();
    cache.tree.detach_checkpoint(node, cp);
    let path = cache.tree.lookup(&request.cache_domain, &request.token_ids);
    assert!(path.checkpoints.is_empty());
    assert_eq!(
        path.pages.iter().map(|entry| entry.2).collect::<Vec<_>>(),
        [p1, p2]
    );
    let cp = checkpoint(&mut cache, &request, 128);
    let node = cache.pages.detach(p1.0).unwrap();
    cache.tree.detach_page(node, p1);
    let path = cache.tree.lookup(&request.cache_domain, &request.token_ids);
    assert_eq!(path.checkpoints[0].2, cp);
    assert_eq!(path.pages[0].2, p2);
}

#[test]
fn checkpoint_pool_exhaustion_does_not_evict_the_selected_source_or_reserve_pages() {
    let mut cache = cache();
    let request = request(8192, 128, 8192);
    let source = checkpoint(&mut cache, &request, 128);
    let prefix_page = page(&mut cache, &request, 128);
    let held = cache.checkpoints.reserve(3).unwrap();
    let before = cache.resource_counts();
    let hit = cache.find_prefix(&request);
    assert!(!cache.allocate(&request, hit, 8064, 0, false));
    assert_eq!(cache.resource_counts(), before);
    assert_eq!(cache.checkpoints.pins(source.0), 0);
    assert_eq!(cache.pages.pins(prefix_page.0), 0);
    assert_eq!(cache.find_prefix(&request).length, 128);
    for key in held.keys() {
        cache.checkpoints.unpin(*key);
    }
}

#[test]
fn page_pool_exhaustion_rolls_back_checkpoint_and_cow_pins() {
    let mut cache = cache();
    let request = request(132, 131, 132);
    let source = checkpoint(&mut cache, &request, 131);
    page(&mut cache, &request, 128);
    let tail = page(&mut cache, &request, 131);
    let held = cache.pages.reserve(cache.pages.capacity() - 2).unwrap();
    let before = cache.resource_counts();
    assert!(!cache.allocate(&request, cache.find_prefix(&request), 1, 0, false));
    assert_eq!(cache.resource_counts(), before);
    assert_eq!(cache.checkpoints.pins(source.0), 0);
    assert_eq!(cache.pages.pins(tail.0), 0);
    for key in held.keys() {
        cache.pages.unpin(*key);
    }
}

#[test]
fn zero_row_head_reserves_cow_and_future_mtp_page_before_writing() {
    let mut cache = cache();
    let mut request = request(255, 255, 255);
    request.work = Some(WorkSpec {
        purpose: WorkPurpose::Advance,
        action: WorkAction::HeadOnly { endpoint: 255 },
        sampling: SamplingMode::Target,
    });
    checkpoint(&mut cache, &request, 255);
    page(&mut cache, &request, 128);
    let source = page(&mut cache, &request, 255);
    assert!(cache.allocate(&request, cache.find_prefix(&request), 0, 4, false));
    let state = cache.assignment(1);
    assert_eq!(state.page_copies.len(), 1);
    assert_eq!(state.page_copies[0].source, source);
    assert_eq!(state.page_copies[0].valid_tokens, 127);
    assert_eq!(state.page_table.len(), 3);
    assert_eq!(state.page_table[1].access, PageAccess::UniqueWrite);
    assert_eq!(state.page_table[2].access, PageAccess::UniqueWrite);
    assert_eq!(
        state
            .exports
            .iter()
            .filter(|export| export.role == ExportRole::PrivateEndpoint)
            .count(),
        1
    );
    cache.free(1);
    assert_eq!(cache.pages.pins(source.0), 0);
}

#[test]
fn worker_epoch_reset_rejects_hits_from_the_old_worker() {
    let mut cache = cache();
    let request = request(129, 128, 129);
    checkpoint(&mut cache, &request, 128);
    page(&mut cache, &request, 128);
    let old = cache.find_prefix(&request);
    cache.reset_worker_epoch(2);
    let new_id = checkpoint(&mut cache, &request, 128);
    page(&mut cache, &request, 128);
    assert_eq!(new_id.0.epoch, 2);
    let before = cache.resource_counts();
    assert!(!cache.allocate(&request, old, 1, 0, false));
    assert_eq!(cache.resource_counts(), before);
    assert!(cache.allocate(&request, cache.find_prefix(&request), 1, 0, false));
}

#[test]
fn mtp_requires_validated_context_at_the_selected_checkpoint() {
    let mut cache = cache();
    let mut request = request(128, 128, 128);
    request.enable_mtp = true;
    let id = checkpoint(&mut cache, &request, 128);
    page(&mut cache, &request, 128);
    assert_eq!(cache.find_prefix(&request).length, 0);
    cache.mtp_contexts.insert(id, 127);
    let hit = cache.find_prefix(&request);
    assert_eq!(hit.length, 128);
    assert_eq!(hit.blocks.mtp_endpoint, Some((127, id)));
}
