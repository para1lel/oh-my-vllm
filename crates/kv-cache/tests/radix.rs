use oh_my_vllm_kv_cache::radix::{CacheDomain, CheckpointId, PageId, SlotPool, TokenRadix};

#[test]
fn exact_compressed_edges_and_domains_do_not_alias_matching_token_hashes() {
    let domain = CacheDomain::default();
    let mut other = domain.clone();
    other.namespace = "other".into();
    let mut tree = TokenRadix::default();
    let mut pool = SlotPool::new(7, 1, 2);
    let slots = pool.reserve(2).unwrap();
    let first = tree.ensure(&domain, &[1, 2, 3, 4]);
    let second = tree.ensure(&domain, &[1, 2, 9]);
    tree.attach_checkpoint(first, CheckpointId(slots.keys()[0]));
    tree.attach_checkpoint(second, CheckpointId(slots.keys()[1]));
    assert_eq!(
        tree.lookup(&domain, &[1, 2, 3, 4, 8]).checkpoints[0].1,
        first
    );
    assert_eq!(tree.lookup(&domain, &[1, 2, 9, 8]).checkpoints[0].1, second);
    assert!(tree.lookup(&domain, &[1, 2, 3, 5]).checkpoints.is_empty());
    assert!(tree.lookup(&other, &[1, 2, 3, 4]).checkpoints.is_empty());
}

#[test]
fn live_readers_block_slot_reuse_and_versions_reject_aba() {
    let mut pool = SlotPool::new(3, 1, 1);
    let first = pool.reserve(1).unwrap().keys()[0];
    assert!(pool.pin(first));
    pool.unpin(first);
    assert!(pool.reserve(1).is_none());
    pool.unpin(first);
    let next = pool.reserve(1).unwrap().keys()[0];
    assert_eq!(first.slot, next.slot);
    assert_ne!(first.version, next.version);
    assert!(!pool.pin(first));
    let foreign_epoch = oh_my_vllm_kv_cache::radix::SlotKey { epoch: 4, ..next };
    assert!(!pool.pin(foreign_epoch));
}

#[test]
fn independent_detach_keeps_a_shared_node_and_recycled_node_id_is_stale() {
    let mut tree = TokenRadix::default();
    let mut checkpoints = SlotPool::new(1, 1, 1);
    let mut pages = SlotPool::new(1, 1, 1);
    let checkpoint = CheckpointId(checkpoints.reserve(1).unwrap().keys()[0]);
    let page = PageId(pages.reserve(1).unwrap().keys()[0]);
    let domain = CacheDomain::default();
    let node = tree.ensure(&domain, &[4, 5]);
    tree.attach_checkpoint(node, checkpoint);
    tree.attach_page(node, page);
    tree.detach_page(node, page);
    assert_eq!(tree.checkpoint(node), Some(checkpoint));
    tree.detach_checkpoint(node, checkpoint);
    let replacement = tree.ensure(&domain, &[6, 7]);
    tree.attach_checkpoint(replacement, checkpoint);
    assert_eq!(tree.checkpoint(node), None);
    tree.detach_checkpoint(node, checkpoint);
    assert_eq!(tree.checkpoint(replacement), Some(checkpoint));
}
