use oh_my_vllm_kv_cache::state::{AllocationRequest, StateArena};
use oh_my_vllm_model_contract::{
    CacheGroup, CacheKind, CacheSchema, CacheSource, CacheStorage, ParallelPlan, ParallelStrategy,
    RequestPlacement, ScalarFormat,
};

fn schema() -> CacheSchema {
    let group = |name: &str, kind, layers, source| CacheGroup {
        name: name.into(),
        kind,
        block_tokens: 8,
        capacity_blocks: 4,
        storage: CacheStorage::Primitive {
            format: ScalarFormat::F32,
        },
        layers,
        source,
    };
    CacheSchema {
        schema_version: 1,
        groups: vec![
            group(
                "latent",
                CacheKind::CompressedKv,
                vec![0],
                CacheSource::Owned {},
            ),
            group(
                "window",
                CacheKind::SlidingWindow,
                vec![0],
                CacheSource::Owned {},
            ),
            group(
                "reuse",
                CacheKind::CompressedKv,
                vec![1],
                CacheSource::Shared {
                    producer: "latent".into(),
                },
            ),
        ],
    }
}

fn parallel() -> ParallelPlan {
    ParallelPlan {
        strategy: ParallelStrategy::Dep4,
        world_size: 4,
        ranks: vec![0, 1, 2, 3],
        request_placement: RequestPlacement::RankPartitioned,
    }
}

fn arena(epoch: u64) -> StateArena {
    StateArena::new(epoch, &schema(), &parallel(), 32).unwrap()
}

fn request(group: &str, rank: u32, blocks: u32) -> AllocationRequest {
    AllocationRequest {
        group: group.into(),
        rank,
        blocks,
    }
}

#[test]
fn reservation_checks_all_groups_before_allocation() {
    let state = arena(7);
    let first = state
        .reserve(
            1,
            19,
            &[request("latent", 0, 2), request("window", 3, 2)],
            &[],
        )
        .unwrap();
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 1);
    assert_eq!(state.free_blocks("reuse", 0).unwrap(), 1);
    assert_eq!(state.free_blocks("window", 3).unwrap(), 1);
    assert!(
        state
            .reserve(
                2,
                20,
                &[request("latent", 1, 2), request("window", 3, 2)],
                &[]
            )
            .is_err()
    );
    assert_eq!(state.free_blocks("latent", 1).unwrap(), 3);
    assert_eq!(state.free_blocks("window", 3).unwrap(), 1);
    drop(first);
    for rank in 0..4 {
        for group in ["latent", "window", "reuse"] {
            assert_eq!(state.free_blocks(group, rank).unwrap(), 3);
        }
    }
    assert!(StateArena::new(7, &schema(), &parallel(), 31).is_err());
    let mut bad = schema();
    bad.groups[2].block_tokens = 4;
    bad.validate().unwrap();
    assert!(StateArena::new(7, &bad, &parallel(), 32).is_err());
    assert!(StateArena::new(0, &schema(), &parallel(), 32).is_err());
}

#[test]
fn cancelled_writes_keep_readers_alive_until_the_last_lease() {
    let state = arena(1);
    let accepted = state
        .reserve(1, 41, &[request("latent", 0, 2)], &[])
        .unwrap()
        .into_storage();
    let saved = accepted.blocks().to_vec();
    let pinned = accepted.pin().unwrap();
    let pending = state
        .reserve(2, 41, &[request("window", 0, 2)], &[&accepted, &pinned])
        .unwrap();
    assert!(pending.blocks().iter().all(|block| !saved.contains(block)));
    drop(accepted);
    drop(pinned);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 1);
    assert_eq!(state.free_blocks("window", 0).unwrap(), 1);
    pending.abort();
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);
    assert_eq!(state.free_blocks("window", 0).unwrap(), 3);
    let owned = state
        .reserve(3, 42, &[request("latent", 0, 3)], &[])
        .unwrap()
        .into_storage();
    let late_reader = owned.pin().unwrap();
    drop(owned);
    assert_eq!(late_reader.blocks().len(), 3);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 0);
    drop(state);
    assert_eq!(late_reader.epoch(), 1);
    drop(late_reader);
}

#[test]
fn foreign_owners_stale_steps_and_reused_slot_identities_are_rejected() {
    let state = arena(3);
    let foreign = arena(3);
    let lease = foreign
        .reserve(1, 1, &[request("latent", 0, 1)], &[])
        .unwrap()
        .into_storage();
    assert!(
        state
            .reserve(1, 1, &[request("latent", 0, 1)], &[&lease])
            .is_err()
    );
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);
    let first = state
        .reserve(2, 1, &[request("latent", 0, 3)], &[])
        .unwrap();
    let old_blocks = first.blocks().to_vec();
    let first_key = first.key();
    assert_eq!(first_key.epoch, 3);
    assert_eq!(first_key.step_id, 2);
    assert_eq!(first_key.request_id, 1);
    first.abort();
    let second = state
        .reserve(2, 2, &[request("latent", 0, 3)], &[])
        .unwrap();
    assert!(second.key().transaction_id > first_key.transaction_id);
    for new in second.blocks() {
        let old = old_blocks.iter().find(|old| old.slot == new.slot).unwrap();
        assert_eq!(old.group, new.group);
        assert_eq!(old.rank, new.rank);
        assert!(new.generation > old.generation);
    }
    drop(second);
    for (step, requests) in [
        (0, vec![request("latent", 0, 1)]),
        (1, vec![request("latent", 0, 1)]),
        (3, vec![request("reuse", 0, 1)]),
        (3, vec![request("missing", 0, 1)]),
        (3, vec![request("latent", 4, 1)]),
        (3, vec![request("latent", 0, 0)]),
        (3, vec![request("latent", 0, 1), request("latent", 0, 1)]),
    ] {
        assert!(state.reserve(step, 2, &requests, &[]).is_err());
        assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);
    }
    assert!(state.free_blocks("missing", 0).is_err());
    assert!(state.free_blocks("latent", 4).is_err());
    assert_eq!(foreign.free_blocks("latent", 0).unwrap(), 2);
}

#[test]
fn selected_storage_keeps_only_authorized_old_and_new_blocks() {
    let state = arena(9);
    let accepted = state
        .reserve(1, 41, &[request("latent", 0, 2)], &[])
        .unwrap()
        .into_storage();
    let pending = state
        .reserve(2, 41, &[request("window", 3, 3)], &[&accepted])
        .unwrap();
    let selected_blocks = [pending.blocks()[2].clone(), accepted.blocks()[0].clone()];
    let selected = pending.select_storage(&selected_blocks).unwrap();
    assert_eq!(selected.blocks(), &selected_blocks);
    assert_eq!(selected.epoch(), 9);
    drop(accepted);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 1);
    assert_eq!(state.free_blocks("window", 3).unwrap(), 0);
    drop(pending);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 2);
    assert_eq!(state.free_blocks("window", 3).unwrap(), 2);

    let old_only = selected.select(&selected_blocks[1..]).unwrap();
    let empty = selected.select(&[]).unwrap();
    assert!(empty.blocks().is_empty());
    drop(selected);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 2);
    assert_eq!(state.free_blocks("window", 3).unwrap(), 3);
    drop(old_only);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);
    drop(state);
    assert_eq!(empty.epoch(), 9);
    drop(empty);
}

#[test]
fn invalid_selections_do_not_pin_valid_prefixes_or_change_source_owners() {
    let state = arena(11);
    let source = state
        .reserve(1, 5, &[request("latent", 0, 2)], &[])
        .unwrap()
        .into_storage();
    let unrelated = state
        .reserve(1, 6, &[request("window", 1, 1)], &[])
        .unwrap();
    let pending = state
        .reserve(2, 5, &[request("window", 3, 2)], &[&source])
        .unwrap();
    let valid = source.blocks()[0].clone();
    let mut stale = valid.clone();
    stale.generation += 1;
    let mut unknown = valid.clone();
    unknown.group = "missing".into();
    let mut alias = valid.clone();
    alias.group = "reuse".into();
    let mut null = valid.clone();
    null.slot = 0;
    for invalid in [
        valid.clone(),
        stale,
        unknown,
        alias,
        null,
        unrelated.blocks()[0].clone(),
    ] {
        let selection = [valid.clone(), invalid];
        assert!(source.select(&selection).is_err());
        assert!(pending.select_storage(&selection).is_err());
        assert_eq!(state.free_blocks("latent", 0).unwrap(), 1);
        assert_eq!(state.free_blocks("window", 3).unwrap(), 1);
    }
    assert!(source.select(pending.blocks()).is_err());
    assert!(pending.select_storage(&[]).unwrap().blocks().is_empty());
    drop(pending);
    drop(source);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);
    assert_eq!(state.free_blocks("window", 3).unwrap(), 3);
    assert_eq!(state.free_blocks("window", 1).unwrap(), 2);
    drop(unrelated);
    assert_eq!(state.free_blocks("window", 1).unwrap(), 3);
}

#[test]
fn append_rows_require_live_pins_and_publish_atomic_versions() {
    let epoch = (1_u64 << 40) + 9;
    let owner = (1_u64 << 40) + 41;
    let state = arena(epoch);
    let mut initial = state
        .reserve(1, owner, &[request("latent", 0, 2)], &[])
        .unwrap();
    let blocks = initial.blocks().to_vec();
    assert!(initial.create_row_root(&blocks[0], 0).is_err());
    let first = initial.create_row_root(&blocks[0], 8).unwrap();
    let second = initial.create_row_root(&blocks[1], 4).unwrap();
    assert!(initial.create_row_root(&blocks[0], 9).is_err());
    assert_eq!(first.block(), &blocks[0]);
    assert_eq!(first.storage().blocks(), &blocks[..1]);
    assert_eq!((first.version(), first.rows(), first.capacity()), (0, 0, 8));

    let foreign = arena(epoch);
    let foreign_tx = foreign
        .reserve(1, owner, &[request("latent", 0, 2)], &[])
        .unwrap();
    let foreign_root = foreign_tx
        .create_row_root(&foreign_tx.blocks()[0], 8)
        .unwrap();
    assert_eq!(foreign_root.block(), first.block());
    assert!(initial.append_rows(&[(foreign_root, 1)]).is_err());
    let mut forged = blocks[0].clone();
    forged.generation += 1;
    assert!(initial.create_row_root(&forged, 8).is_err());

    let mut unpinned = state.reserve(2, owner, &[], &[]).unwrap();
    assert!(unpinned.append_rows(&[(first.clone(), 1)]).is_err());
    let mut other_request = state
        .reserve(2, owner + (1 << 32), &[], &[first.storage()])
        .unwrap();
    initial.validate_row_version(&first).unwrap();
    other_request.validate_row_version(&first).unwrap();
    assert!(unpinned.validate_row_version(&first).is_err());
    assert!(foreign_tx.validate_row_version(&first).is_err());
    assert!(other_request.append_rows(&[(first.clone(), 1)]).is_err());
    assert!(other_request.create_row_root(first.block(), 8).is_err());

    assert!(initial.append_rows(&[]).is_ok());
    assert!(initial.row_grants().is_empty());
    assert!(initial.publish_rows(&[]).is_err());
    assert!(initial.append_rows(&[(first.clone(), 0)]).is_err());
    assert!(
        initial
            .append_rows(&[(first.clone(), 1), (first.clone(), 1)])
            .is_err()
    );
    assert!(
        initial
            .append_rows(&[(first.clone(), 4), (second.clone(), 5)])
            .is_err()
    );
    assert!(initial.row_grants().is_empty());
    initial
        .append_rows(&[(first.clone(), 4), (second.clone(), 2)])
        .unwrap();
    let grants = initial.row_grants().to_vec();
    assert_eq!(grants.len(), 2);
    for (grant, block, end) in [(&grants[0], &blocks[0], 4), (&grants[1], &blocks[1], 2)] {
        assert_eq!(grant.transaction, initial.key());
        assert_eq!(&grant.root, block);
        assert_eq!((grant.version, grant.begin, grant.end), (0, 0, end));
    }
    assert!(initial.append_rows(&[(first.clone(), 1)]).is_err());
    let mut contender = state
        .reserve(3, owner, &[], &[first.storage(), second.storage()])
        .unwrap();
    assert!(contender.append_rows(&[(second.clone(), 1)]).is_err());
    assert!(initial.publish_rows(&[3]).is_err());
    assert!(initial.publish_rows(&[3, 3]).is_err());
    assert!(contender.append_rows(&[(first.clone(), 1)]).is_err());
    let published = initial.publish_rows(&[3, 0]).unwrap();
    assert!(initial.row_grants().is_empty());
    assert_eq!(published.len(), 2);
    assert_eq!((published[0].version(), published[0].rows()), (1, 3));
    assert_eq!((published[1].version(), published[1].rows()), (1, 0));
    assert_eq!((first.version(), first.rows()), (0, 0));
    initial.validate_row_read(&published[0], 0, 3).unwrap();
    other_request
        .validate_row_read(&published[0], 1, 3)
        .unwrap();
    assert!(unpinned.validate_row_read(&published[0], 0, 1).is_err());
    assert!(foreign_tx.validate_row_read(&published[0], 0, 1).is_err());
    assert!(initial.validate_row_read(&first, 0, 1).is_err());
    for (begin, end) in [(0, 0), (2, 1), (0, 4), (3, u32::MAX)] {
        assert!(
            initial
                .validate_row_read(&published[0], begin, end)
                .is_err()
        );
    }
    assert!(initial.publish_rows(&[3, 0]).is_err());
    assert!(initial.append_rows(&[(published[0].clone(), 1)]).is_err());
    assert!(contender.append_rows(&[(first.clone(), 1)]).is_err());
    assert!(contender.append_rows(&[(published[0].clone(), 6)]).is_err());
    contender.append_rows(&[(published[0].clone(), 2)]).unwrap();
    other_request
        .validate_row_read(&published[0], 0, 3)
        .unwrap();
    assert!(contender.validate_row_read(&published[0], 0, 5).is_err());
    assert_eq!(
        (
            contender.row_grants()[0].begin,
            contender.row_grants()[0].end
        ),
        (3, 5)
    );
    assert!(contender.publish_rows(&[2]).is_err());
    let detached = vec![contender];
    let mut retry = state
        .reserve(4, owner, &[], &[published[0].storage()])
        .unwrap();
    assert!(retry.append_rows(&[(published[0].clone(), 1)]).is_err());
    drop(detached);
    retry.append_rows(&[(published[0].clone(), 1)]).unwrap();
    assert_eq!(
        (retry.row_grants()[0].begin, retry.row_grants()[0].end),
        (3, 4)
    );
    let newest = retry.publish_rows(&[4]).unwrap().remove(0);
    assert_eq!((newest.version(), newest.rows()), (2, 4));
    other_request
        .validate_row_read(&published[0], 0, 3)
        .unwrap();
    assert!(
        other_request
            .validate_row_read(&published[0], 0, 4)
            .is_err()
    );
    assert!(other_request.append_rows(&[(newest.clone(), 1)]).is_err());
    drop(retry);
    drop(initial.into_storage());
    drop(unpinned);
    drop(other_request);
    drop(published);
    drop(second);
    drop(newest);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 2);
    drop(first);
    assert_eq!(state.free_blocks("latent", 0).unwrap(), 3);

    let mut reused = state
        .reserve(5, owner, &[request("latent", 0, 3)], &[])
        .unwrap();
    let reused_block = reused
        .blocks()
        .iter()
        .find(|block| block.slot == blocks[0].slot)
        .unwrap()
        .clone();
    assert!(reused_block.generation > blocks[0].generation);
    let root = reused.create_row_root(&reused_block, u32::MAX).unwrap();
    assert_eq!((root.version(), root.rows()), (0, 0));
    assert!(reused.create_row_root(&blocks[0], 8).is_err());
    reused.append_rows(&[(root.clone(), u32::MAX)]).unwrap();
    let full = reused.publish_rows(&[u32::MAX]).unwrap().remove(0);
    let mut overflow = state.reserve(6, owner, &[], &[full.storage()]).unwrap();
    assert!(overflow.append_rows(&[(full, 1)]).is_err());
    assert!(overflow.row_grants().is_empty());
    drop(overflow);
    drop(state);
    drop(reused.into_storage());
    assert_eq!(root.storage().epoch(), epoch);
    assert_eq!(root.block(), &reused_block);
}
