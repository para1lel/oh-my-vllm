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
