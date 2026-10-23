use std::time::{Duration, Instant};

use oh_my_vllm_model_contract::ModelDescriptor;
use oh_my_vllm_zmq_worker::group::protocol::{
    ExecutionMode, RankHeader, RankReady, RankResult, RequestWork, StepPlan,
};
use oh_my_vllm_zmq_worker::group::{GroupGate, GroupLimits};
use serde_json::json;

fn descriptor(strategy: &str) -> ModelDescriptor {
    serde_json::from_value(json!({
        "schema_version": 1, "architecture": "SyntheticForGeneration",
        "model_type": "synthetic", "checkpoint_sha256": "a".repeat(64),
        "config_sha256": "b".repeat(64), "tokenizer_sha256": "c".repeat(64),
        "vocabulary_size": 101, "max_context_tokens": 97,
        "capabilities": ["text"], "weight_formats": ["bf16"],
        "cache_schema": {"schema_version": 1, "groups": [
            {"name": "latent", "kind": "compressed_kv", "block_tokens": 8,
             "capacity_blocks": 4, "storage": {"type": "primitive", "format": "bf16"},
             "layers": [0], "source": {"type": "owned"}},
            {"name": "reuse", "kind": "compressed_kv", "block_tokens": 8,
             "capacity_blocks": 4, "storage": {"type": "primitive", "format": "bf16"},
             "layers": [1], "source": {"type": "shared", "producer": "latent"}}
        ]},
        "parallel_plan": {"strategy": strategy, "world_size": 4, "ranks": [0, 1, 2, 3],
            "request_placement": if strategy == "dep4" {"rank_partitioned"} else {"replicated"}},
        "speculation": []
    }))
    .unwrap()
}

fn device(rank: u32) -> String {
    format!("GPU-{:08x}-{:04x}-{:04x}-{:04x}-{rank:012x}", 0, 0, 0, 0)
}

fn gate(strategy: &str, now: Instant) -> GroupGate {
    GroupGate::new(
        17,
        descriptor(strategy),
        (0..4).map(device).collect(),
        GroupLimits {
            max_slots: 24,
            max_requests: 64,
            max_rows: 1024,
        },
        now + Duration::from_secs(10),
    )
    .unwrap()
}

fn ready(strategy: &str, rank: u32) -> RankReady {
    let mut descriptor = descriptor(strategy);
    for group in &mut descriptor.cache_schema.groups {
        group.capacity_blocks = u64::from(4 + rank);
    }
    RankReady {
        header: RankHeader {
            version: 1,
            epoch: 17,
            rank,
        },
        device_uuid: device(rank),
        descriptor,
    }
}

fn started(strategy: &str, now: Instant) -> GroupGate {
    let mut group = gate(strategy, now);
    for rank in 0..4 {
        assert_eq!(
            group
                .accept_ready(rank, ready(strategy, rank), now)
                .unwrap(),
            rank == 3
        );
    }
    group
}

fn plan() -> StepPlan {
    StepPlan {
        step_id: 1,
        collective_plan_sha256: "d".repeat(64),
        collective_count: 9,
        requests: vec![
            RequestWork {
                request_id: 10,
                transaction_id: 90,
                owner_rank: 0,
                rows: 3,
                prefill: true,
            },
            RequestWork {
                request_id: 11,
                transaction_id: 91,
                owner_rank: 2,
                rows: 1,
                prefill: false,
            },
        ],
    }
}

fn result(rank: u32, transactions: &[u64]) -> RankResult {
    RankResult {
        header: RankHeader {
            version: 1,
            epoch: 17,
            rank,
        },
        step_id: 1,
        collective_plan_sha256: "d".repeat(64),
        completed_collectives: 9,
        transaction_ids: transactions.to_vec(),
        device_work_complete: true,
    }
}

#[test]
fn readiness_binds_rank_identity_layout_and_the_full_capacity_budget() {
    let now = Instant::now();
    let mut group = gate("dep4", now);
    assert!(group.capacities().is_none());
    assert!(
        group
            .begin(plan(), now + Duration::from_secs(1), now)
            .is_err()
    );
    for rank in [2, 0, 3, 1] {
        group.accept_ready(rank, ready("dep4", rank), now).unwrap();
    }
    let capacity = group.capacities().unwrap();
    assert_eq!(capacity.common_schema.groups[0].capacity_blocks, 4);
    assert_eq!(capacity.per_rank[&3]["latent"], 7);
    assert_eq!(capacity.owned_slots, 22);
    assert!(!group.is_failed());

    for case in 0..10 {
        let mut group = gate("dep4", now);
        let mut message = ready("dep4", 0);
        match case {
            0 => message.header.version = 2,
            1 => message.header.epoch = 18,
            2 => message.header.rank = 1,
            3 => message.device_uuid = device(3),
            4 => message.descriptor.checkpoint_sha256 = "f".repeat(64),
            5 => {
                message.descriptor.cache_schema.groups[0].storage =
                    oh_my_vllm_model_contract::CacheStorage::Primitive {
                        format: oh_my_vllm_model_contract::ScalarFormat::F32,
                    }
            }
            6 => message.descriptor.cache_schema.groups[0].capacity_blocks = 3,
            7 => message.descriptor.cache_schema.groups[0].capacity_blocks = 5,
            8 => {
                for entry in &mut message.descriptor.cache_schema.groups {
                    entry.capacity_blocks = u64::from(u32::MAX - 1);
                }
            }
            9 => {
                for entry in &mut message.descriptor.cache_schema.groups {
                    entry.capacity_blocks = 25;
                }
            }
            _ => unreachable!(),
        }
        assert!(group.accept_ready(0, message, now).is_err(), "case {case}");
        assert!(group.is_failed());
        assert!(group.accept_ready(0, ready("dep4", 0), now).is_err());
    }
    let mut group = gate("dep4", now);
    group.accept_ready(0, ready("dep4", 0), now).unwrap();
    assert!(group.accept_ready(0, ready("dep4", 0), now).is_err());
    assert!(group.is_failed());
    let mut group = gate("dep4", now);
    assert!(group.accept_ready(4, ready("dep4", 3), now).is_err());
    for case in 0..5 {
        let mut devices: Vec<_> = (0..4).map(device).collect();
        let mut limits = GroupLimits {
            max_slots: 24,
            max_requests: 64,
            max_rows: 1024,
        };
        let mut epoch = 17;
        match case {
            0 => epoch = 0,
            1 => devices[3] = device(0),
            2 => devices[3] = "not-a-device".into(),
            3 => limits.max_slots = 15,
            4 => limits.max_requests = 0,
            _ => unreachable!(),
        }
        assert!(GroupGate::new(epoch, descriptor("dep4"), devices, limits, now).is_err());
    }
}

#[test]
fn all_ranks_complete_one_plan_including_empty_partitions() {
    let now = Instant::now();
    for strategy in ["tp4", "tep4", "dep4"] {
        let mut group = started(strategy, now);
        let mut step = plan();
        if strategy != "dep4" {
            step.requests[1].owner_rank = 0;
        }
        for case in 0..8 {
            let mut invalid = step.clone();
            match case {
                0 => invalid.step_id = 0,
                1 => invalid.requests[1].request_id = 10,
                2 => invalid.requests[1].transaction_id = 90,
                3 => invalid.requests[0].transaction_id = 0,
                4 => invalid.requests[0].owner_rank = 4,
                5 => invalid.requests[0].rows = 0,
                6 => invalid.requests[0].rows = u32::MAX,
                7 => invalid.collective_plan_sha256 = "invalid".into(),
                _ => unreachable!(),
            }
            assert!(
                group
                    .begin(invalid, now + Duration::from_secs(1), now)
                    .is_err()
            );
            assert!(!group.is_failed());
        }
        let dispatch = group
            .begin(step.clone(), now + Duration::from_secs(1), now)
            .unwrap();
        assert_eq!(dispatch.mode, ExecutionMode::Eager);
        let rows: Vec<_> = dispatch.ranks.iter().map(|rank| rank.rows).collect();
        assert_eq!(
            rows,
            if strategy == "dep4" {
                vec![3, 0, 1, 0]
            } else {
                vec![4; 4]
            }
        );
        let transactions: [&[u64]; 4] = if strategy == "dep4" {
            [&[90], &[], &[91], &[]]
        } else {
            [&[90, 91]; 4]
        };
        for rank in [3, 0, 2] {
            let message = result(rank, transactions[rank as usize]);
            let bytes = rmp_serde::to_vec_named(&message).unwrap();
            let decoded: RankResult = rmp_serde::from_slice(&bytes).unwrap();
            assert!(group.accept_result(rank, decoded, now).unwrap().is_none());
        }
        let complete = group
            .accept_result(1, result(1, transactions[1]), now)
            .unwrap()
            .unwrap();
        assert_eq!(complete.epoch(), 17);
        assert_eq!(complete.step_id(), 1);
        assert_eq!(complete.transactions(), &[(10, 90), (11, 91)]);
        assert!(
            group
                .begin(step.clone(), now + Duration::from_secs(1), now)
                .is_err()
        );
        step.step_id = 2;
        for request in &mut step.requests {
            request.prefill = false;
            request.transaction_id += 2;
        }
        assert_eq!(
            group
                .begin(step, now + Duration::from_secs(1), now)
                .unwrap()
                .mode,
            ExecutionMode::Graph
        );
    }
}

#[test]
fn invalid_or_missing_completion_fails_without_partial_publication() {
    let now = Instant::now();
    for case in 0..10 {
        let mut group = started("dep4", now);
        group
            .begin(plan(), now + Duration::from_secs(1), now)
            .unwrap();
        assert!(
            group
                .accept_result(0, result(0, &[90]), now)
                .unwrap()
                .is_none()
        );
        let mut message = result(2, &[91]);
        match case {
            0 => message.header.version = 2,
            1 => message.header.epoch = 16,
            2 => message.header.rank = 3,
            3 => message.step_id = 2,
            4 => message.collective_plan_sha256 = "f".repeat(64),
            5 => message.completed_collectives = 8,
            6 => message.transaction_ids.clear(),
            7 => message.transaction_ids = vec![91, 91],
            8 => message.transaction_ids = vec![90],
            9 => message.device_work_complete = false,
            _ => unreachable!(),
        }
        assert!(group.accept_result(2, message, now).is_err(), "case {case}");
        assert!(group.is_failed());
        assert!(group.accept_result(2, result(2, &[91]), now).is_err());
    }
    let mut group = started("dep4", now);
    group
        .begin(plan(), now + Duration::from_secs(1), now)
        .unwrap();
    assert!(
        group
            .accept_result(1, result(1, &[]), now)
            .unwrap()
            .is_none()
    );
    assert!(group.accept_result(1, result(1, &[]), now).is_err());
    assert!(group.is_failed());

    let mut group = started("dep4", now);
    group
        .begin(plan(), now + Duration::from_secs(1), now)
        .unwrap();
    for rank in 0..3 {
        let ids: &[u64] = match rank {
            0 => &[90],
            2 => &[91],
            _ => &[],
        };
        assert!(
            group
                .accept_result(rank, result(rank, ids), now)
                .unwrap()
                .is_none()
        );
    }
    assert!(
        group
            .accept_result(3, result(3, &[]), now + Duration::from_secs(1))
            .is_err()
    );
    assert!(group.is_failed());
    let mut starting = gate("dep4", now);
    assert!(
        starting
            .check_deadline(now + Duration::from_secs(10))
            .is_err()
    );
    assert!(starting.is_failed());
    let mut group = started("dep4", now);
    group.fail_rank(2);
    assert!(
        group
            .begin(plan(), now + Duration::from_secs(1), now)
            .is_err()
    );
    assert!(group.is_failed());
}
