use oh_my_vllm_model_contract::ModelDescriptor;
use serde_json::{Value, json};

fn descriptor() -> Value {
    json!({
        "schema_version": 1,
        "architecture": "SyntheticForGeneration",
        "model_type": "synthetic",
        "checkpoint_sha256": "a".repeat(64),
        "config_sha256": "b".repeat(64),
        "tokenizer_sha256": "c".repeat(64),
        "vocabulary_size": 101,
        "max_context_tokens": 97,
        "capabilities": ["text", "images", "video", "tools", "reasoning",
                         "structured_outputs", "prefix_cache", "host_offload"],
        "weight_formats": ["bf16", "f32", "qwen_fp8_e4m3_block128x128_fp32_scales"],
        "cache_schema": {
            "schema_version": 1,
            "groups": [
                {"name": "latent", "kind": "compressed_kv", "block_tokens": 7,
                 "capacity_blocks": 13, "storage": {"type": "primitive", "format": "bf16"},
                 "layers": [0, 2], "source": {"type": "owned"}},
                {"name": "index", "kind": "index_keys", "block_tokens": 3,
                 "capacity_blocks": 5, "storage": {"type": "layout", "id": "index_v1",
                 "version": 1}, "layers": [1], "source": {"type": "shared", "producer": "latent"}}
            ]
        },
        "parallel_plan": {
            "strategy": "tp1", "world_size": 1, "ranks": [0], "request_placement": "replicated"
        },
        "speculation": [
            {"method": "mtp", "predictor_stages": 1, "max_draft_tokens": 4,
             "max_verification_rows": 5}
        ]
    })
}

fn decode(value: Value) -> ModelDescriptor {
    serde_json::from_value(value).unwrap()
}

#[test]
fn wire_roundtrips_and_rejects_unknown_fields() {
    let value = descriptor();
    let contract = decode(value.clone());
    contract.validate().unwrap();
    assert_eq!(serde_json::to_value(&contract).unwrap(), value);
    let bytes = rmp_serde::to_vec_named(&contract).unwrap();
    let restored: ModelDescriptor = rmp_serde::from_slice(&bytes).unwrap();
    restored.validate().unwrap();
    assert_eq!(restored, contract);

    for path in [
        "",
        "/cache_schema",
        "/cache_schema/groups/0",
        "/cache_schema/groups/0/storage",
        "/cache_schema/groups/0/source",
        "/cache_schema/groups/1/storage",
        "/cache_schema/groups/1/source",
        "/parallel_plan",
        "/speculation/0",
    ] {
        let mut value = descriptor();
        value.pointer_mut(path).unwrap()["unknown_field"] = json!(true);
        assert!(
            serde_json::from_value::<ModelDescriptor>(value.clone()).is_err(),
            "{path}"
        );
        let bytes = rmp_serde::to_vec_named(&value).unwrap();
        assert!(
            rmp_serde::from_slice::<ModelDescriptor>(&bytes).is_err(),
            "{path}"
        );
    }
}

#[test]
fn rejects_invalid_descriptor_mutations() {
    let cases = [
        ("/schema_version", json!(2)),
        ("/cache_schema/schema_version", json!(2)),
        ("/architecture", json!("")),
        ("/model_type", json!("/checkpoint/path")),
        ("/architecture", json!("x".repeat(129))),
        ("/checkpoint_sha256", json!("")),
        ("/checkpoint_sha256", json!("0".repeat(63))),
        ("/config_sha256", json!("A".repeat(64))),
        ("/tokenizer_sha256", json!("g".repeat(64))),
        ("/vocabulary_size", json!(0)),
        ("/max_context_tokens", json!(0)),
        ("/vocabulary_size", json!(-1)),
        ("/capabilities", json!(["text", "text"])),
        ("/capabilities/0", json!("audio")),
        ("/weight_formats", json!([])),
        ("/weight_formats", json!(["bf16", "bf16"])),
        ("/weight_formats/0", json!("fp8")),
        ("/cache_schema/groups", json!([])),
        ("/cache_schema/groups/0/name", json!("")),
        ("/cache_schema/groups/1/name", json!("latent")),
        ("/cache_schema/groups/0/kind", json!("untyped")),
        ("/cache_schema/groups/0/block_tokens", json!(0)),
        ("/cache_schema/groups/0/capacity_blocks", json!(0)),
        ("/cache_schema/groups/0/layers", json!([])),
        ("/cache_schema/groups/0/layers", json!([0, 0])),
        (
            "/cache_schema/groups/0/layers",
            json!((0..4097).collect::<Vec<_>>()),
        ),
        ("/cache_schema/groups/0/storage/format", json!("fp8")),
        ("/cache_schema/groups/1/storage/id", json!("")),
        ("/cache_schema/groups/1/storage/version", json!(0)),
        ("/cache_schema/groups/1/source/producer", json!("missing")),
        ("/cache_schema/groups/1/source/producer", json!("index")),
        ("/speculation/0/method", json!("unknown")),
        (
            "/speculation",
            json!([
                {"method": "mtp", "predictor_stages": 1, "max_draft_tokens": 4,
                 "max_verification_rows": 5},
                {"method": "mtp", "predictor_stages": 1, "max_draft_tokens": 4,
                 "max_verification_rows": 5}
            ]),
        ),
        ("/speculation/0/predictor_stages", json!(0)),
        ("/speculation/0/max_draft_tokens", json!(0)),
        ("/speculation/0/max_verification_rows", json!(0)),
        ("/speculation/0/max_verification_rows", json!(4)),
        ("/speculation/0/max_draft_tokens", json!(u32::MAX)),
    ];
    for (path, replacement) in cases {
        let mut value = descriptor();
        *value.pointer_mut(path).unwrap() = replacement;
        let result = serde_json::from_value::<ModelDescriptor>(value);
        assert!(
            !result.is_ok_and(|contract| contract.validate().is_ok()),
            "{path}"
        );
    }
}

#[test]
fn validates_cache_parallel_and_speculation_rules() {
    for (strategy, ranks, placement) in [
        ("tp1", vec![0], "replicated"),
        ("tp4", vec![3, 1, 0, 2], "replicated"),
        ("tep4", vec![0, 1, 2, 3], "replicated"),
        ("dep4", vec![0, 1, 2, 3], "rank_partitioned"),
    ] {
        let mut value = descriptor();
        value["parallel_plan"] = json!({
            "strategy": strategy, "world_size": ranks.len(),
            "ranks": ranks, "request_placement": placement
        });
        let contract = decode(value);
        contract.parallel_plan.validate().unwrap();
        contract.validate().unwrap();
    }
    for kind in [
        "full_attention",
        "recurrent",
        "sliding_window",
        "compressed_kv",
        "index_keys",
        "model_checkpoint",
    ] {
        let mut value = descriptor();
        value["cache_schema"]["groups"][0]["kind"] = json!(kind);
        value["speculation"] = json!([]);
        decode(value).validate().unwrap();
    }
    let mut value = descriptor();
    value["speculation"][0]["predictor_stages"] = json!(9);
    decode(value).speculation[0].validate().unwrap();

    for (path, replacement) in [
        ("/parallel_plan/strategy", json!("tp16")),
        ("/parallel_plan/world_size", json!(0)),
        ("/parallel_plan/world_size", json!(4)),
        ("/parallel_plan/world_size", json!(16)),
        ("/parallel_plan/ranks", json!([])),
        ("/parallel_plan/ranks", json!([1])),
        (
            "/parallel_plan/request_placement",
            json!("rank_partitioned"),
        ),
    ] {
        let mut value = descriptor();
        *value.pointer_mut(path).unwrap() = replacement;
        let result = serde_json::from_value::<ModelDescriptor>(value);
        assert!(
            !result.is_ok_and(|contract| contract.validate().is_ok()),
            "{path}"
        );
    }
    for ranks in [vec![0, 0, 2, 3], vec![0, 1, 2, 4], vec![0, 1, 2]] {
        let mut value = descriptor();
        value["parallel_plan"] = json!({
            "strategy": "dep4", "world_size": 4,
            "ranks": ranks, "request_placement": "rank_partitioned"
        });
        assert!(decode(value).parallel_plan.validate().is_err());
    }
    let mut value = descriptor();
    value["cache_schema"]["groups"][0]["source"] = json!({"type": "shared", "producer": "index"});
    assert!(decode(value).cache_schema.validate().is_err());

    let mut value = descriptor();
    let groups = value["cache_schema"]["groups"].as_array_mut().unwrap();
    let mut group = groups[0].clone();
    groups.clear();
    for index in 0..257 {
        group["name"] = json!(format!("group_{index}"));
        groups.push(group.clone());
    }
    assert!(decode(value).cache_schema.validate().is_err());
}
