use oh_my_vllm_zmq_worker::group::collectives::CollectivePlan;
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/collectives.json")).unwrap()
}

#[test]
fn ordered_plan_binds_counts_and_rejects_invalid_storage_before_dispatch() {
    let fixture = fixture();
    let plan: CollectivePlan = serde_json::from_value(fixture["plan"].clone()).unwrap();
    assert_eq!(plan.identity(4).unwrap(), fixture["sha256"]);
    for rank in 0..4 {
        let counts = serde_json::to_value(plan.buffer_counts(rank).unwrap()).unwrap();
        assert_eq!(counts, fixture["buffer_counts"][rank as usize]);
    }
    assert!(plan.buffer_counts(4).is_err());
    assert!(plan.identity(1).is_err());
    for case in 0..11 {
        let mut input = fixture["plan"].clone();
        match case {
            0 => input["version"] = json!(2),
            1 => input["world_size"] = json!(0),
            2 => input["operations"][0]["count"] = json!(u64::MAX),
            3 => input["operations"][1]["count"] = json!(u64::MAX / 4),
            4 => input["operations"][2]["count"] = json!(u64::MAX / 4),
            5 => input["operations"][3]["counts"] = json!([[0, 0]]),
            6 => input["operations"][3]["counts"][0][0] = json!(u64::MAX),
            7 => input["operations"][3]["counts"][1][0] = json!(i64::MAX / 4),
            8 => input["operations"] = json!(vec![input["operations"][0].clone(); 4097]),
            9 => input["operations"][0]["dtype"] = json!("f64"),
            10 => input["operations"][0]["extra"] = json!(0),
            _ => unreachable!(),
        }
        match serde_json::from_value::<CollectivePlan>(input) {
            Ok(invalid) => assert!(invalid.identity(4).is_err(), "case {case}"),
            Err(_) => assert!(case >= 9),
        }
    }
    for case in 0..3 {
        let mut input = fixture["plan"].clone();
        match case {
            0 => input["operations"].as_array_mut().unwrap().swap(0, 1),
            1 => input["operations"][0]["dtype"] = json!("f16"),
            2 => input["operations"][3]["counts"][0][2] = json!(1),
            _ => unreachable!(),
        }
        let changed: CollectivePlan = serde_json::from_value(input).unwrap();
        assert_ne!(changed.identity(4).unwrap(), fixture["sha256"]);
    }
}
