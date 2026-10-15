use anyhow::{Result, ensure};
use serde_json::{Value, json};

const KEYS: [&str; 8] = [
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "repetition_guard",
];

const DEFAULT_PRESENCE_PENALTY: f64 = 0.5;

pub fn extract(body: &Value) -> Result<Value> {
    let mut sampling = json!({});
    for key in KEYS {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            if key == "repetition_guard" {
                ensure!(value.is_boolean(), "repetition_guard must be a boolean");
            } else {
                ensure!(value.is_number(), "{key} must be numeric");
            }
            sampling[key] = value.clone();
        }
    }
    if let Some(value) = sampling["temperature"].as_f64() {
        ensure!(value >= 0.0, "temperature must be nonnegative");
    }
    if let Some(value) = sampling["top_p"].as_f64() {
        ensure!(value > 0.0 && value <= 1.0, "top_p must be in (0,1]");
    }
    if !sampling["top_k"].is_null() {
        ensure!(
            sampling["top_k"].as_i64().is_some_and(|value| value >= -1),
            "top_k must be an integer >= -1"
        );
    }
    if !sampling["seed"].is_null() {
        ensure!(
            sampling["seed"].as_i64().is_some(),
            "seed must be an integer"
        );
    }
    for key in ["frequency_penalty", "presence_penalty"] {
        if let Some(value) = sampling[key].as_f64() {
            ensure!((-2.0..=2.0).contains(&value), "{key} must be in [-2,2]");
        }
    }
    if sampling["temperature"]
        .as_f64()
        .is_some_and(|t| t > 0.0 && t < 1e-5)
    {
        sampling["temperature"] = json!(0.0);
    }
    if let Some(penalty) = sampling["repetition_penalty"].as_f64() {
        ensure!(
            (1e-5..=1e5).contains(&penalty),
            "repetition_penalty must be in [1e-5,1e5]"
        );
    }
    Ok(sampling)
}

pub fn validate_overrides(value: &Value) -> Result<Value> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected JSON object"))?;
    for (key, parameter) in object {
        ensure!(
            KEYS.contains(&key.as_str()),
            "unsupported sampling default: {key}"
        );
        ensure!(!parameter.is_null(), "{key} default cannot be null");
    }
    extract(value)
}

pub fn parse_overrides(text: &str) -> std::result::Result<Value, String> {
    let parse = || -> Result<Value> {
        ensure!(text.len() <= 4096, "sampling defaults exceed 4096 bytes");
        validate_overrides(&serde_json::from_str::<Value>(text)?)
    };
    parse().map_err(|error| error.to_string())
}

pub fn service_defaults(overrides: &Value) -> Result<Value> {
    let validated = validate_overrides(overrides)?;
    let mut defaults = json!({
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "repetition_guard": true,
    });
    defaults
        .as_object_mut()
        .unwrap()
        .extend(validated.as_object().unwrap().clone());
    Ok(defaults)
}

pub fn apply_defaults(sampling: &mut Value, defaults: &Value) {
    let parameters = sampling
        .as_object_mut()
        .expect("normalized sampling object");
    for (key, value) in defaults.as_object().expect("validated sampling defaults") {
        parameters
            .entry(key.clone())
            .or_insert_with(|| value.clone());
    }
}
