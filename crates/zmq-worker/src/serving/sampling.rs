//! Validated service defaults and explicit per-request sampling parameters.
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

// Shared by every HTTP API and speculative mode, including test services.
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
    // A temperature in (0, 1e-5) causes NaN in softmax; treat it as greedy.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_zero_and_null_omission_have_different_precedence() {
        let defaults = parse_overrides(r#"{"presence_penalty":0.5,"top_p":0.9}"#).unwrap();
        let mut explicit = extract(&json!({"presence_penalty": 0, "top_p": null})).unwrap();
        apply_defaults(&mut explicit, &defaults);
        assert_eq!(explicit, json!({"presence_penalty": 0, "top_p": 0.9}));
        assert_eq!(defaults, json!({"presence_penalty": 0.5, "top_p": 0.9}));
    }

    #[test]
    fn service_defaults_use_request_validation_without_silent_unsupported_keys() {
        let valid = json!({
            "temperature": 0.000001, "top_p": 0.95, "top_k": 20, "seed": -7,
            "frequency_penalty": -2, "presence_penalty": 2, "repetition_penalty": 1,
        });
        assert_eq!(
            parse_overrides(&valid.to_string()).unwrap(),
            extract(&valid).unwrap()
        );
        assert_eq!(extract(&valid).unwrap()["temperature"], 0.0);
        for invalid in [
            json!([]),
            json!(null),
            json!({"min_p": 0}),
            json!({"max_tokens": 100}),
            json!({"presence_penalty": null}),
            json!({"presence_penalty": true}),
            json!({"presence_penalty": 2.1}),
            json!({"frequency_penalty": -2.1}),
            json!({"temperature": -1}),
            json!({"top_p": 0}),
            json!({"top_p": 1.1}),
            json!({"top_k": 1.5}),
            json!({"top_k": -2}),
            json!({"seed": 1.5}),
            json!({"seed": 1u64 << 63}),
            json!({"repetition_penalty": 0}),
        ] {
            assert!(parse_overrides(&invalid.to_string()).is_err(), "{invalid}");
        }
        assert!(parse_overrides(&" ".repeat(4097)).is_err());
        assert_eq!(parse_overrides("{}").unwrap(), json!({}));
    }

    #[test]
    fn shared_defaults_survive_unrelated_server_overrides_and_accept_explicit_zero() {
        assert_eq!(
            service_defaults(&json!({})).unwrap(),
            json!({"presence_penalty": 0.5, "repetition_guard": true})
        );
        assert_eq!(
            service_defaults(&json!({"top_p": 0.8})).unwrap(),
            json!({"presence_penalty": 0.5, "top_p": 0.8, "repetition_guard": true})
        );
        assert_eq!(
            service_defaults(&json!({"presence_penalty": 0})).unwrap(),
            json!({"presence_penalty": 0, "repetition_guard": true})
        );
    }

    #[test]
    fn repetition_guard_validates_boolean_and_preserves_request_precedence() {
        for invalid in [json!(0), json!(1), json!("true"), json!([])] {
            assert!(extract(&json!({"repetition_guard": invalid})).is_err());
            assert!(validate_overrides(&json!({"repetition_guard": invalid})).is_err());
        }
        assert!(validate_overrides(&json!({"repetition_guard": null})).is_err());
        let defaults = service_defaults(&json!({"repetition_guard": false})).unwrap();
        assert_eq!(defaults["repetition_guard"], false);
        for (request, expected) in [(json!(null), false), (json!(true), true)] {
            let mut sampling = extract(&json!({"repetition_guard": request})).unwrap();
            apply_defaults(&mut sampling, &defaults);
            assert_eq!(sampling["repetition_guard"], expected);
        }
    }
}
