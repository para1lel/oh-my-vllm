use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

pub(super) fn tokens(body: &Value) -> Result<(u64, u64, Value)> {
    let limits: Vec<_> = ["max_tokens", "max_completion_tokens", "max_output_tokens"]
        .iter()
        .filter_map(|k| body.get(*k))
        .collect();
    ensure!(limits.len() <= 1, "specify only one output token limit");
    let max_tokens = limits.first().map_or(Ok(8192), |v| {
        v.as_u64()
            .ok_or_else(|| anyhow::anyhow!("output limit must be an integer"))
    })?;
    ensure!(
        max_tokens > 0 && max_tokens <= 1_048_576,
        "invalid output limit"
    );
    let min_tokens = body.get("min_tokens").map_or(Ok(0), |value| {
        value
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("min_tokens must be an integer"))
    })?;
    ensure!(
        min_tokens <= max_tokens,
        "min_tokens exceeds the output limit"
    );
    let thinking_budget = match body.get("thinking_token_budget") {
        None => Value::Null,
        Some(value) => {
            let budget = value
                .as_i64()
                .ok_or_else(|| anyhow::anyhow!("thinking_token_budget must be an integer"))?;
            ensure!(
                (-1..=1_048_576).contains(&budget),
                "thinking_token_budget must be -1 or in [0,1048576]"
            );
            json!(budget)
        }
    };
    Ok((max_tokens, min_tokens, thinking_budget))
}

pub(super) fn stop(body: &Value) -> Result<Vec<String>> {
    let stop = match body.get("stop") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("stop must contain strings"))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("invalid stop"),
    };
    ensure!(
        stop.len() <= 4 && stop.iter().all(|s| !s.is_empty() && s.len() <= 4096),
        "invalid stop strings"
    );
    Ok(stop)
}
