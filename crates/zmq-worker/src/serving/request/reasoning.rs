use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

pub(super) fn template(body: &Value, responses: bool) -> Result<Value> {
    for value in [
        body.get("reasoning_effort"),
        body.get("reasoning").and_then(|v| v.get("effort")),
    ]
    .into_iter()
    .flatten()
    {
        ensure!(value.is_string(), "reasoning effort must be a string");
    }
    if let Some(text) = body.get("text") {
        ensure!(
            responses && text.is_object(),
            "text must be a Responses object"
        );
    }
    let template = body
        .get("chat_template_kwargs")
        .cloned()
        .unwrap_or(json!({}));
    ensure!(
        template.is_object(),
        "chat_template_kwargs must be an object"
    );
    for (key, value) in template.as_object().unwrap() {
        ensure!(
            ["preserve_thinking", "enable_thinking", "reasoning_effort"].contains(&key.as_str()),
            "unsupported template option: {key}"
        );
        ensure!(
            if key == "reasoning_effort" {
                value.is_string()
            } else {
                value.is_boolean()
            },
            "invalid template option type: {key}"
        );
    }
    if let Some(value) = body.get("enable_thinking") {
        ensure!(value.is_boolean(), "enable_thinking must be boolean");
    }
    Ok(template)
}

pub(super) fn effort(body: &Value, template: &Value, responses: bool) -> Result<String> {
    let effort = if responses {
        body["reasoning"]["effort"].as_str()
    } else {
        body["reasoning_effort"].as_str()
    }
    .or_else(|| template["reasoning_effort"].as_str())
    .unwrap_or("medium");
    let effort = match effort {
        "off" | "low" | "medium" | "xhigh" => effort,
        "high" => "xhigh",
        "none" => "off",
        _ => bail!("unsupported reasoning effort: {effort}"),
    }
    .to_owned();
    let effort = if body
        .get("enable_thinking")
        .or_else(|| template.get("enable_thinking"))
        == Some(&json!(false))
    {
        "off".to_owned()
    } else {
        effort
    };
    Ok(effort)
}

pub(super) fn options(body: &Value, responses: bool) -> Result<()> {
    if let Some(reasoning) = body.get("reasoning") {
        ensure!(
            responses && reasoning.is_object(),
            "reasoning must be a Responses object"
        );
        for (key, value) in reasoning.as_object().unwrap() {
            ensure!(
                key == "effort" || key == "summary",
                "unsupported reasoning field: {key}"
            );
            if key == "summary" {
                ensure!(
                    value.is_null() || ["auto", "concise", "detailed"].iter().any(|s| value == s),
                    "invalid reasoning summary"
                );
            }
        }
    }
    if let Some(options) = body.get("stream_options").filter(|v| !v.is_null()) {
        let options = options
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("stream_options must be an object"))?;
        for (key, value) in options {
            ensure!(
                (key == "include_usage" && !responses && value.is_boolean())
                    || (key == "include_obfuscation" && value == false),
                "unsupported stream option: {key}"
            );
        }
    }
    Ok(())
}
