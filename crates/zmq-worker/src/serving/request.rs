#[path = "request/format.rs"]
mod format;
#[path = "request/limits.rs"]
mod limits;
#[path = "request/messages.rs"]
mod messages;
#[path = "request/reasoning.rs"]
mod reasoning;
#[path = "request/tools.rs"]
mod tools;
#[path = "request/validation.rs"]
mod validation;

use anyhow::{Result, ensure};
use serde_json::{Value, json};

#[derive(Clone)]
pub struct Request {
    pub normalized: Value,
    pub responses: bool,
    pub stream: bool,
    pub store: bool,
    pub include_usage: bool,
    pub model: String,
    pub original: Value,
    pub(super) identity: super::identity::Identity,
}

pub fn normalize(
    mut body: Value,
    responses: bool,
    model: &str,
    previous: Option<Vec<Value>>,
) -> Result<Request> {
    let normalized = normalize_body(&body, responses, model, previous)?;
    if body.get("metadata").is_none() {
        body["metadata"] = json!({});
    }
    let identity = super::identity::resolve("", &body, &Default::default(), None, responses)?;
    Ok(Request {
        identity,
        responses,
        stream: body["stream"].as_bool().unwrap_or(false),
        store: responses && body["store"].as_bool().unwrap_or(true),
        include_usage: body["stream_options"]["include_usage"]
            .as_bool()
            .unwrap_or(false),
        model: model.to_owned(),
        normalized,
        original: body,
    })
}

fn normalize_body(
    body: &Value,
    responses: bool,
    model: &str,
    previous: Option<Vec<Value>>,
) -> Result<Value> {
    validation::fields(body)?;
    let requested_model = body["model"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("model is required"))?;
    ensure!(requested_model == model, "unknown model: {requested_model}");
    let template = reasoning::template(body, responses)?;
    let effort = reasoning::effort(body, &template, responses)?;
    reasoning::options(body, responses)?;
    let messages = messages::normalize(body, responses, previous)?;
    let tools = tools::extract(body, responses)?;
    let choice = tools::choice(body, responses, &tools)?;
    let format = format::extract(body, responses)?;
    ensure!(
        format["type"] == "text" || tools.is_empty() || choice == "none",
        "structured response and enabled tools cannot be combined"
    );
    let (max_tokens, min_tokens, thinking_budget) = limits::tokens(body)?;
    let sampling = super::sampling::extract(body)?;
    let stop = limits::stop(body)?;
    ensure!(
        stop.is_empty() || (format["type"] == "text" && (tools.is_empty() || choice == "none")),
        "stop strings cannot interrupt structured output or tool calls"
    );
    let normalized = json!({
        "messages": messages,
        "tools": tools,
        "tool_choice": choice,
        "parallel_tool_calls": body.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
        "effort": effort,
        "preserve_thinking": template.get("preserve_thinking").cloned().unwrap_or(json!(true)),
        "format": format,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "ignore_eos": body.get("ignore_eos").cloned().unwrap_or(json!(false)),
        "thinking_token_budget": thinking_budget,
        "sampling": sampling,
        "stop": stop,
    });
    Ok(normalized)
}
