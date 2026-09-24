use anyhow::{Result, bail, ensure};
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
}

pub fn normalize(
    mut body: Value,
    responses: bool,
    model: &str,
    previous: Option<Vec<Value>>,
) -> Result<Request> {
    let object = body
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("request must be an object"))?;
    let allowed = [
        "model",
        "messages",
        "input",
        "instructions",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "reasoning",
        "response_format",
        "text",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "stop",
        "store",
        "previous_response_id",
        "metadata",
        "user",
        "service_tier",
        "truncation",
        "include",
        "n",
        "logprobs",
        "top_logprobs",
        "background",
        "prompt_cache_key",
        "safety_identifier",
        "chat_template_kwargs",
        "enable_thinking",
    ];
    for key in object.keys() {
        ensure!(allowed.contains(&key.as_str()), "unsupported field: {key}");
    }
    for key in ["stream", "store", "parallel_tool_calls", "background"] {
        if let Some(v) = body.get(key) {
            ensure!(v.is_boolean(), "{key} must be a boolean");
        }
    }
    ensure!(
        body.get("n").is_none_or(|v| v == 1),
        "only n=1 is supported"
    );
    ensure!(
        body.get("logprobs")
            .is_none_or(|v| v == false || v.is_null()),
        "logprobs is unsupported"
    );
    ensure!(
        body.get("top_logprobs")
            .is_none_or(|v| v == 0 || v.is_null()),
        "top_logprobs is unsupported"
    );
    ensure!(
        body.get("background").is_none_or(|v| v == false),
        "background is unsupported"
    );
    ensure!(
        body.get("truncation").is_none_or(|v| v == "disabled"),
        "automatic truncation is unsupported"
    );
    ensure!(
        body.get("service_tier")
            .is_none_or(|v| v == "auto" || v == "default"),
        "unsupported service_tier"
    );
    let requested_model = body["model"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("model is required"))?;
    ensure!(requested_model == model, "unknown model: {requested_model}");
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
    let mut messages = previous.unwrap_or_default();
    if responses {
        if let Some(instructions) = body.get("instructions").filter(|v| !v.is_null()) {
            ensure!(instructions.is_string(), "instructions must be a string");
            messages.retain(|m| m["role"] != "system");
            messages.insert(
                0,
                json!({
                    "role": "system",
                    "content": instructions,
                }),
            );
        }
        match body.get("input") {
            Some(Value::String(text)) => messages.push(json!({
                "role": "user",
                "content": text,
            })),
            Some(Value::Array(items)) => {
                for item in items {
                    append_input(&mut messages, item)?;
                }
            }
            _ => bail!("input must be text or an array"),
        }
    } else {
        ensure!(
            messages.is_empty(),
            "stored continuation requires Responses"
        );
        messages = body["messages"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("messages must be an array"))?
            .clone();
    }
    ensure!(!messages.is_empty(), "messages cannot be empty");
    let mut outstanding = std::collections::BTreeSet::new();
    for message in &mut messages {
        let role = message["role"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("message role is required"))?
            .to_owned();
        ensure!(
            ["system", "developer", "user", "assistant", "tool"].contains(&role.as_str()),
            "unsupported message role"
        );
        if role == "developer" {
            message["role"] = json!("system");
        }
        if let Some(content) = message.get_mut("content") {
            *content = text_content(content)?;
        }
        for call in message["tool_calls"].as_array().into_iter().flatten() {
            let id = call["id"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tool call id is required"))?;
            ensure!(outstanding.insert(id.to_owned()), "duplicate tool call id");
            ensure!(
                call["function"]["name"].is_string(),
                "function name is required"
            );
            let args = call["function"]["arguments"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("function arguments must be a JSON string"))?;
            ensure!(
                serde_json::from_str::<Value>(args)?.is_object(),
                "function arguments must encode an object"
            );
        }
        if role == "tool" {
            let id = message["tool_call_id"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tool result call id is required"))?;
            ensure!(
                outstanding.remove(id),
                "tool result has no matching outstanding call: {id}"
            );
        }
    }
    let mut tools = Vec::new();
    if let Some(raw) = body.get("tools") {
        for tool in raw
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tools must be an array"))?
        {
            ensure!(
                tool["type"] == "function",
                "only function tools are supported"
            );
            let function = if responses {
                let mut f = tool.clone();
                f.as_object_mut().unwrap().remove("type");
                f
            } else {
                tool["function"].clone()
            };
            if let Some(strict) = function.get("strict") {
                ensure!(strict.is_boolean(), "tool strict must be boolean");
            }
            let name = function["name"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tool name is required"))?;
            ensure!(
                !name.is_empty()
                    && name.len() <= 64
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
                "invalid tool name"
            );
            ensure!(
                !tools.iter().any(|t: &Value| t["function"]["name"] == name),
                "duplicate tool name"
            );
            tools.push(json!({
                "type": "function",
                "function": function,
            }));
        }
    }
    let mut choice = body
        .get("tool_choice")
        .cloned()
        .unwrap_or(json!(if tools.is_empty() {
            "none"
        } else {
            "auto"
        }));
    if responses && choice.is_object() && choice["type"] == "function" {
        choice = json!({
            "type": "function",
            "function": {
                "name": choice["name"],
            },
        });
    }
    match &choice {
        Value::String(s) if ["auto", "none", "required"].contains(&s.as_str()) => {
            ensure!(s != "required" || !tools.is_empty(), "required needs tools");
        }
        Value::Object(_) => {
            ensure!(
                choice["type"] == "function"
                    && tools
                        .iter()
                        .any(|t| t["function"]["name"] == choice["function"]["name"]),
                "unknown forced tool"
            );
        }
        _ => bail!("unsupported tool_choice"),
    }
    let mut format = json!({
        "type": "text",
    });
    if responses {
        if let Some(text) = body.get("text") {
            if let Some(f) = text.get("format") {
                format = f.clone();
            }
            ensure!(
                text.get("verbosity").is_none(),
                "text.verbosity is unsupported"
            );
        }
    } else if let Some(f) = body.get("response_format") {
        format = if f["type"] == "json_schema" {
            let mut schema = f["json_schema"].clone();
            schema["type"] = json!("json_schema");
            schema
        } else {
            f.clone()
        };
    }
    ensure!(
        ["text", "json_object", "json_schema"]
            .iter()
            .any(|t| format["type"] == *t),
        "unsupported output format"
    );
    if let Some(strict) = format.get("strict") {
        ensure!(strict.is_boolean(), "format strict must be boolean");
    }
    if format["type"] == "json_schema" {
        ensure!(format["schema"].is_object(), "json_schema requires schema");
    }
    ensure!(
        format["type"] == "text" || tools.is_empty() || choice == "none",
        "structured response and enabled tools cannot be combined"
    );
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
    let mut sampling = json!({});
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ] {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            ensure!(value.is_number(), "{key} must be numeric");
            sampling[key] = value.clone();
        }
    }
    // A temperature in (0, 1e-5) causes NaN in softmax; treat it as greedy.
    if sampling["temperature"]
        .as_f64()
        .is_some_and(|t| t > 0.0 && t < 1e-5)
    {
        sampling["temperature"] = json!(0.0);
    }
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
        "sampling": sampling,
        "stop": stop,
    });
    // Metadata is retained in the response but never used as a model instruction.
    if body.get("metadata").is_none() {
        body["metadata"] = json!({});
    }
    Ok(Request {
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

fn text_content(value: &Value) -> Result<Value> {
    match value {
        Value::Null | Value::String(_) => Ok(value.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                ensure!(
                    ["text", "input_text", "output_text"]
                        .iter()
                        .any(|t| part["type"] == *t),
                    "only text content is supported"
                );
                text.push_str(
                    part["text"]
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("text part requires text"))?,
                );
            }
            Ok(json!(text))
        }
        _ => bail!("invalid text content"),
    }
}

fn assistant_item(messages: &mut Vec<Value>) -> &mut Value {
    if messages.last().is_none_or(|m| m["role"] != "assistant") {
        messages.push(json!({
            "role": "assistant",
            "content": "",
        }));
    }
    messages.last_mut().unwrap()
}

fn append_input(messages: &mut Vec<Value>, item: &Value) -> Result<()> {
    match item["type"].as_str().unwrap_or("message") {
        "message" => {
            let content = text_content(&item["content"])?;
            if item["role"] == "assistant" {
                let message = assistant_item(messages);
                let mut text = message["content"].as_str().unwrap_or_default().to_owned();
                text.push_str(content.as_str().unwrap_or_default());
                message["content"] = json!(text);
            } else {
                messages.push(json!({
                    "role": item["role"],
                    "content": content,
                }));
            }
        }
        "function_call" => {
            let call = json!({
                "id": item["call_id"],
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            });
            let message = assistant_item(messages);
            if message.get("tool_calls").is_none() {
                message["tool_calls"] = json!([]);
            }
            message["tool_calls"].as_array_mut().unwrap().push(call);
        }
        "function_call_output" => messages.push(json!({
            "role": "tool",
            "tool_call_id": item["call_id"],
            "content": text_content(&item["output"])?,
        })),
        "reasoning" => {
            let mut text = String::new();
            for part in item["summary"].as_array().into_iter().flatten() {
                ensure!(
                    part["type"] == "summary_text" && part["text"].is_string(),
                    "invalid reasoning summary"
                );
                text.push_str(part["text"].as_str().unwrap());
            }
            assistant_item(messages)["reasoning_content"] = json!(text);
        }
        _ => bail!("unsupported Responses input item"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_tool_association_and_mapping() {
        let r = normalize(
            json!({
                "model": "m",
                "reasoning": {
                    "effort": "high",
                },
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "c",
                        "name": "read",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "c",
                        "output": "file",
                    },
                ],
            }),
            true,
            "m",
            None,
        )
        .unwrap();
        assert_eq!(r.normalized["effort"], "xhigh");
        assert_eq!(r.normalized["messages"][1]["tool_call_id"], "c");
        assert!(
            normalize(
                json!({
                    "model": "m",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "missing",
                            "output": "x",
                        },
                    ],
                }),
                true,
                "m",
                None
            )
            .is_err()
        );
    }
}
