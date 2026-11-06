use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

use super::super::media;

pub(super) fn normalize(
    body: &Value,
    responses: bool,
    previous: Option<Vec<Value>>,
) -> Result<Vec<Value>> {
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
    validate(&mut messages)?;
    Ok(messages)
}

fn validate(messages: &mut [Value]) -> Result<()> {
    let mut outstanding = std::collections::BTreeSet::new();
    let mut media_count = 0;
    for message in messages {
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
            *content = media::content(content, role == "user")?;
            media_count += media::count(content);
            ensure!(media_count <= 16, "at most 16 media items are supported");
        }
        let tool_calls = match message.get("tool_calls") {
            Some(value) => Some(
                value
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("tool_calls must be an array"))?,
            ),
            None => None,
        };
        for call in tool_calls.into_iter().flatten() {
            ensure!(call.is_object(), "tool call must be an object");
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
    Ok(())
}

fn text_content(value: &Value) -> Result<Value> {
    media::content(value, false)
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
            let content = media::content(&item["content"], item["role"] == "user")?;
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
