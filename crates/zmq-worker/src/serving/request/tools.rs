use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

pub(super) fn extract(body: &Value, responses: bool) -> Result<Vec<Value>> {
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
    Ok(tools)
}

pub(super) fn choice(body: &Value, responses: bool, tools: &[Value]) -> Result<Value> {
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
    Ok(choice)
}
