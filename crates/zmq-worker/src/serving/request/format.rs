use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn extract(body: &Value, responses: bool) -> Result<Value> {
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
            ensure!(schema.is_object(), "json_schema must be an object");
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
    Ok(format)
}
