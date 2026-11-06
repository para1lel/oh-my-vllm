use anyhow::{Result, ensure};
use serde_json::Value;

const ALLOWED: &[&str] = &[
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
    "min_tokens",
    "ignore_eos",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "repetition_guard",
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
    "thinking_token_budget",
];

pub(super) fn fields(body: &Value) -> Result<()> {
    let object = body
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("request must be an object"))?;
    for key in object.keys() {
        ensure!(ALLOWED.contains(&key.as_str()), "unsupported field: {key}");
    }
    for key in [
        "stream",
        "store",
        "parallel_tool_calls",
        "background",
        "ignore_eos",
    ] {
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
    Ok(())
}
