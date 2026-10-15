use std::io::{self, BufRead, Write};

use serde_json::{Value, json};

#[allow(dead_code)]
#[path = "../serving/media.rs"]
mod media;

#[allow(dead_code)]
#[path = "../serving/request.rs"]
mod request;

#[allow(dead_code)]
#[path = "../serving/sampling.rs"]
mod sampling;

fn main() -> anyhow::Result<()> {
    let mut output = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let result = (|| -> anyhow::Result<Value> {
            let input: Value = serde_json::from_str(&line?)?;
            let api = input["api"].as_str().unwrap_or("");
            anyhow::ensure!(["chat", "responses"].contains(&api), "unknown API");
            let model = input["model"].as_str().unwrap_or("qwen3.8-27b-fp8");
            let normalized =
                request::normalize(input["request"].clone(), api == "responses", model, None)?;
            Ok(json!({"normalized": normalized.normalized}))
        })();
        let response = match result {
            Ok(value) => value,
            Err(error) => json!({"error": error.to_string()}),
        };
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
