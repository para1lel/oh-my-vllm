use std::io::{self, BufRead, Write};
use std::path::Path;

use oh_my_vllm_zmq_worker::models::select_model;
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
    let default_model = select_model(Path::new("stub"))?.served_model_name();
    let mut output = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let result = (|| -> anyhow::Result<Value> {
            let input: Value = serde_json::from_str(&line?)?;
            let api = input["api"].as_str().unwrap_or("");
            anyhow::ensure!(["chat", "responses"].contains(&api), "unknown API");
            let model = input["model"].as_str().unwrap_or(default_model);
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
