use crate::protocol::WorkerMessage;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Default)]
pub(in crate::client) struct PreparationTrace {
    pending: Option<Value>,
    ready: Vec<Value>,
}

impl PreparationTrace {
    pub(in crate::client) fn begin(&mut self, epoch: u64, rpc: u64, id: u64, request: &Value) {
        self.pending = None;
        if std::env::var_os("OH_MY_VLLM_RUST_EXECUTION_TRACE").is_none() {
            return;
        }
        self.pending = Some(intent(epoch, rpc, id, request));
    }

    pub(in crate::client) fn finish(&mut self, pending: Option<(u64, u64)>, reply: &WorkerMessage) {
        let Some(mut fact) = self.pending.take() else {
            return;
        };
        let WorkerMessage::Prepared {
            rpc_id,
            prompt_token_ids,
        } = reply
        else {
            return;
        };
        if pending != Some((fact["request_id"].as_u64().unwrap(), *rpc_id))
            || fact["rpc_id"].as_u64() != Some(*rpc_id)
        {
            return;
        }
        let Ok(tokens) = serde_json::to_vec(prompt_token_ids) else {
            return;
        };
        fact["operation"] = json!("prepare_validated");
        fact["prompt_token_sha256"] = json!(format!("{:x}", Sha256::digest(tokens)));
        self.ready.push(fact);
    }

    pub(in crate::client) fn cancel(&mut self, id: u64) {
        if self
            .pending
            .as_ref()
            .is_some_and(|fact| fact["request_id"] == id)
        {
            self.pending = None;
        }
    }

    pub(in crate::client) fn take(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.ready)
    }
}

fn intent(epoch: u64, rpc: u64, id: u64, request: &Value) -> Value {
    let mut bits = serde_json::Map::new();
    for name in [
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ] {
        if let Some(value) = request["sampling"][name].as_f64() {
            bits.insert(
                name.to_owned(),
                json!(format!("{:08x}", (value as f32).to_bits())),
            );
        }
    }
    json!({"worker_epoch": epoch, "rpc_id": rpc, "request_id": id,
        "normalized_sampling_f32_bits": bits,
        "seed": request["sampling"]["seed"],
        "repetition_guard": request["sampling"]["repetition_guard"],
        "min_tokens": request["min_tokens"], "max_tokens": request["max_tokens"]})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_matching_successful_prepare_publishes_observation() {
        let request = json!({"sampling": {"temperature": 0.95, "seed": 0,
            "repetition_guard": true}, "min_tokens": 64, "max_tokens": 64});
        let original = intent(1, 9, 7, &request);
        let mut trace = PreparationTrace {
            pending: Some(original.clone()),
            ..Default::default()
        };
        let reply = WorkerMessage::Prepared {
            rpc_id: 9,
            prompt_token_ids: vec![10, 11],
        };
        trace.finish(Some((7, 9)), &reply);
        let facts = trace.take();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0]["operation"], "prepare_validated");
        assert_eq!(
            facts[0]["normalized_sampling_f32_bits"]["temperature"],
            "3f733333"
        );
        assert!(
            facts[0]["normalized_sampling_f32_bits"]
                .get("top_p")
                .is_none()
        );
        assert_eq!(
            facts[0]["prompt_token_sha256"],
            format!("{:x}", Sha256::digest(b"[10,11]"))
        );
        trace.pending = Some(original.clone());
        trace.finish(Some((8, 9)), &reply);
        assert!(trace.take().is_empty());
        trace.pending = Some(original);
        trace.cancel(7);
        trace.finish(Some((7, 9)), &reply);
        assert!(trace.take().is_empty());
    }
}
