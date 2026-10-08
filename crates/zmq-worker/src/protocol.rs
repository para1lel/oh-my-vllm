//! msgpack message types shared between Rust and Python.

use serde::{Deserialize, Serialize};

// ── Rust → Python ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RustMessage {
    Init(InitMsg),
    Register(RegisterMsg),
    Prepare {
        rpc_id: u64,
        request_id: u64,
        request: serde_json::Value,
    },
    Execute(ExecuteMsg),
    SetSpeculativeMode {
        rpc_id: u64,
        mode: String,
    },
    Abort(AbortMsg),
    Shutdown,
}

#[derive(Debug, Serialize)]
pub struct InitMsg {
    pub model_path: String,
    pub num_gpu_blocks: u32,
    pub mamba_blocks: Option<u32>,
    pub block_size: u32,
    pub tensor_parallel_size: u32,
    pub max_model_len: u32,
    pub num_speculative_tokens: usize,
    /// Optional explicit mode. An absent mode preserves the legacy 0/4 selection.
    pub speculative_mode: Option<String>,
    /// DSpark checkpoint location, independent of the target model location.
    pub draft_model_path: Option<String>,
    /// Load both proposers for the idle-only, shared-target comparison command.
    pub comparison: bool,
    /// Minimum cumulative prefix confidence; zero retains the full legal draft.
    pub dspark_confidence_threshold: f64,
}

#[derive(Debug, Serialize)]
pub struct RegisterMsg {
    pub request_id: u64,
    pub prompt_token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct ExecuteMsg {
    pub rpc_id: u64,
    pub step_id: u64,
    pub scheduled: Vec<ScheduledRequestMsg>,
    pub finished_request_ids: Vec<u64>,
    pub preempted_request_ids: Vec<u64>,
    pub num_batched_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ScheduledRequestMsg {
    pub request_id: u64,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_token_ids: Option<Vec<u32>>,
    pub num_computed_tokens: u32,
    pub fa_block_table: Vec<u32>,
    pub mamba_block_table: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct AbortMsg {
    pub request_id: u64,
}

// ── Python → Rust ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PythonMessage {
    Ready {
        logical_num_blocks: u32,
        mamba_blocks: u32,
    },
    ExecuteResult(ExecuteResultMsg),
    Prepared {
        rpc_id: u64,
        prompt_token_ids: Vec<u32>,
    },
    ModeChanged {
        rpc_id: u64,
        mode: String,
    },
    Error(ErrorMsg),
}

#[derive(Debug, Deserialize)]
pub struct ExecuteResultMsg {
    pub rpc_id: u64,
    pub outputs: Vec<RequestResultMsg>,
}

#[derive(Debug, Deserialize)]
pub struct RequestResultMsg {
    pub request_id: u64,
    #[serde(default)]
    pub error: Option<String>,
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub num_accepted_draft_tokens: u32,
    #[serde(default)]
    pub new_draft_token_ids: Vec<u32>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct ErrorMsg {
    pub message: String,
    #[serde(default)]
    pub rpc_id: u64,
    #[serde(default)]
    pub kind: WorkerErrorKind,
}

#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerErrorKind {
    Validation,
    #[default]
    Internal,
}

// ── encode / decode helpers ───────────────────────────────────────────────────

pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(msg)
}

pub fn decode(bytes: &[u8]) -> Result<PythonMessage, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_wire_contains_only_live_cache_fields() {
        let message = RustMessage::Execute(ExecuteMsg {
            rpc_id: 1,
            step_id: 2,
            scheduled: vec![ScheduledRequestMsg {
                request_id: 3,
                token_ids: vec![4],
                prefill_token_ids: Some(vec![4]),
                num_computed_tokens: 0,
                fa_block_table: vec![1],
                mamba_block_table: vec![2],
            }],
            finished_request_ids: vec![],
            preempted_request_ids: vec![],
            num_batched_tokens: 1,
        });
        let bytes = encode(&message).unwrap();
        let value: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
        let request = &value["scheduled"][0];
        assert!(request.get("new_block_ids_to_zero").is_none());
        assert_eq!(request["fa_block_table"], serde_json::json!([1]));
        assert_eq!(request["mamba_block_table"], serde_json::json!([2]));
    }

    #[test]
    fn worker_error_kind_defaults_to_internal() {
        let old = rmp_serde::to_vec_named(&serde_json::json!({
            "type": "error",
            "message": "tokenizer failed",
            "rpc_id": 4,
        }))
        .unwrap();
        let PythonMessage::Error(error) = decode(&old).unwrap() else {
            panic!("expected error frame");
        };
        assert_eq!(error.kind, WorkerErrorKind::Internal);

        let validation = rmp_serde::to_vec_named(&serde_json::json!({
            "type": "error",
            "message": "invalid schema",
            "kind": "validation",
            "rpc_id": 5,
        }))
        .unwrap();
        let PythonMessage::Error(error) = decode(&validation).unwrap() else {
            panic!("expected error frame");
        };
        assert_eq!(error.kind, WorkerErrorKind::Validation);
    }
}
