use oh_my_vllm_kv_cache::state::{StateBlock, TransactionKey};
use oh_my_vllm_scheduler::RequestOutput;
use serde::{Deserialize, Serialize};

use super::protocol::{RankResult, StepDispatch};

pub const MAX_MODEL_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelState<L> {
    pub transaction: TransactionKey,
    pub reads: Vec<StateBlock>,
    pub writes: Vec<StateBlock>,
    pub layout: L,
}

#[derive(Serialize)]
pub(crate) struct ModelInput<'a, L> {
    pub request_id: u64,
    pub token_ids: &'a [u32],
    pub num_computed_tokens: usize,
    pub state: &'a ModelState<L>,
}

#[derive(Serialize)]
pub(crate) struct ModelPayload<'a, L> {
    pub requests: Vec<ModelInput<'a, L>>,
    pub finished_request_ids: &'a [u64],
    pub preempted_request_ids: &'a [u64],
}

#[derive(Serialize)]
pub(crate) struct ModelCommand<'a> {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub dispatch: &'a StepDispatch,
    pub payload_sha256: &'a str,
    #[serde(serialize_with = "binary_payload")]
    pub payload: &'a [u8],
}

fn binary_payload<S: serde::Serializer>(payload: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(payload)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelRankResult {
    pub completion: RankResult,
    pub payload_sha256: String,
    pub outputs: Vec<RequestOutput>,
}
