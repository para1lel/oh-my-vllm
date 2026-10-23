use oh_my_vllm_model_contract::ModelDescriptor;
use serde::{Deserialize, Serialize};

use crate::group::protocol::{RankHeader, RankReady, RankResult, StepDispatch};

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command<'a> {
    Init {
        header: RankHeader,
        descriptor: &'a ModelDescriptor,
        device_uuids: &'a [String],
    },
    Execute(&'a StepDispatch),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "Ready supplies the descriptor only at startup"
)]
pub enum Reply {
    Ready(RankReady),
    Result(RankResult),
    Error { header: RankHeader, message: String },
}
