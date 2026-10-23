use oh_my_vllm_model_contract::ModelDescriptor;
use serde::{Deserialize, Serialize};

use crate::group::protocol::{RankHeader, RankReady, RankResult, StepDispatch};

use super::GroupTransport;

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command<'a> {
    Init {
        header: RankHeader,
        descriptor: &'a ModelDescriptor,
        device_uuids: &'a [String],
        transport: GroupTransport,
    },
    Connect {
        header: RankHeader,
        unique_id: &'a [u8],
        bootstrap_nonce: &'a str,
    },
    Execute(&'a StepDispatch),
    Shutdown {
        header: RankHeader,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankBootstrap {
    pub header: RankHeader,
    pub device_uuid: String,
    pub unique_id: Option<Vec<u8>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "Ready supplies the descriptor only at startup"
)]
pub enum Reply {
    Bootstrap(RankBootstrap),
    Ready(RankReady),
    Result(RankResult),
    Closed { header: RankHeader },
    Error { header: RankHeader, message: String },
}
