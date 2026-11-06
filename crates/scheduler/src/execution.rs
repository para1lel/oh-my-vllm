use oh_my_vllm_kv_cache::group::RequestId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionKey {
    pub worker_epoch: u64,
    pub step_id: u64,
    pub request_id: RequestId,
    pub lease_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkPurpose {
    Advance,
    Finalize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkAction {
    Forward { start: usize, end: usize },
    HeadOnly { endpoint: usize },
    ExportOnly { endpoint: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    None,
    Target,
    Verify { drafts: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkSpec {
    pub purpose: WorkPurpose,
    pub action: WorkAction,
    pub sampling: SamplingMode,
}

impl WorkSpec {
    pub fn validate(self) -> Result<(), &'static str> {
        if let WorkAction::Forward { start, end } = self.action
            && start >= end
        {
            return Err("forward range must contain rows");
        }
        if self.purpose == WorkPurpose::Finalize && self.sampling != SamplingMode::None {
            return Err("finalize cannot sample");
        }
        if let SamplingMode::Verify { drafts } = self.sampling {
            let (start, end) = self.range();
            if drafts == 0 || drafts >= end.saturating_sub(start) {
                return Err("verify requires forwarded target and draft rows");
            }
        }
        match (self.purpose, self.action, self.sampling) {
            (WorkPurpose::Finalize, WorkAction::ExportOnly { .. }, SamplingMode::None)
            | (WorkPurpose::Advance, WorkAction::HeadOnly { .. }, SamplingMode::Target)
            | (_, WorkAction::Forward { .. }, _) => Ok(()),
            _ => Err("invalid purpose, action and sampling combination"),
        }
    }

    pub fn range(self) -> (usize, usize) {
        match self.action {
            WorkAction::Forward { start, end } => (start, end),
            WorkAction::HeadOnly { endpoint } | WorkAction::ExportOnly { endpoint } => {
                (endpoint, endpoint)
            }
        }
    }

    pub fn expected_tokens(self, accepted_drafts: usize) -> Option<usize> {
        match self.sampling {
            SamplingMode::None => (accepted_drafts == 0).then_some(0),
            SamplingMode::Target => (accepted_drafts == 0).then_some(1),
            SamplingMode::Verify { drafts } => (accepted_drafts <= drafts)
                .then(|| accepted_drafts.checked_add(1))
                .flatten(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStepResult {
    pub execution: ExecutionKey,
    pub computed_end: usize,
    pub accepted_end: usize,
    pub exported_checkpoints: Vec<(usize, u32, u64)>,
    pub page_valid_ends: Vec<(u32, u64, usize)>,
    pub mtp_accepted_end: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionPlan {
    ForwardAccepted { start: usize, end: usize },
    ExportEndpoint { endpoint: usize },
    Ready { endpoint: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishOutcome {
    Succeeded { publish_tail: bool },
    Cancelled,
    Failed,
}
