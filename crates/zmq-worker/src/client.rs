use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use zeromq::DealerSendHalf;

use crate::error::{Error, Result};
use crate::models::{ModelFamily, ModelPolicy};
use crate::protocol::ModelDescriptor;

mod execution;
pub(crate) mod lifecycle;
mod preparation;
mod shutdown;
mod startup;
mod transport;

use lifecycle::{Binding, ManagedChild};

const WORKER_CLEANUP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub model_path: PathBuf,
    pub socket_path: PathBuf,
    pub num_gpu_blocks: Option<u32>,
    pub mamba_blocks: Option<u32>,
    pub worker_epoch: u64,
    pub block_size: u32,
    pub tensor_parallel_size: u32,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub max_num_seqs: u32,
    pub num_speculative_tokens: usize,
    pub speculative_mode: Option<String>,
    pub python_executable: PathBuf,
    pub init_timeout: Duration,
    pub torch_profile: Option<crate::protocol::BenchmarkProfile>,
}

impl WorkerConfig {
    pub fn for_model(model_path: PathBuf, model: ModelPolicy) -> Self {
        let defaults = model.defaults();
        Self {
            model_path,
            socket_path: PathBuf::from("/tmp/oh-my-vllm.ipc"),
            num_gpu_blocks: None,
            mamba_blocks: Some(defaults.mamba_blocks),
            worker_epoch: 1,
            block_size: defaults.block_size,
            tensor_parallel_size: defaults.tensor_parallel_size,
            max_model_len: defaults.max_model_len,
            max_num_batched_tokens: defaults.max_num_batched_tokens,
            max_num_seqs: defaults.max_num_seqs,
            num_speculative_tokens: 0,
            speculative_mode: None,
            python_executable: std::env::var_os("OH_MY_VLLM_WORKER_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("python3")),
            init_timeout: Duration::from_secs(defaults.init_timeout_seconds),
            torch_profile: None,
        }
    }
}

pub struct WorkerClient {
    config: WorkerConfig,
    descriptor: ModelDescriptor,
    model_family: ModelFamily,
    execution: crate::models::execution::ModelExecution,
    sock: DealerSendHalf,
    inbox: mpsc::Receiver<Result<crate::protocol::WorkerMessage>>,
    reader: JoinHandle<()>,
    _child: ManagedChild,
    step_id: u64,
    execution_trace: Option<serde_json::Value>,
    preparation_trace: preparation::trace::PreparationTrace,
    rpc_id: u64,
    pending_prepare: Option<(u64, u64)>,
    last_prepare: Option<(u64, u64)>,
    cancelled_prepares: BTreeSet<u64>,
    prepared_reply: Option<crate::protocol::WorkerMessage>,
    serving_enabled: bool,
    verified_reasoning_tokens: std::collections::BTreeMap<u64, usize>,
    pub logical_num_blocks: u32,
    pub mamba_blocks: u32,
    pub cache_layout: crate::protocol::CacheLayout,
    pub serving_outputs: std::collections::BTreeMap<u64, (String, Option<String>, usize)>,
    pub serving_errors: std::collections::BTreeMap<u64, String>,
    binding: Binding,
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl WorkerClient {
    pub(crate) fn take_prepare_traces(&mut self) -> Vec<serde_json::Value> {
        self.preparation_trace.take()
    }

    pub(crate) fn take_execution_trace(&mut self) -> Option<serde_json::Value> {
        self.execution_trace.take()
    }

    pub fn configuration(&self) -> WorkerConfig {
        self.config.clone()
    }

    pub fn model_descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    pub(crate) fn model_family(&self) -> ModelFamily {
        self.model_family
    }

    pub fn dep4_cache(
        &self,
        template: oh_my_vllm_kv_cache::HybridCoordinator,
    ) -> Option<crate::models::Dep4Cache> {
        self.execution.owners.as_ref().map(|owners| {
            crate::models::Dep4Cache::new(
                template,
                owners.clone(),
                self.configuration().max_num_seqs as usize,
            )
        })
    }

    pub(crate) fn attach_request_owners(
        &mut self,
        owners: std::sync::Arc<
            std::sync::Mutex<crate::models::deepseek_v41::ownership::RequestOwners>,
        >,
    ) {
        self.execution.owners = Some(owners);
    }

    pub fn bind_request_cache(&mut self, request: &mut oh_my_vllm_scheduler::Request) {
        self.execution.bind_cache(request);
    }

    pub fn model_policy(&self) -> ModelPolicy {
        ModelPolicy::for_family(self.model_family)
    }
}

const MAX_READY_MESSAGE_BYTES: usize = 1 << 20;

fn worker_error(error: crate::protocol::ErrorMsg) -> Error {
    use crate::protocol::WorkerErrorKind;
    match error.kind {
        WorkerErrorKind::Validation => Error::WorkerValidation(error.message),
        WorkerErrorKind::Internal => Error::WorkerError(error.message),
        WorkerErrorKind::Oom => Error::WorkerOom(error.message),
        WorkerErrorKind::Device => Error::WorkerDevice(error.message),
    }
}

#[cfg(all(test, unix))]
#[path = "test_worker.rs"]
pub(crate) mod fixture;
