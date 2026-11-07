use oh_my_vllm_scheduler::{execution::ExecutionKey, output::RequestOutput};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Write};
use tracing::warn;

#[derive(Default)]
pub(super) struct ExecutionTrace {
    output: Option<File>,
    histories: BTreeMap<u64, (ExecutionKey, Vec<u32>)>,
    accepted: BTreeMap<u64, usize>,
}

impl ExecutionTrace {
    pub(super) fn open(client: &crate::client::WorkerClient) -> Self {
        let mut trace = Self::default();
        let Some(path) = std::env::var_os("OH_MY_VLLM_RUST_EXECUTION_TRACE") else {
            return trace;
        };
        let opened = (|| -> anyhow::Result<File> {
            let path = std::path::PathBuf::from(path);
            let parent = path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .canonicalize()?;
            anyhow::ensure!(
                !parent.starts_with(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../..")
                        .canonicalize()?
                ),
                "execution trace must be outside the repository"
            );
            Ok(File::options().create(true).append(true).open(path)?)
        })();
        match opened {
            Ok(output) => trace.output = Some(output),
            Err(error) => warn!(%error, "Rust execution trace unavailable"),
        }
        let identity = (|| -> anyhow::Result<Value> {
            let manifest = std::fs::read(
                std::env::var_os("OH_MY_VLLM_RUN_MANIFEST")
                    .ok_or_else(|| anyhow::anyhow!("missing run manifest"))?,
            )?;
            let binary = std::fs::read("/proc/self/exe")?;
            Ok(json!({"operation": "rust_identity",
                "worker_epoch": client.configuration().worker_epoch,
                "binary": format!("{:x}", Sha256::digest(binary)),
                "rust_sources_sha256": env!("OH_MY_VLLM_RUST_SOURCES_SHA256"),
                "artifact_id": format!("{:x}", Sha256::digest(manifest)),
                "cache_layout": client.cache_layout}))
        })();
        match identity {
            Ok(identity) => trace.record(identity),
            Err(error) => {
                warn!(%error, "Rust execution identity unavailable");
                trace.output = None;
            }
        }
        trace
    }

    fn record(&mut self, event: Value) {
        let Some(output) = self.output.as_mut() else {
            return;
        };
        let written = (|| -> anyhow::Result<()> {
            serde_json::to_writer(&mut *output, &event)?;
            output.write_all(b"\n")?;
            output.flush()?;
            Ok(())
        })();
        if let Err(error) = written {
            warn!(%error, "Rust execution trace write failed");
            self.output = None;
        }
    }

    pub(super) fn validated<S>(
        &mut self,
        scheduled: &[oh_my_vllm_scheduler::ScheduledRequest<S>],
        outputs: &[RequestOutput],
        wire: Option<Value>,
    ) {
        if self.output.is_none() || outputs.is_empty() {
            return;
        }
        for row in scheduled {
            if let Some(tokens) = &row.prefill_token_ids {
                self.histories
                    .insert(row.request_id, (row.execution, tokens.clone()));
            }
        }
        for row in outputs {
            if let Some((execution, tokens)) = self.histories.get_mut(&row.request_id) {
                *execution = row.execution;
                tokens.extend(&row.token_ids);
            }
        }
        let executions: Vec<_> = outputs.iter().map(|row| row.execution).collect();
        self.record(json!({"operation": "validated", "wire": wire, "executions": executions}));
    }

    pub(super) fn prepared(&mut self, event: Value) {
        self.record(event);
    }

    pub(super) fn accepted(&mut self, id: u64, end: usize) {
        if self.output.is_none() {
            return;
        }
        self.accepted.insert(id, end);
        if let Some((_, tokens)) = self.histories.get_mut(&id) {
            tokens.truncate(end);
        }
        self.record(
            json!({"operation": "accepted", "requests": [{"id": id, "accepted_end": end}]}),
        );
    }

    pub(super) fn retain(&mut self, ids: impl Iterator<Item = u64>) {
        let ids: std::collections::BTreeSet<_> = ids.collect();
        self.histories.retain(|id, _| ids.contains(id));
        self.accepted.retain(|id, _| ids.contains(id));
    }

    pub(super) fn complete(&mut self, id: u64, response: &Value) {
        if self.output.is_none() {
            return;
        }
        let Some((execution, tokens)) = self.histories.remove(&id) else {
            return;
        };
        let Some(end) = self.accepted.remove(&id) else {
            return;
        };
        if tokens.len() != end {
            return;
        }
        let Ok(encoded) = serde_json::to_vec(&tokens) else {
            return;
        };
        self.record(json!({"operation": "complete", "requests": [{"id": id,
            "status": "completed", "accepted_end": end, "execution": execution,
            "token_sha256": format!("{:x}", Sha256::digest(encoded)), "response": response}]}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording(name: &str) -> (ExecutionTrace, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "oh-my-vllm-trace-{}-{name}.jsonl",
            std::process::id(),
        ));
        let output = File::options()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        let execution = ExecutionKey {
            worker_epoch: 1,
            step_id: 3,
            request_id: 7,
            lease_id: 2,
        };
        let mut trace = ExecutionTrace {
            output: Some(output),
            ..Default::default()
        };
        trace.histories.insert(7, (execution, vec![10, 11, 12, 99]));
        (trace, path)
    }

    #[test]
    fn complete_requires_parser_acceptance_and_hashes_the_retained_tokens() {
        let (mut trace, path) = recording("accepted");
        let response = json!({"id": "chatcmpl_trace_7", "choices": [{"message": {
            "role": "assistant", "content": "已准备", "tool_calls": [{"function": {
                "name": "lookup_reference", "arguments": "{\"topic\":\"trace\"}"}}]}}],
            "usage": {"completion_tokens": 1}});
        trace.accepted(7, 3);
        trace.complete(7, &response);
        drop(trace);
        let bytes = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let events: Vec<Value> = bytes
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events[0]["operation"], "accepted");
        assert_eq!(events[1]["operation"], "complete");
        let row = &events[1]["requests"][0];
        assert_eq!(row["accepted_end"], 3);
        assert_eq!(row["execution"]["lease_id"], 2);
        assert_eq!(row["response"], response);
        assert_eq!(
            row["token_sha256"],
            format!("{:x}", Sha256::digest(b"[10,11,12]"))
        );
    }

    #[test]
    fn missing_acceptance_emits_no_success() {
        let (mut trace, path) = recording("missing");
        trace.complete(7, &Value::Null);
        drop(trace);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn cancellation_discards_trace_history_without_emitting_success() {
        let (mut trace, path) = recording("cancelled");
        trace.accepted(7, 3);
        trace.retain(std::iter::empty());
        trace.complete(7, &Value::Null);
        assert!(trace.histories.is_empty());
        assert!(trace.accepted.is_empty());
        drop(trace);
        let bytes = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(bytes.lines().count(), 1);
        assert!(!bytes.contains("completed"));
    }
}
