use oh_my_vllm_model_contract::ModelDescriptor;

use crate::client::WorkerConfig;
use crate::error::{Error, Result};
use crate::models::{ModelFamily, ModelPolicy};

pub(crate) fn validate_ready(
    policy: ModelPolicy,
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    logical_num_blocks: u32,
    mamba_blocks: u32,
) -> Result<ModelFamily> {
    descriptor
        .validate()
        .map_err(|error| Error::WorkerError(format!("invalid model descriptor: {error}")))?;
    policy.validate_ready(config, descriptor, logical_num_blocks, mamba_blocks)
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use serde_json::{Value, json};

    use super::*;
    use crate::protocol::{WorkerMessage, decode, encode};

    fn python_ready(drafts: usize) -> (Vec<u8>, Vec<u8>) {
        let python =
            std::env::var_os("OH_MY_VLLM_WORKER_PYTHON").unwrap_or_else(|| "python3".into());
        let source = r#"
import json
import msgpack
import sys
from tests.test_models import sample_descriptor
d = sample_descriptor(num_speculative_tokens=int(sys.argv[1]))
# The synthetic helper patches config bytes only inside its test context.
d["config_sha256"] = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
ready = dict(type="ready", descriptor=d, logical_num_blocks=1608, mamba_blocks=128)
sys.stdout.buffer.write(json.dumps(ready).encode("ascii") + b"\n")
sys.stdout.buffer.write(msgpack.packb(ready, use_bin_type=True))
"#;
        let output = Command::new(python)
            .args(["-c", source, &drafts.to_string()])
            .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let separator = output
            .stdout
            .iter()
            .position(|&byte| byte == b'\n')
            .unwrap();
        (
            output.stdout[..separator].to_vec(),
            output.stdout[separator + 1..].to_vec(),
        )
    }

    fn configuration(drafts: usize) -> WorkerConfig {
        WorkerConfig {
            num_gpu_blocks: Some(1608),
            mamba_blocks: Some(128),
            max_model_len: 4096,
            num_speculative_tokens: drafts,
            ..WorkerConfig::for_model(
                "/synthetic-qwen".into(),
                ModelPolicy::for_family(ModelFamily::Qwen38),
            )
        }
    }

    fn accepted(value: Value, config: &WorkerConfig) -> bool {
        let bytes = encode(&value).unwrap();
        let Ok(WorkerMessage::Ready {
            descriptor,
            logical_num_blocks,
            mamba_blocks,
            ..
        }) = decode(&bytes)
        else {
            return false;
        };
        let policy = if config.model_path.as_os_str() == "stub" {
            ModelPolicy::for_family(ModelFamily::Stub)
        } else {
            ModelPolicy::for_family(ModelFamily::Qwen38)
        };
        validate_ready(
            policy,
            config,
            &descriptor,
            logical_num_blocks,
            mamba_blocks,
        )
        .is_ok()
    }

    #[test]
    fn python_qwen_ready_agrees_with_rust_and_rejects_altered_contracts() {
        for drafts in [0, 4] {
            let (json_bytes, bytes) = python_ready(drafts);
            let ready: Value = rmp_serde::from_slice(&bytes).unwrap();
            let config = configuration(drafts);
            assert!(accepted(ready.clone(), &config));
            let WorkerMessage::Ready {
                descriptor,
                logical_num_blocks,
                mamba_blocks,
                ..
            } = serde_json::from_slice(&json_bytes).unwrap()
            else {
                panic!("expected Python JSON Ready");
            };
            assert_eq!(
                validate_ready(
                    ModelPolicy::for_family(ModelFamily::Qwen38),
                    &config,
                    &descriptor,
                    logical_num_blocks,
                    mamba_blocks,
                )
                .unwrap(),
                crate::models::ModelFamily::Qwen38
            );
            assert!(matches!(
                validate_ready(
                    ModelPolicy::for_family(ModelFamily::Stub),
                    &config,
                    &descriptor,
                    logical_num_blocks,
                    mamba_blocks,
                ),
                Err(Error::WorkerError(_))
            ));
            let cases = [
                ("/descriptor/schema_version", json!(2)),
                ("/descriptor/cache_schema/schema_version", json!(2)),
                ("/descriptor/architecture", json!("DeepseekV4ForCausalLM")),
                ("/descriptor/model_type", json!("deepseek_v4")),
                ("/descriptor/config_sha256", json!("a".repeat(64))),
                ("/descriptor/checkpoint_sha256", json!("invalid")),
                ("/descriptor/tokenizer_sha256", json!("A".repeat(64))),
                ("/descriptor/vocabulary_size", json!(1)),
                ("/descriptor/max_context_tokens", json!(4096)),
                ("/descriptor/capabilities", json!(["text"])),
                ("/descriptor/weight_formats", json!(["f32"])),
                (
                    "/descriptor/cache_schema/groups/0/storage/id",
                    json!("unknown"),
                ),
                (
                    "/descriptor/cache_schema/groups/1/storage/version",
                    json!(2),
                ),
                ("/descriptor/cache_schema/groups/0/block_tokens", json!(785)),
                (
                    "/descriptor/cache_schema/groups/0/capacity_blocks",
                    json!(1607),
                ),
                (
                    "/descriptor/cache_schema/groups/1/capacity_blocks",
                    json!(127),
                ),
                ("/descriptor/cache_schema/groups/0/layers", json!([3])),
                ("/descriptor/cache_schema/groups/1/layers", json!([0])),
                (
                    "/descriptor/cache_schema/groups/1/kind",
                    json!("model_checkpoint"),
                ),
                (
                    "/descriptor/cache_schema/groups/1/source",
                    json!({"type": "shared", "producer": "fa"}),
                ),
                ("/descriptor/speculation/0/method", json!("dspark")),
                ("/descriptor/speculation/0/predictor_stages", json!(2)),
                ("/descriptor/speculation/0/max_draft_tokens", json!(3)),
                ("/descriptor/speculation/0/max_verification_rows", json!(6)),
                ("/logical_num_blocks", json!(1607)),
                ("/mamba_blocks", json!(127)),
            ];
            for (path, replacement) in cases {
                let mut changed = ready.clone();
                *changed.pointer_mut(path).unwrap() = replacement;
                assert!(
                    !accepted(changed, &config),
                    "accepted {path}: drafts={drafts}"
                );
            }
            let mut changed = ready.clone();
            changed["descriptor"]["parallel_plan"] = json!({
                "strategy": "tp4", "world_size": 4, "ranks": [0, 1, 2, 3],
                "request_placement": "replicated",
            });
            assert!(!accepted(changed, &config));
            let mut changed = ready.clone();
            changed["descriptor"]["speculation"] = json!([{
                "method": "mtp", "predictor_stages": 1,
                "max_draft_tokens": 3, "max_verification_rows": 4,
            }]);
            assert!(!accepted(changed, &config));
            let mut changed = ready.clone();
            changed["descriptor"]["cache_schema"]["groups"][0]["layers"] = json!(
                (3..64)
                    .step_by(4)
                    .chain((drafts == 0).then_some(64))
                    .collect::<Vec<_>>()
            );
            assert!(!accepted(changed, &config));
            for invalid_config in [
                WorkerConfig {
                    tensor_parallel_size: 4,
                    ..config.clone()
                },
                WorkerConfig {
                    block_size: 785,
                    ..config.clone()
                },
                WorkerConfig {
                    max_model_len: 262145,
                    ..config.clone()
                },
                WorkerConfig {
                    max_model_len: 0,
                    ..config.clone()
                },
                WorkerConfig {
                    num_speculative_tokens: 1,
                    ..config.clone()
                },
                WorkerConfig {
                    speculative_mode: Some("dspark".into()),
                    ..config.clone()
                },
                WorkerConfig {
                    speculative_mode: Some(
                        if drafts == 0 {
                            "mtp"
                        } else {
                            "none"
                        }
                        .into(),
                    ),
                    ..config.clone()
                },
                WorkerConfig {
                    num_gpu_blocks: Some(1),
                    ..config.clone()
                },
                WorkerConfig {
                    mamba_blocks: Some(1),
                    ..config.clone()
                },
                WorkerConfig {
                    model_path: "stub".into(),
                    ..config.clone()
                },
            ] {
                assert!(
                    !accepted(ready.clone(), &invalid_config),
                    "{invalid_config:?}"
                );
            }
            assert!(accepted(
                ready.clone(),
                &WorkerConfig {
                    max_model_len: 262144,
                    num_gpu_blocks: None,
                    mamba_blocks: None,
                    ..config.clone()
                }
            ));
            let mut changed = ready.clone();
            changed["descriptor"]["checkpoint_sha256"] = json!("b".repeat(64));
            changed["descriptor"]["tokenizer_sha256"] = json!("c".repeat(64));
            assert!(accepted(changed, &config));
        }
    }

    #[test]
    fn python_stub_ready_is_bound_to_the_explicit_test_path_and_fixed_identities() {
        let python =
            std::env::var_os("OH_MY_VLLM_WORKER_PYTHON").unwrap_or_else(|| "python3".into());
        let source = r#"
import msgpack
import sys
from oh_my_vllm.models.stub import build_descriptor
ready = dict(type="ready", descriptor=build_descriptor(1608, 128),
             logical_num_blocks=1608, mamba_blocks=128)
sys.stdout.buffer.write(msgpack.packb(ready, use_bin_type=True))
"#;
        let output = Command::new(python)
            .args(["-c", source])
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ready: Value = rmp_serde::from_slice(&output.stdout).unwrap();
        let mut config = configuration(0);
        for path in ["/synthetic-qwen", "stub/", "stub/.", "./stub"] {
            config.model_path = path.into();
            assert!(
                !accepted(ready.clone(), &config),
                "accepted stub path {path}"
            );
        }
        config.model_path = "stub".into();
        for drafts in [0, 4] {
            config.num_speculative_tokens = drafts;
            assert!(accepted(ready.clone(), &config));
            let descriptor = serde_json::from_value(ready["descriptor"].clone()).unwrap();
            assert_eq!(
                validate_ready(
                    ModelPolicy::for_family(ModelFamily::Stub),
                    &config,
                    &descriptor,
                    1608,
                    128,
                )
                .unwrap(),
                crate::models::ModelFamily::Stub
            );
            assert!(matches!(
                validate_ready(
                    ModelPolicy::for_family(ModelFamily::Qwen38),
                    &config,
                    &descriptor,
                    1608,
                    128,
                ),
                Err(Error::WorkerError(_))
            ));
            for (path, value) in [
                ("/descriptor/config_sha256", json!("a".repeat(64))),
                ("/descriptor/checkpoint_sha256", json!("b".repeat(64))),
                ("/descriptor/tokenizer_sha256", json!("c".repeat(64))),
                (
                    "/descriptor/cache_schema/groups/0/storage/version",
                    json!(2),
                ),
                (
                    "/descriptor/cache_schema/groups/0/storage/id",
                    json!("qwen38_paired_fa_kv_bf16"),
                ),
                ("/descriptor/cache_schema/groups/1/layers", json!([1])),
                ("/descriptor/max_context_tokens", json!(4096)),
                ("/descriptor/speculation", json!([])),
            ] {
                let mut changed = ready.clone();
                *changed.pointer_mut(path).unwrap() = value;
                assert!(!accepted(changed, &config), "accepted {path}");
            }
        }
    }
}
