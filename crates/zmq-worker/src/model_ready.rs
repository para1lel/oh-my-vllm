use std::collections::BTreeSet;

use oh_my_vllm_model_contract::{
    CacheGroup, CacheKind, CacheSource, CacheStorage, Capability, ModelDescriptor,
    ParallelStrategy, SpeculationMethod, SpeculationSpec, WeightFormat,
};

use crate::client::WorkerConfig;
use crate::error::{Error, Result};
use crate::models::ModelFamily;

const QWEN_CONFIG_SHA256: &str = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc";
const STUB_CONFIG_SHA256: &str = "3b7c7f7627a034cd753e4d4db30346a940e49b635952460c51b2ba5f4f332f18";
const STUB_CHECKPOINT_SHA256: &str =
    "6cafe580d6f80a355f3e30bf2c8a7559aef2b081845084d16e9eca733c09d7c2";
const STUB_TOKENIZER_SHA256: &str =
    "243e380a5f7db0cea0f70d4ddc53663d47e9f2b191b928fa5ccbba201ecf99b7";
const CAPABILITIES: [Capability; 8] = [
    Capability::Text,
    Capability::Images,
    Capability::Video,
    Capability::Tools,
    Capability::Reasoning,
    Capability::StructuredOutputs,
    Capability::PrefixCache,
    Capability::HostOffload,
];

pub(crate) fn validate_ready(
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    logical_num_blocks: u32,
    mamba_blocks: u32,
) -> Result<ModelFamily> {
    descriptor
        .validate()
        .map_err(|error| Error::WorkerError(format!("invalid model descriptor: {error}")))?;
    require(
        config.tensor_parallel_size == 1
            && descriptor.parallel_plan.strategy == ParallelStrategy::Tp1,
        "only TP1 is implemented",
    )?;
    require(
        descriptor.vocabulary_size == 248320 && descriptor.max_context_tokens == 262144,
        "unsupported vocabulary or context ceiling",
    )?;
    require(
        config.max_model_len > 0
            && u64::from(config.max_model_len) <= descriptor.max_context_tokens,
        "requested context exceeds supported limits",
    )?;
    require(
        config.block_size == 784,
        "only 784-token cache blocks are implemented",
    )?;
    require(
        logical_num_blocks > 1
            && (2..=65536).contains(&mamba_blocks)
            && config
                .num_gpu_blocks
                .is_none_or(|count| count == logical_num_blocks)
            && config
                .mamba_blocks
                .is_none_or(|count| count == mamba_blocks),
        "invalid worker cache capacities",
    )?;
    require(
        same_members(&descriptor.capabilities, &CAPABILITIES),
        "unsupported capability set",
    )?;
    validate_speculation(config, descriptor)?;
    require(
        descriptor.cache_schema.groups.len() == 2,
        "expected paired FA and GDN cache groups",
    )?;
    let fa = cache_group(descriptor, "fa")?;
    let gdn = cache_group(descriptor, "gdn")?;
    require(
        fa.capacity_blocks == u64::from(logical_num_blocks)
            && gdn.capacity_blocks == u64::from(mamba_blocks)
            && fa.block_tokens == config.block_size
            && gdn.block_tokens == config.block_size,
        "cache groups disagree with Ready capacities or Init block size",
    )?;
    if config.model_path.as_os_str() == "stub" {
        validate_stub(descriptor, fa, gdn)?;
        Ok(ModelFamily::Stub)
    } else {
        validate_qwen(config, descriptor, fa, gdn)?;
        Ok(ModelFamily::Qwen38)
    }
}

fn validate_speculation(config: &WorkerConfig, descriptor: &ModelDescriptor) -> Result<()> {
    require(
        matches!(config.num_speculative_tokens, 0 | 4),
        "only ordinary decode and four-draft MTP are implemented",
    )?;
    let mode = if config.num_speculative_tokens == 4 {
        "mtp"
    } else {
        "none"
    };
    require(
        config
            .speculative_mode
            .as_deref()
            .is_none_or(|value| value == mode),
        "speculative mode disagrees with the requested draft count",
    )?;
    require(
        descriptor.speculation
            == [SpeculationSpec {
                method: SpeculationMethod::Mtp,
                predictor_stages: 1,
                max_draft_tokens: 4,
                max_verification_rows: 5,
            }],
        "unsupported speculation limits",
    )
}

fn validate_qwen(
    config: &WorkerConfig,
    descriptor: &ModelDescriptor,
    fa: &CacheGroup,
    gdn: &CacheGroup,
) -> Result<()> {
    require(
        descriptor.architecture == "Qwen3_5ForConditionalGeneration"
            && descriptor.model_type == "qwen3_5"
            && descriptor.config_sha256 == QWEN_CONFIG_SHA256,
        "unsupported Qwen model identity",
    )?;
    require(
        same_members(
            &descriptor.weight_formats,
            &[
                WeightFormat::Bf16,
                WeightFormat::F32,
                WeightFormat::QwenFp8E4m3Block128x128Fp32Scales,
            ],
        ),
        "unsupported Qwen weight formats",
    )?;
    let mut fa_layers: Vec<_> = (3..64).step_by(4).collect();
    if config.num_speculative_tokens == 4 {
        fa_layers.push(64);
    }
    validate_group(
        fa,
        CacheKind::FullAttention,
        "qwen38_paired_fa_kv_bf16",
        &fa_layers,
    )?;
    validate_group(
        gdn,
        CacheKind::Recurrent,
        "qwen38_paired_gdn_f32_conv_bf16_hidden_bf16",
        &(0..64).filter(|layer| layer % 4 != 3).collect::<Vec<_>>(),
    )
}

fn validate_stub(descriptor: &ModelDescriptor, fa: &CacheGroup, gdn: &CacheGroup) -> Result<()> {
    require(
        descriptor.architecture == "OhMyVllmStubForTesting"
            && descriptor.model_type == "oh_my_vllm_stub"
            && descriptor.config_sha256 == STUB_CONFIG_SHA256
            && descriptor.checkpoint_sha256 == STUB_CHECKPOINT_SHA256
            && descriptor.tokenizer_sha256 == STUB_TOKENIZER_SHA256
            && descriptor.weight_formats == [WeightFormat::F32],
        "unsupported test stub identity or sentinel",
    )?;
    validate_group(
        fa,
        CacheKind::ModelCheckpoint,
        "oh_my_vllm_stub_fa_metadata",
        &[0],
    )?;
    validate_group(
        gdn,
        CacheKind::ModelCheckpoint,
        "oh_my_vllm_stub_gdn_metadata",
        &[0],
    )
}

fn cache_group<'a>(descriptor: &'a ModelDescriptor, name: &str) -> Result<&'a CacheGroup> {
    descriptor
        .cache_schema
        .groups
        .iter()
        .find(|group| group.name == name)
        .ok_or_else(|| Error::WorkerError(format!("missing {name} cache group")))
}

fn validate_group(
    group: &CacheGroup,
    kind: CacheKind,
    layout_id: &str,
    layers: &[u32],
) -> Result<()> {
    require(
        group.kind == kind
            && group.layers == layers
            && group.source == (CacheSource::Owned {})
            && matches!(
                &group.storage,
                CacheStorage::Layout { id, version: 1 } if id == layout_id
            ),
        "unsupported cache kind, layout, layer membership, or producer",
    )
}

fn same_members<T: Ord>(actual: &[T], expected: &[T]) -> bool {
    actual.iter().collect::<BTreeSet<_>>() == expected.iter().collect::<BTreeSet<_>>()
}

fn require(valid: bool, reason: &str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(Error::WorkerError(format!(
            "invalid model readiness: {reason}"
        )))
    }
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
from tests.test_model_descriptor import sample_descriptor
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
            model_path: "/synthetic-qwen".into(),
            num_gpu_blocks: Some(1608),
            mamba_blocks: Some(128),
            max_model_len: 4096,
            num_speculative_tokens: drafts,
            ..Default::default()
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
        validate_ready(config, &descriptor, logical_num_blocks, mamba_blocks).is_ok()
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
                validate_ready(&config, &descriptor, logical_num_blocks, mamba_blocks).unwrap(),
                crate::models::ModelFamily::Qwen38
            );
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
                validate_ready(&config, &descriptor, 1608, 128).unwrap(),
                crate::models::ModelFamily::Stub
            );
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
