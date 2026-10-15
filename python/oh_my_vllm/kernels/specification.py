from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    stages: tuple[str, ...]
    state: tuple[str, ...] = ()


MODEL = {
    "config_sha256": "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc",
    "hidden": 5120,
    "intermediate": 17408,
    "vocabulary": 248320,
    "layers": 64,
    "fa_layers": 16,
    "gdn_layers": 48,
    "query_heads": 24,
    "kv_heads": 4,
    "head_dim": 256,
    "rotary_dim": 64,
    "rope_theta": 10000000,
    "mrope_section": (11, 11, 10),
    "gdn_key_heads": 16,
    "gdn_value_heads": 48,
    "gdn_dim": 128,
    "conv_width": 4,
    "scale_block": 128,
    "block_size": 784,
    "step_tokens": 32768,
    "context_tokens": 262144,
    "mtp_steps": 4,
    "epsilon": 1e-6,
    "vision_layers": 27,
    "vision_hidden": 1152,
    "vision_intermediate": 4304,
    "vision_heads": 16,
    "vision_head_dim": 72,
    "vision_positions": 2304,
    "patch": 16,
    "temporal_patch": 2,
    "spatial_merge": 2,
}

OPERATORS = (
    Operator(
        "gdn_input",
        (
            "x,residual:bf16[T,5120]",
            "w_qkvz:e4m3[16384,5120];scale:fp32[128,40]",
            "w_ba:bf16[96,5120]",
            "norm:fp32[5120];conv:bf16[10240,4]",
            "A_log,dt_bias:fp32[48]",
            "conv_state:bf16[slots,10240,3]",
            "sequence_rows:i32[B+1];read_slots:i32[B];write_slots:i32[T]",
            "checkpoint_rows,checkpoint_slots:i32[C]",
            "token_ids:i64[T];embedding:bf16[248320,5120] (first layer)",
            "media:bf16[P,5120];media_rows:i32[P] (first layer)",
        ),
        (
            "q,k:bf16[T,16,128];v,z:bf16[T,48,128]",
            "decay,beta:fp32[T,48];residual:bf16[T,5120]",
        ),
        (
            "optional_embedding_media_gather",
            "residual_add_rms",
            "group128_activation_quantization",
            "packed_fp8_projection_and_bf16_ba_projection",
            "causal_convolution_silu",
            "qk_l2_normalization",
            "decay_negative_exp_a_log_softplus_a_dt_bias",
            "beta_sigmoid",
        ),
        ("read_conv_checkpoint", "write_private_conv_rows"),
    ),
    Operator(
        "gdn_recurrent",
        ("q,k,v,decay,beta", "state:bf16[slots,48,128,128]", "sequence_rows:i32[B+1]"),
        ("output:bf16[T,48,128]",),
        ("decay_state", "delta_rank1_update", "query_state_product"),
        ("read_accepted_state", "write_private_state_per_verification_row"),
    ),
    Operator(
        "gdn_prefill",
        ("q,k,v,decay,beta", "state:bf16[slots,48,128,128]", "sequence_rows:i32[B+1]"),
        ("output:bf16[T,48,128]",),
        (
            "chunk64_delta_scan",
            "fp32_state_accumulation",
            "checkpoint_at_784_boundaries",
        ),
        ("read_accepted_state", "write_aligned_checkpoints_and_final_state"),
    ),
    Operator(
        "gdn_output",
        (
            "x,z:bf16[T,48,128];norm:fp32[128]",
            "weight:e4m3[5120,6144];scale:fp32[40,48]",
        ),
        ("output:bf16[T,5120]",),
        ("head_rms", "silu_z_gate", "group128_quantization", "fp8_projection"),
    ),
    Operator(
        "gqa_input",
        (
            "x,residual:bf16[T,5120];norm:fp32[5120]",
            "weight:e4m3[14336,5120];scale:fp32[112,40]",
            "q_norm,k_norm:fp32[256];positions:i64[3,T]",
            "kv:bf16[blocks,2,784,4,256];slots:i32[T]",
            "variant:full|kv_only|query_only;query_rows:i32[Q]",
        ),
        ("query,gate:bf16[Q,24,256]", "residual:bf16[Q,5120]"),
        (
            "residual_add_rms",
            "group128_quantization",
            "packed_q_gate_k_v_projection",
            "qk_head_rms",
            "interleaved_partial_mrope",
            "paged_kv_store",
        ),
        ("write_owned_kv_slots_only",),
    ),
    Operator(
        "gqa_attention_output",
        (
            "query,gate:bf16[T,24,256]",
            "kv:bf16[blocks,2,784,4,256];block_table:i32[B,Lblocks]",
            "sequence_rows,context_lengths:i32",
            "weight:e4m3[5120,6144];scale:fp32[40,48]",
        ),
        ("output:bf16[T,5120]",),
        (
            "causal_gqa_online_softmax",
            "sigmoid_gate",
            "group128_quantization",
            "fp8_projection",
        ),
        ("read_prefix_and_private_verification_rows",),
    ),
    Operator(
        "ffn_gate_up",
        (
            "x,residual:bf16[T,5120];norm:fp32[5120]",
            "weight:e4m3[34816,5120];scale:fp32[272,40]",
        ),
        ("gate_up:bf16[T,34816]", "residual:bf16[T,5120]"),
        ("residual_add_rms", "group128_quantization", "packed_gate_up_projection"),
    ),
    Operator(
        "ffn_down",
        (
            "gate_up:bf16[T,34816]",
            "weight:e4m3[5120,17408];scale:fp32[40,136]",
        ),
        ("output:bf16[T,5120]",),
        ("silu_gate_multiply", "group128_quantization", "fp8_projection"),
    ),
    Operator(
        "output_sample",
        (
            "x,residual:bf16[R,5120];norm:fp32[5120]",
            "weight:bf16[248320,5120]",
            "history,grammar_mask,sampling_parameters,philox_seed_and_counter",
        ),
        (
            "token_ids:i64[R]",
            "selected_probabilities:fp32[R]",
            "proposal_distribution (MTP)",
        ),
        (
            "selected_row_residual_rms",
            "bf16_vocabulary_projection",
            "penalties_temperature_grammar",
            "greedy_or_topk_topp_sample",
        ),
        ("advance_request_rng_counter",),
    ),
    Operator(
        "mtp_input",
        (
            "hidden:bf16[B,5120];token_ids:i64[B]",
            "context_hidden:bf16[T,5120];context_token_ids:i64[T] (context fill)",
            "boundary_hidden:bf16[B,5120];positions:i64[T]",
            "embedding:bf16[248320,5120];norm_hidden,norm_embedding:fp32[5120]",
            "weight:bf16[5120,10240]",
        ),
        ("output:bf16[B,5120]",),
        ("embedding_gather", "dual_rms", "implicit_concatenation_bf16_projection"),
    ),
    Operator(
        "mtp_accept_commit",
        (
            "target_probabilities:fp32[B,5,248320]",
            "proposal_probabilities:fp32[B,4,248320]",
            "proposals:i64[B,4];philox_seed_and_counter",
            "private_state,conv_state;accepted_state_slots:i32[B]",
        ),
        ("kept_tokens:i64[B,5]", "accepted_counts:i32[B]"),
        (
            "conditional_accept_reject",
            "residual_distribution_sample",
            "accepted_state_selection",
        ),
        ("publish_accepted_gdn_and_conv_state", "discard_rejected_rows"),
    ),
    Operator(
        "vision_patch_input",
        (
            "rgb:u8[frames,height,width,3];grid:i32[media,3]",
            "weight:bf16[1152,1536];bias:bf16[1152]",
            "position_embedding:bf16[2304,1152]",
        ),
        ("output:bf16[P,1152]",),
        (
            "bicubic_resize_normalize",
            "temporal2_spatial16_patch_projection",
            "bilinear_absolute_position_interpolation",
            "spatial_merge_order",
        ),
    ),
    Operator(
        "vision_qkv",
        (
            "x,residual:bf16[P,1152];norm,bias:bf16[1152]",
            "weight:bf16[3456,1152];projection_bias:bf16[3456]",
            "spatial_positions:i32[P,2]",
        ),
        ("q,k,v:bf16[P,16,72]", "residual:bf16[P,1152]"),
        ("residual_layer_norm", "bf16_qkv_bias_projection", "spatial_2d_rope"),
    ),
    Operator(
        "vision_attention_output",
        (
            "q,k,v:bf16[P,16,72];frame_rows:i32[frame_groups+1]",
            "weight:bf16[1152,1152];bias:bf16[1152]",
        ),
        ("output:bf16[P,1152]",),
        ("frame_local_noncausal_online_softmax", "bf16_projection_bias"),
    ),
    Operator(
        "vision_ffn_up",
        (
            "x,residual:bf16[P,1152];norm,bias:bf16[1152]",
            "weight:bf16[4304,1152];projection_bias:bf16[4304]",
        ),
        ("output:bf16[P,4304]", "residual:bf16[P,1152]"),
        ("residual_layer_norm", "bf16_projection_bias_gelu_tanh"),
    ),
    Operator(
        "vision_ffn_down",
        ("x:bf16[P,4304]", "weight:bf16[1152,4304];bias:bf16[1152]"),
        ("output:bf16[P,1152]",),
        ("bf16_projection_bias",),
    ),
    Operator(
        "vision_merge_up",
        (
            "x,residual:bf16[P,1152];norm,bias:bf16[1152]",
            "weight:bf16[4608,4608];projection_bias:bf16[4608]",
        ),
        ("output:bf16[P/4,4608]",),
        ("residual_layer_norm", "spatial2x2_gather", "bf16_projection_bias_gelu_exact"),
    ),
    Operator(
        "vision_merge_down",
        ("x:bf16[P/4,4608]", "weight:bf16[5120,4608];bias:bf16[5120]"),
        ("output:bf16[P/4,5120]",),
        ("bf16_projection_bias",),
    ),
)
