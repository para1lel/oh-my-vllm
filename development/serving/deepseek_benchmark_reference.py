from dataclasses import asdict

from . import deepseek_accounting as accounting
from . import deepseek_accounting_floor as accounting_floor
from . import deepseek_accounting_packed as accounting_packed
from . import deepseek_accounting_roofline as accounting_roofline
from .deepseek_accounting import BindingError, reference_trace


def _configure_floor(configuration):
    configuration.update(
        rate_profile=accounting_floor.rate_profile_digest(),
        assumptions_sha256=accounting_floor.assumptions_digest(),
        reference_scope="conditional partial resource floor",
        reference_conditions=[
            (
                "Released target matrix arithmetic, numerical order, formats, "
                "parameter placement, and replicated projections stay fixed."
            ),
            (
                "MoE output shards stay fixed, with no extra weight replication "
                "or projection recomputation."
            ),
            "Every credited MoE delivery uses complete BF16 rows.",
            "There is no unmodeled value compression or cross-row value reuse.",
            (
                "Quantized transfer, reuse, or resharding requires a new review "
                "and identity."
            ),
            (
                "This is not a complete roofline, capacity proof, or floor for "
                "every same-output algorithm."
            ),
            (
                "This selection does not establish frozen numerical, GPU, HTTP, "
                "or 33% acceptance."
            ),
        ],
        reference_zero_credit=[
            (
                "HBM and host strict credit is zero; retention and fresh-transfer "
                "proofs remain pending."
            ),
            "Embedding, attention wo_b, vocabulary-head, index, and Engram wire.",
            "DEP4 dispatch and combine strict wire.",
            (
                "MTP proposals, rejected rows, repeated prefill, replay, padding, "
                "state copies, and scheduler overhead."
            ),
            "Empirical collective latency, serial-chain estimates, and S/2.",
            "Nonlinear normalization and Sinkhorn scalar estimates.",
        ],
        reference_diagnostics=[
            "weight_union_lower_bytes",
            "direct_protocol_sent_bytes",
            "direct_protocol_received_bytes",
        ],
    )


def reference_configuration(version=accounting.VERSION):
    if version not in (
        accounting.VERSION,
        accounting_floor.VERSION,
        accounting_packed.VERSION,
        accounting_roofline.VERSION,
    ):
        raise ValueError("unknown benchmark reference version")
    configuration = {
        "accounting_version": version,
        "source_trace_version": accounting.VERSION,
        "source_trace_rate_profile": accounting.rate_profile_digest(),
        "rate_profile": accounting.rate_profile_digest(),
        "assumptions_sha256": None,
        "reference_scope": "version-bound reference estimate",
        "reference_conditions": [],
        "reference_zero_credit": [],
        "reference_diagnostics": [],
    }
    if version == accounting_floor.VERSION:
        _configure_floor(configuration)
    if version == accounting_packed.VERSION:
        configuration.update(
            rate_profile=accounting_packed.rate_profile_digest(),
            assumptions_sha256=accounting_packed.assumptions_digest(),
            reference_scope="packed-aware project reference estimate",
            reference_conditions=list(accounting_packed.ASSUMPTIONS.values()),
            reference_zero_credit=[
                "Protocol metadata and padding do not enter the project reference.",
                "No universal physical minimum or full-model numerical claim.",
            ],
            reference_diagnostics=[
                "protocol_serial_seconds",
                "protocol_metadata_pressure_bytes",
                "conditional_partial_floor_seconds",
            ],
        )
    if version == accounting_roofline.VERSION:
        configuration.update(
            rate_profile=accounting_roofline.rate_profile_digest(),
            assumptions_sha256=accounting_roofline.assumptions_digest(),
            reference_scope="fixed-algorithm resource roofline",
            reference_conditions=list(accounting_roofline.ASSUMPTIONS.values()),
            reference_zero_credit=[
                (
                    "MTP proposals, rejected rows, repeated prefill, replay, padding, "
                    "state copies, metadata, launches, and scalar estimates."
                ),
                (
                    "Unproved history reloads and embedding, attention wo_b, "
                    "vocabulary-head, index, and Engram wire."
                ),
            ],
            reference_diagnostics=[
                "weight_union_lower_bytes",
                "cache_payload_bytes",
                "protocol_stages",
            ],
        )
    return configuration


def runtime_reference(trace, *, strategy, expected_binding, version=accounting.VERSION):
    metadata = reference_configuration(version)
    if not isinstance(expected_binding, dict):
        raise BindingError("runtime reference requires an expected source binding")
    binding = trace.get("binding") if isinstance(trace, dict) else None
    if not isinstance(binding, dict):
        raise BindingError("runtime reference requires a source binding")
    calculators = {
        accounting_floor.VERSION: (
            accounting_floor,
            "conditional-partial-resource-floor",
        ),
        accounting_packed.VERSION: (
            accounting_packed,
            "fixed-protocol-project-estimate",
        ),
        accounting_roofline.VERSION: (
            accounting_roofline,
            "fixed-algorithm-resource-roofline",
        ),
    }
    if version in calculators:
        calculator, communication_model = calculators[version]
        reference = calculator.reference_trace(
            trace, strategy=strategy, expected_binding=expected_binding
        )
        metadata.update(
            communication_model=communication_model,
            profile_digest=None,
            communication_rule_version=None,
            dep4_communication_modeled=False,
            dep4_communication_measured=False,
            reference_resources=asdict(reference),
        )
    else:
        reference = reference_trace(
            trace, strategy=strategy, expected_binding=expected_binding
        )
        metadata.update(
            communication_model=reference.communication_model,
            profile_digest=reference.profile_digest,
            communication_rule_version=reference.communication_rule_version,
            dep4_communication_modeled=reference.communication_is_modeled,
            dep4_communication_measured=reference.communication_is_measured,
        )
    if reference.version != version:
        raise BindingError("reference result differs from the selected version")
    metadata.update(
        source_trace_version=trace["binding"]["version"],
        source_trace_rate_profile=trace["binding"]["rate_profile"],
    )
    return reference, metadata
