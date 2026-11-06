"""DeepSeek target dimensions, resource arithmetic, and communication formulas."""

import hashlib

from .deepseek_accounting_collectives import (
    all_gather_wire_bytes as all_gather_wire_bytes,
)
from .deepseek_accounting_collectives import (
    all_reduce_wire_bytes as all_reduce_wire_bytes,
)
from .deepseek_accounting_collectives import (
    collective_latency_seconds as collective_latency_seconds,
)
from .deepseek_accounting_collectives import collective_work as collective_work
from .deepseek_accounting_collectives import (
    dep4_all_to_all_seconds as dep4_all_to_all_seconds,
)
from .deepseek_accounting_collectives import (
    reduce_scatter_wire_bytes as reduce_scatter_wire_bytes,
)
from .deepseek_accounting_collectives import (
    route_wire_pressure_bytes as route_wire_pressure_bytes,
)
from .deepseek_accounting_layers import _expert_shards as _expert_shards
from .deepseek_accounting_layers import _tensor_shards as _tensor_shards
from .deepseek_accounting_layers import attention_input_work as attention_input_work
from .deepseek_accounting_layers import attention_output_work as attention_output_work
from .deepseek_accounting_layers import compressor_work as compressor_work
from .deepseek_accounting_layers import embedding_work as embedding_work
from .deepseek_accounting_layers import engram_gate_work as engram_gate_work
from .deepseek_accounting_layers import engram_lookup_work as engram_lookup_work
from .deepseek_accounting_layers import engram_projection_work as engram_projection_work
from .deepseek_accounting_layers import hyper_connection_work as hyper_connection_work
from .deepseek_accounting_layers import index_projection_work as index_projection_work
from .deepseek_accounting_layers import local_work as local_work
from .deepseek_accounting_layers import output_head_work as output_head_work
from .deepseek_accounting_layers import routed_expert_work as routed_expert_work
from .deepseek_accounting_layers import router_work as router_work
from .deepseek_accounting_layers import shared_expert_work as shared_expert_work
from .deepseek_accounting_projection import (
    _activation_element_bytes as _activation_element_bytes,
)
from .deepseek_accounting_projection import _local_shape as _local_shape
from .deepseek_accounting_projection import _prefix_capped as _prefix_capped
from .deepseek_accounting_projection import _prefix_floor as _prefix_floor
from .deepseek_accounting_projection import _prefix_window as _prefix_window
from .deepseek_accounting_projection import _projection_parts as _projection_parts
from .deepseek_accounting_projection import _weight_bytes as _weight_bytes
from .deepseek_accounting_projection import candidate_pairs as candidate_pairs
from .deepseek_accounting_projection import compressed_pairs as compressed_pairs
from .deepseek_accounting_projection import index_score_pairs as index_score_pairs
from .deepseek_accounting_projection import projection_work as projection_work
from .deepseek_accounting_projection import window_pairs as window_pairs
from .deepseek_accounting_spec import (
    COLLECTIVE_LATENCY_PROFILE_US as COLLECTIVE_LATENCY_PROFILE_US,
)
from .deepseek_accounting_spec import (
    COLLECTIVE_PROFILE_VERSION as COLLECTIVE_PROFILE_VERSION,
)
from .deepseek_accounting_spec import DEEPSEEK_V41_MODEL as DEEPSEEK_V41_MODEL
from .deepseek_accounting_spec import DEP4_PROXY_RULE_VERSION as DEP4_PROXY_RULE_VERSION
from .deepseek_accounting_spec import ENGRAM_BYTES_PER_ROW as ENGRAM_BYTES_PER_ROW
from .deepseek_accounting_spec import RATE_PROFILE as RATE_PROFILE
from .deepseek_accounting_spec import RATES as RATES
from .deepseek_accounting_spec import STRATEGIES as STRATEGIES
from .deepseek_accounting_spec import WORK_FIELDS as WORK_FIELDS
from .deepseek_accounting_spec import WORLD_SIZE as WORLD_SIZE
from .deepseek_accounting_spec import CriticalPath as CriticalPath
from .deepseek_accounting_spec import Group as Group
from .deepseek_accounting_spec import LayerSpec as LayerSpec
from .deepseek_accounting_spec import ModelSpec as ModelSpec
from .deepseek_accounting_spec import Work as Work
from .deepseek_accounting_spec import (
    _checked_nonnegative_int as _checked_nonnegative_int,
)
from .deepseek_accounting_spec import _checked_positive_int as _checked_positive_int
from .deepseek_accounting_spec import add_work as add_work
from .deepseek_accounting_spec import critical_seconds as critical_seconds
from .deepseek_accounting_spec import group_seconds as group_seconds
from .deepseek_accounting_spec import layer_model as layer_model
from .deepseek_accounting_spec import layer_table as layer_table
from .deepseek_accounting_spec import scale_accumulation_work as scale_accumulation_work
from .deepseek_accounting_spec import scale_work as scale_work
from .deepseek_accounting_spec import serial_seconds as serial_seconds
from .deepseek_accounting_spec import (
    validate_execution_strategy as validate_execution_strategy,
)

COLLECTIVE_PROFILE_RAW_SHA256 = (
    "c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49"
)


def rate_profile_digest():
    text = "\n".join(f"{name}={value!r}" for name, value, _source in RATE_PROFILE)
    text += f"\ncollective_profile={collective_profile_digest()}"
    text += f"\ncollective_profile_version={COLLECTIVE_PROFILE_VERSION}"
    text += f"\ndep4_proxy_rule={DEP4_PROXY_RULE_VERSION}"
    return hashlib.sha256(text.encode()).hexdigest()


def collective_profile_digest():
    content = repr(COLLECTIVE_LATENCY_PROFILE_US).encode()
    raw_digest = bytes.fromhex(COLLECTIVE_PROFILE_RAW_SHA256)
    return hashlib.sha256(raw_digest + content).hexdigest()
