"""REQ-CONTEXT-001: maximum context 262144 tokens boundary placeholder.

This module registers the boundary test so it is visible to pytest. The test
requires a full model run on a B200 GPU and is skipped in the automated suite.
Run it manually with the command shown in docs/testing.md under the
"Maximum context" section:

    scripts/with-gpu.sh scripts/with-env.sh \\
        target/release/oh-my-vllm-zmq-worker \\
        --socket /tmp/boundary-mtp-4.ipc --num-gpu-blocks 4200 \\
        --mamba-blocks 128 --max-model-len 262144 \\
        --num-speculative-tokens 4 \\
        bench --batch-size 4 --input-len 258048 --output-len 4096 \\
        --warmup 0 --repetitions 1

Each of the six boundary rows (ordinary and MTP4, batch 1/2/4) must complete
with no OOM and zero preemptions.
"""

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.skip(reason="requires GPU and full model: see docs/testing.md")
def test_max_context_length_ordinary_bs1():
    """REQ-CONTEXT-001: ordinary, batch 1, 258048+4096=262144 tokens, no OOM."""
    ...


@pytest.mark.skip(reason="requires GPU and full model: see docs/testing.md")
def test_max_context_length_ordinary_bs2():
    """REQ-CONTEXT-001: ordinary, batch 2, 258048+4096=262144 tokens, no OOM."""
    ...


@pytest.mark.skip(reason="requires GPU and full model: see docs/testing.md")
def test_max_context_length_ordinary_bs4():
    """REQ-CONTEXT-001: ordinary, batch 4, 258048+4096=262144 tokens, no OOM."""
    ...


@pytest.mark.skip(reason="requires GPU and full model: see docs/testing.md")
def test_max_context_length_mtp4_bs1():
    """REQ-CONTEXT-001: MTP4, batch 1, 258048+4096=262144 tokens, no OOM."""
    ...


@pytest.mark.skip(reason="requires GPU and full model: see docs/testing.md")
def test_max_context_length_mtp4_bs2():
    """REQ-CONTEXT-001: MTP4, batch 2, 258048+4096=262144 tokens, no OOM."""
    ...


@pytest.mark.skip(reason="requires GPU and full model: see docs/testing.md")
def test_max_context_length_mtp4_bs4():
    """REQ-CONTEXT-001: MTP4, batch 4, 258048+4096=262144 tokens, no OOM."""
    ...
