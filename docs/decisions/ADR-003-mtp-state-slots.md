# ADR-003: MTP target-state slots and precision

Accepted 2026-09-19 within the authorized module/precision implementation scope.

The draft model does not contain GDN layers, but target verification of K draft
tokens needs K additional recurrent-state slots. Rust reserves these slots,
reuses them after rejected drafts, migrates them after large prefill chunks,
and preserves the previously accepted state until the next execution consumes
it. The worker receives actual scheduled drafts and returns actual accepted
output and next-draft token IDs. Rust rolls back only scheduled rejected drafts.

For the installed model/backend, default FP32 SSM plus four speculative slots
causes vLLM's platform cache sizing to increase block_size from 784 to 1568.
The project requires 784. MTP therefore explicitly uses BF16 SSM cache storage;
ordinary mode retains the backend's auto precision. The matched vLLM baseline
uses exactly the same choice. Initialization asserts that block_size stays 784.
This precision choice is part of the benchmark configuration, not a universal
claim about MTP accuracy or compatibility with arbitrary draft counts.

Actual-path tests instrument target GDN fused MTP verification and all its draft
states, together with actual GQA and prefill calls, against CPU FP64 references
using the same rounded inputs. Short and cross-block Chinese generation passed
these checks and produced coherent text. State error tolerances and limitations
are recorded in testing.md. Full-model token identity is not an acceptance gate.
