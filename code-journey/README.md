# oh-my-vllm code journey

A Chinese interactive essay about this repository, built with genuine Twine /
SugarCube 2.37.3 and Tweego 2.1.1. The first slice follows one request from token
IDs to its first retained output token. Eleven reading passages and a reading
map are implemented. The twelve-chapter curriculum below covers the rest of the
current core; later chapters are planned.

## Read and run

The fixed intranet preview is http://10.30.64.14:18084/. The listener serves only
the generated static site, binds 0.0.0.0:18084, and runs without a GPU or model.

From the repository root:

    npm --prefix code-journey ci --ignore-scripts
    npm --prefix code-journey run build
    npm --prefix code-journey run serve

Node 24, curl, unzip, and the existing Rust environment are needed. The first
build downloads pinned Tweego, SugarCube and Fira Code Nerd Font assets, verifies
their SHA-256 hashes before extraction/execution, and caches them in .tools.
KaTeX 0.19.0, LXGW WenKai and Playwright are pinned in package-lock.json. Runtime
assets and their licenses are served locally; reading requires no CDN.

The Rust build goes through scripts/with-env.sh, including both required
environment variables. The trace crate depends on the actual scheduler/KV crates.
It constructs synthetic worker feedback to exercise their CPU output contract.
The displayed output count is a contract illustration; token 42 in the fixture
is synthetic. The page does not run a model or measure GPU performance.

## Reading behavior

The opening choices record interest in fundamentals, request flow, or GPU
ownership. Navigation resolves unread prerequisite passages first, then resumes
the selected destination. Core articles assume only their listed prerequisites.

The read marker advances when the reader chooses a continuation; the opening
article is marked at entry. A separate visited marker distinguishes an opened
article from a completed one. Read markers survive backward navigation. Refresh
restores the last lesson and experiment step from a checked localStorage schema.
Restart clears lesson state and Twine history. Theme preference is independent;
the initial theme follows the OS, and an explicit choice persists.

The five experiment lengths are 784, 785, 1568, 1569 and 32768. Source snippets,
line numbers and full-file hashes are extracted at build time. An anchor mismatch
stops the build. Rebuild after changing the Rust/Python code.

## Complete curriculum

| Chapter | Question | Current source entry points |
|---|---|---|
| 1 | How does text become input and position? | crates/scheduler/src/request.rs; python/oh_my_vllm/worker/serving.py |
| 2 | Who decides and who computes? | crates/zmq-worker/src/client.rs; python/oh_my_vllm/worker/zmq_bridge.py; docs/architecture.md |
| 3 | How does a request enter continuous batching? | crates/scheduler/src/lib.rs; request.rs; output.rs |
| 4 | How are token budgets and prefill chunks chosen? | Scheduler::schedule; Scheduler::aligned_prefill |
| 5 | Why do FA pages and GDN states differ? | crates/kv-cache/src/group.rs; coordinator.rs; python/oh_my_vllm/worker/batch_plan.py |
| 6 | How do prefix reuse, references and preemption work? | kv-cache/src/hash.rs; pool.rs; block.rs; Scheduler::preempt_request |
| 7 | How do transport, cancellation and errors interact? | zmq-worker/src/protocol.rs; client.rs; error.rs; worker/protocol.py; zmq_bridge.py |
| 8 | How does Qwen execute FP8, FA and GDN layers? | models/qwen.py; kernels/fp8.py; attention.py; decode_attention.py; attention_prepare.py; gdn.py; convolution.py; normalization.py; elementwise.py; mtp_attention.py |
| 9 | How do sampling, grammar and streaming work? | worker/sampler.py; sampling.py; serving.py; zmq-worker/src/serving/request.rs; parser.rs; events.rs; mod.rs |
| 10 | How are MTP proposals verified and committed? | worker/mtp.py; batch_plan.py; model_runner.py; Scheduler::validate_output; Scheduler::update |
| 11 | How do semantic IR, compilation and CUDA Graphs fit? | ir/core.py; ir/coverage.py; ir/attention.py; ir/recurrent.py; ir/fp8.py; ir/pointwise.py; ir/logits.py; ir/state.py; ir/gdn_prefill.py; worker/graph_cache.py; decode_graph.py; kernels/cuda_backend/kernels.cu |
| 12 | What establishes correctness and performance acceptance? | tests/; worker/runtime.py; logging_utils.py; tensors.py; benchmarks/ttft.py; measurement.py; development/kernels/cases.py; bench/baseline/; docs/acceptance.md |

Each later chapter should connect a concrete problem, an interactive prediction,
the current source, a design decision and a check. Historical vLLM adapter/V2 ADRs
belong in dated side articles. Current runtime, CUDA-default and semantic-IR
behavior form the main route. Performance evidence remains tied to the source
revision named in its acceptance artifacts.

## Design and authoring

The Matrix67 reference supplies the quiet, branching reading model: true-white
background, blue-gray text and underlined green choices. Image Gen concepts were
created for the light reading screen, the experiment, and the dark reading screen.
The design uses one open 950px column, deliberate text/control sizing and no card
grid. Mobile uses a 24px gutter. The dark palette uses #161D23, #D5DFE6 and #8FC99F.

Prose uses LXGW WenKai; code uses Fira Code Nerd Font Mono. Formula macros render
with local KaTeX, including accessible MathML. UI/content remain native HTML and
SugarCube passages. The narrow code preview is extracted from the same source
as its expandable full excerpt.

Chinese prose uses halfwidth punctuation, a space before an opening parenthesis,
and spaces after punctuation and between Chinese and Latin/formula runs. Source
code, identifiers, commands and formulas preserve their syntax. Humanizer-zh was
installed in the user's Codex skills and used to review prose for repetition and
unsupported claims.

## Verification

With the static preview running:

    npm --prefix code-journey test
    scripts/with-env.sh cargo test --locked --manifest-path code-journey/trace/Cargo.toml
    scripts/with-env.sh cargo fmt --manifest-path code-journey/trace/Cargo.toml --check
    scripts/with-env.sh cargo clippy --manifest-path code-journey/trace/Cargo.toml --all-targets -- -D warnings

If Chromium is absent, run the pinned Playwright installer from code-journey:

    npx playwright install chromium

JOURNEY_URL overrides the test base URL. JOURNEY_QA_DIR selects a screenshot
directory outside the repository; the default is /tmp/oh-my-vllm-journey-qa.
Playwright checks real Twine identity, prerequisite routing, all five boundary
traces, source display, theme/refresh/reset behavior, mobile overflow and console
health. Screenshots cover 1536x1024 desktop and 390x844 mobile in both themes.
The separate CPU test checks the published chunk counts against the actual Rust
scheduler. Repository-wide Rust/Ruff checks remain required.

The Browser plugin was unavailable, so the preview was verified with Playwright
Chromium. The generated design and rendered screenshots were inspected with
view_image. Functional tests and visual comparison are separate checks.

## Files

- src/story.twee: passages and prerequisite teaching content.
- src/setup.js: SugarCube macros, checked persistence, theme and trace playback.
- src/style.css and src/head.html: design tokens, fonts and local KaTeX.
- scripts/build.mjs: checksum-verified tool setup, current source/trace extraction.
- scripts/serve.mjs: static-only fixed-port listener.
- trace/: CPU trace executable and scheduler-based boundary test.
- tests/: documented browser acceptance checks.

Generated dist, .tools, node_modules, generated data and browser scratch stay
outside version control. No inference runtime, kernel or baseline artifact was
changed by this slice.
