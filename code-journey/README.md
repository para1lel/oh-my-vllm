# oh-my-vllm code journey

A self-contained Chinese tutorial built with genuine Twine / SugarCube 2.37.3
and Tweego 2.1.1. Twelve complete chapters follow the current inference flow from
HTTP input through scheduling, cache management, model computation and output,
then explain compilation and verification. Each chapter connects the problem,
execution steps, design decisions, field/variable guides and current source.

## Read and run

The requested fixed intranet preview is http://10.30.64.14:18084/. This static
listener binds 0.0.0.0:18084 and needs no GPU or model.

From the repository root:

    npm --prefix code-journey ci --ignore-scripts
    npm --prefix code-journey run build
    npm --prefix code-journey run serve

Node 24, curl, unzip and the existing Rust environment are needed. The first
build downloads pinned Tweego, SugarCube and Fira Code Nerd Font assets and checks
SHA-256 before extraction/execution. npm dependencies are pinned in the lockfile.
Runtime fonts, KaTeX and licenses are local; reading needs no CDN.

The build runs the actual scheduler/KV crates through scripts/with-env.sh,
which supplies the required environment variables. The CPU experiment uses
synthetic worker feedback with token 42 to check scheduling and submission.
It displays actual chunk counts, not model output or GPU performance.

## Complete chapters

| Chapter | Topic | Main implementation |
|---|---|---|
| 1 | HTTP input, token IDs, positions, prefill and decode | serving/request.rs; worker/serving.py; scheduler/request.rs |
| 2 | Rust/Python process ownership and initialization | zmq-worker/main.rs; worker/model_runner.py; runtime.py; logging_utils.py |
| 3 | Request state, continuous batching, validation and commit | scheduler/request.rs; lib.rs; output.rs |
| 4 | Token budgets and aligned prefill | Scheduler::schedule; Scheduler::aligned_prefill |
| 5 | FA/GDN layout, allocation and CPU planning | kv-cache/group.rs; coordinator.rs; worker/batch_plan.py; tensors.py |
| 6 | Prefix hashes, references, queues and preemption | kv-cache/hash.rs; pool.rs; block.rs; scheduler/lib.rs |
| 7 | ZMQ messages, registration, preparation, cancellation and errors | zmq-worker/client.rs; protocol.rs; error.rs; worker/zmq_bridge.py; protocol.py |
| 8 | Checkpoint loading, FP8, residuals, FA, GDN and logits | models/qwen.py; kernels/fp8.py; attention.py; attention_prepare.py; convolution.py; gdn.py; normalization.py; elementwise.py |
| 9 | Sampling, grammar, stopping, parsing and streaming | worker/sampling.py; sampler.py; serving.py; serving/mod.rs; parser.rs; events.rs |
| 10 | MTP proposals, grouped verification and state retention | worker/mtp.py; kernels/mtp_attention.py; decode_attention.py; batch_plan.py |
| 11 | Semantic IR, lowering, graph capture/cache and CUDA kernels | ir/; worker/decode_graph.py; graph_cache.py; kernels/backend.py; cuda_backend/ |
| 12 | Correctness, maximum shapes, TTFT, throughput and measurement audit | ir/coverage.py; tests/; benchmarks/ttft.py; measurement.py; development/kernels/cases.py |

src/curriculum.mjs maps substantive Rust/Python/CUDA runtime files to chapter
responsibilities and implementation decisions. The build checks every default
first-party runtime file for coverage, verifies entry anchors, and records real
line numbers and full-file hashes. Test modules, documentation-only initializers
and frozen TileLang comparison providers are outside this default-flow inventory;
chapter 11 explains the comparison backend's contracts. Independent libraries
are explained through their project call sites.

Each chapter's conditional source list opens a locally highlighted complete file
at its entry line. Rust, Python and C++ use the same light/dark code palette.
The source page links back to the chapter; unread prerequisites still apply.
Rebuild after changing referenced source.

## Reading and navigation

All page navigation is an unordered list. Each item states a condition and its
destination, following the supplied Matrix67 example. Opening choices record
interest in basics, request management or model computation. An unread target
resolves to its earliest missing prerequisite; continuation links preserve the
selected route. Chapters assume only their completed internal prerequisites.

Only an explicit understood-and-continue choice completes the current chapter.
Review, map and header links preserve its reading-in-progress state. Read markers
survive backward navigation. A checked localStorage v2 schema restores the last
chapter and experiment step. Old short-passage visits migrate to visited markers,
without treating them as completion of expanded chapters. Restart clears both
reading schemas and Twine history. Theme preference persists separately.

The five experiment lengths are 784, 785, 1568, 1569 and 32768. The last case
executes 32144 then 624 input tokens and produces its first synthetic output at
the end. Controls update the real published scheduler trace.

## Typography and source authoring

The Matrix67 reading style uses a white background, blue-gray text and green
underlined links. The column is 820px, desktop/mobile prose 17/16px, chapter titles
28/24px and source code 14/13px. Prose uses LXGW WenKai, code Fira Code Nerd Font
Mono with LXGW WenKai fallback for Chinese code glyphs, and formulas local KaTeX with accessible MathML. Light/dark themes persist.

Chapter files live in src/chapters/. Inline code uses the icode macro and
textContent so operators such as Python // cannot become Twine formatting. The entry and map are in src/story.twee;
titles, prerequisites and the source inventory are in src/curriculum.mjs.
Use halfwidth punctuation and the requested spaces in prose. Source code and
identifiers preserve their syntax. Humanizer-zh is installed in the user's Codex
skills and applied to prose, preserving facts and meaningful qualifications.

src/source-specs.mjs selects excerpts, and src/source-notes.mjs explains every
shown record field, function parameter, state and local variable. The build
rejects unexplained displayed public Rust fields or Python dataclass fields.
Mobile meaning/purpose tables stack with labels.

scripts/excerpt.mjs removes the shared literal whitespace prefix from nonblank
excerpt lines and normalizes blank rows. Relative indentation, original source
line numbers and full-file hashes remain. Display and copy use the normalized
excerpt; complete source pages preserve the full file. Shiki 4.5.0 tokenizes at
build time, with no runtime highlighter. Focusable code regions scroll locally.
Clipboard API and an HTTP selection fallback both copy the displayed code.

## Verification

With the static listener running:

    npm --prefix code-journey test
    scripts/with-env.sh cargo test --locked --manifest-path code-journey/trace/Cargo.toml
    scripts/with-env.sh cargo fmt --manifest-path code-journey/trace/Cargo.toml --check
    scripts/with-env.sh cargo clippy --manifest-path code-journey/trace/Cargo.toml --all-targets -- -D warnings

npm test first runs Node tests for Rust/Python, tabs, relative indentation and
blank lines, then the pinned Chromium workflow. Playwright checks all twelve
chapter routes, interest and prerequisites, conditional unordered navigation,
completion versus review, progress migration, all five scheduler cases,
light/dark/refresh/reset/history, every displayed field, formulas, fonts and
mobile overflow. It also verifies excerpt/source hashes and line numbers,
normalized copying through both clipboard paths, code contrast >=4.5:1,
keyboard scrolling, every full-source URL and Rust/Python/CUDA source pages.

If Chromium is absent, install the pinned browser from code-journey:

    npx playwright install chromium

JOURNEY_URL overrides the test URL. JOURNEY_QA_DIR selects screenshot storage
outside the repository; default /tmp/oh-my-vllm-journey-qa. Inspect rendered
screenshots separately from functional checks, including desktop 1536x1024 and
mobile 390x844. Browser plugin was unavailable for this revision, so Playwright
Chromium supplies validation. Repository-wide Rust/Ruff checks remain required.

Generated dist, .tools, node_modules, data and browser scratch are ignored.
Inference runtime, kernel implementations and baseline evidence are unchanged.
