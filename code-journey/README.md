# Interactive code tutorial

This static Twine/SugarCube 2.37.3 tutorial gives current inference code in twelve Chinese chapters.
Each chapter includes logic, decisions, implementation, and displayed fields/variables.
Readers use interest choices and internal prerequisites to select their path.
Readers must know basic Rust and Python syntax.

## Build and preview

Use Node 24, curl, unzip, and the configured Rust environment from [development](../docs/development.md).
From the repository root:

```bash
npm --prefix code-journey ci --ignore-scripts
npm --prefix code-journey run build
npm --prefix code-journey run serve
```

The static listener defaults to `0.0.0.0:18084` and uses no GPU or model.
Keep the preview address and process details in ignored `LOCAL.md`.
The first build downloads pinned Tweego, SugarCube, and Fira Code Nerd Font assets.
It checks SHA-256 before extraction or execution.
The npm lock pins the other dependencies.

Fonts, KaTeX, and licenses are served locally without a reading-time CDN.

The build uses project scheduler/KV crates through `scripts/with-env.sh`.
The CPU experiment supplies synthetic worker feedback with token 42.
It shows scheduling and submission, without model output or GPU performance claims.

## Chapters and source

| Chapter | Topic |
|---|---|
| 1 | HTTP preparation, token IDs, positions, prefill, and decode. |
| 2 | Rust/Python processes, environment, and initialization. |
| 3 | Request history, batching, reply validation, and commit. |
| 4 | Token budgets and aligned prefill. |
| 5 | FA/GDN layout, allocation, and CPU planning. |
| 6 | Prefix hashes, references, queues, and preemption. |
| 7 | ZMQ, registration, preparation, cancellation, and failures. |
| 8 | Loading, FP8/ordinary projections, residuals, FA, GDN, and logits. |
| 9 | Sampling, grammar, stopping, parsing, and streams. |
| 10 | MTP proposals, grouped verification, and state ownership. |
| 11 | Semantic IR, lowering, graph caches, and CUDA kernels. |
| 12 | Correctness, maximum shapes, TTFT, throughput, and measurement audit. |

`src/curriculum.mjs` maps each substantive default-runtime Rust/Python/CUDA file to a chapter and entry symbol.
The build rejects missing coverage, anchors, and record-field explanations.
It records current source lines and full-file hashes.
Pinned comparison providers, tests, and documentation-only initializers have different scope.
Third-party libraries have descriptions at their project call sites.

Conditional source lists open locally highlighted full files at the relevant line.
Source pages link back to the chapter and have the two themes.
Rebuild after you change referenced source.

## Navigation and experiment

Each page transition is an unordered-list item with a condition and destination.
Unread destinations resolve to their earliest missing internal prerequisite.
Continuation keeps the selected route.
Only an explicit completion choice completes a chapter.
Review/map visits keep reading progress.

The checked version 2 localStorage schema restores the last chapter and experiment state.
Old short passages migrate to visits.
They do not complete expanded chapters.
Restart clears reading schemas and Twine history. Theme choice has its own state.

The five prompt lengths are 784,785,1568,1569,32768.
The last trace schedules 32144 and 624 tokens before the first synthetic output.
Controls use the published Rust scheduler trace.

## Typography and excerpts

The column is 820px with desktop/mobile prose 17/16px, headings 28/24px, and code 14/13px.
Prose uses LXGW WenKai. Code uses Fira Code Nerd Font Mono with LXGW WenKai for Chinese glyphs.
KaTeX supplies formulas and accessible MathML.
Light and dark themes persist.

Chinese prose uses halfwidth punctuation and the requested spaces with Humanizer-zh editing.
Source identifiers and syntax stay unchanged.

Chapter text is in `src/chapters/`. Entry/map text is in `src/story.twee`.
Use the `icode` macro and `textContent` for inline code so Python `//` stays literal.
`src/source-specs.mjs` selects excerpts. `src/source-notes.mjs` gives their fields, parameters, state, and local variables.

Mobile meaning/purpose tables keep explicit labels.

`scripts/excerpt.mjs` removes the common literal whitespace prefix and normalizes blank lines.
It keeps relative indentation, documentation, attributes, decorators, body docstrings, source lines, and file hashes.
Display and copy use the same normalized excerpt. Full-source pages keep the unmodified source file.
Shiki 4.5.0 tokenizes Rust/Python/C++ at build time.

Focusable code regions permit keyboard scrolling.
Clipboard API and HTTP selection fallback copy the displayed code.

## Verification

With the static listener active:

```bash
npm --prefix code-journey test
scripts/with-env.sh cargo test --locked --manifest-path code-journey/trace/Cargo.toml
scripts/with-env.sh cargo fmt --manifest-path code-journey/trace/Cargo.toml --check
scripts/with-env.sh cargo clippy --manifest-path code-journey/trace/Cargo.toml --all-targets -- -D warnings
```

Node tests include documentation/metadata boundaries, dedentation, tabs, blank lines, and neighboring elements.
Playwright Chromium includes all twelve routes, choices, prerequisites, conditional lists, completion/review, migration, history, restart, and five scheduler cases.
It also checks documentation copying, fields, fonts, formulas, themes, source hashes/anchors/URLs, keyboard scrolling, and code contrast of at least 4.5:1.
Examine desktop 1536x1024 and mobile 390x844 screenshots independently.
Use the Browser plugin when available. Otherwise record the reason for Playwright use.

If the pinned Chromium browser is absent, install it from `code-journey/`:

```bash
npx playwright install chromium
```

`JOURNEY_URL` selects the test URL.
`JOURNEY_QA_DIR` selects external screenshot storage. The default is a temporary directory.
Generated `dist`, `.tools`, `node_modules`, `data`, and browser scratch are ignored.
The repository-wide checks stay mandatory.
