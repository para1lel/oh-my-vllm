# Learning and document contracts

## REQ-LEARN-001: Interactive tutorial

Maintain a Twine-engine tutorial in `code-journey/` with thirteen full Chinese chapters.
Include core inference logic, decisions, and implementation through project source.
Use reading choices for interests and internal prerequisites.
Give a description of displayed fields, parameters, and variables. Assume basic Rust and Python syntax.

Use LXGW WenKai, Fira Code Nerd Font, KaTeX, and 17/16px desktop/mobile prose.
Keep theme selection and reading/experiment progress.
Use halfwidth Chinese punctuation and the requested spaces. Apply Humanizer-zh.
Supply the user-requested preview on port 18084.

Keep its server address and process details in `LOCAL.md`.

Use conditional unordered lists for each page transition.
Only an explicit completion choice completes a chapter.
Review/map visits keep reading progress. Old short passages migrate to visits.
Extract current source with attached documentation, attributes, and decorators.

Remove the leading indentation that is the same on all lines that contain text.
Keep relative indentation, line numbers, and full-file hashes.
Use identical displayed and copied excerpts, readable highlighting, full-source pages, and keyboard scrolling.
Reject missing source coverage, anchors, and record-field explanations at build time.

Run tests for thirteen routes, prerequisites, choices, completion, migration, restart, refresh, and backward history.
Run tests for five Rust CPU traces: 784, 785, 1568, 1569, and 32768 tokens.
Run tests for fonts, formulas, fields, console health, desktop/mobile layout, and the two themes.
Run tests for excerpt documentation, dedentation, clipboard paths, source hashes, line anchors, and URLs.
Check code contrast of at least 4.5:1.
Examine screenshots against the requested reading style.

Synthetic CPU feedback shows scheduling. GPU evidence has its own acceptance records.

## REQ-DOC-001: Documents and portability

Write project-owned English Markdown to ASD-STE100 Issue 9 rules and dictionary.
Maintain full Chinese translations, a technical glossary, automated checks, and semantic review by another agent.
Keep useful current information and key historical decisions.
Keep host details in ignored `LOCAL.md` and original evidence in ignored local storage.
Use environment variables and CLI parameters consistently across Rust, Python, scripts, and tests.

Clean the current tracked content. Keep Git history.

## REQ-OUT-SCOPE-001: Future scope

Current scope excludes multimodal inputs, CPU KV swap, multiple GPUs, tensor parallelism of more than one, gRPC, LoRA, and production packaging.
Document extension points without empty interfaces or untested claims.
