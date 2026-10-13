# Document ownership

Use [writing rules](writing.md) and [technical terms](glossary.md).
Maintain the full same-directory `<stem>.zh.md` translation with each English source change.
Keep commands, identifiers, numbers, evidence scope, and limits equal. English is authoritative.

Exclude third-party sources and generated dependency content.

## Disclosure and sources

README supplies user setup and entry links. Root AGENTS supplies shared principles. Directory AGENTS supplies local contracts.
LOCAL supplies current host configuration and maintenance entry links. Put maintenance history in separate ignored records.
Keep each owned English document at most 2000 English words.

Count headings, prose, tables, link labels, and inline identifiers. Exclude fenced code, link destinations, and source-anchor metadata.
Give background, mathematics, observable behavior, and design decisions. Link code for implementation details.
Use file links for whole-file references. Use a unique text anchor for a specific implementation.

Generate displayed line numbers.
Keep one authoritative source for each fact. Do not repeat API signatures, defaults, or tuning parameters from code.
Historical evidence must not claim current-source acceptance. Keep obsolete repair narratives in Git history.

## Checks

Run document checks and source-link updates through the environment wrapper.
The hook checks local links, unique source anchors, word limits, and translation contracts.
A reviewer must examine technical meaning and full translation fidelity.

Update requirements, architecture, decisions, test commands, and handoff when their contracts change.
