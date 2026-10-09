# Document writing rules

Project-owned English Markdown follows [ASD-STE100 Issue9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf), dated 2025-01-15.
Use its writing rules and approved dictionary meanings, parts of speech, and forms.
Use the [project glossary](glossary.md) for necessary technical nouns and verbs.
The standard permits technical terms. This glossary does not authorize general-purpose synonyms.

## English procedure

Use approved words in their approved meaning and part of speech.
Use technical terms only for their defined project meaning.
Keep one term for one concept.
Prefer short, direct statements with a known subject and active voice.
Use passive voice in descriptive text only when the actor is unknown.

Use infinitive, imperative, simple present, simple past, or simple future verbs.
Use participles as permitted modifiers, not progressive or perfect tenses.
An `-ing` technical noun can name a defined operation.
Keep code syntax, commands, identifiers, messages, licenses, and source excerpts unchanged.

Limit procedure sentences to 20 words and descriptive sentences or notes to 25 words.
Keep each descriptive paragraph to one topic and at most 6 sentences.
Use lists for different actions or parallel contracts.
Count a list introduction and each item independently for sentence length.
Do not use prose semicolons.

Use approved comparison wording, such as more than/less than for quantities.

For word counts, quoted text, proper names, formulas, and numbers with units count as specified in Rule 8.
A parenthetical phrase counts as one word in its outer sentence. Examine its internal sentences independently.
For Markdown, inline code is the project's equivalent of differently typeset quoted text.
List cells and headings keep their technical labels. Meaningful sentences still have length checks.

Fenced code is protected syntax, not English prose.

## Chinese translation

Maintain `<stem>.zh.md` beside each English source in the same change.
Translate all prose and keep facts, status, conditions, uncertainty, and scope.
Keep commands, inline identifiers, IDs, hashes, numbers, and evidence limitations equal.
Translate link labels and use Chinese companions when available.
Keep the English source authoritative.

Exclude third-party submodules and generated/vendored dependencies.

Use halfwidth punctuation in Chinese prose.
Put a space before a left parenthesis and after other punctuation.
Separate Chinese text, formulas, and English words with spaces.
Source code, identifiers, URLs, and payloads keep their syntax.
Apply Humanizer-zh to remove empty phrasing and phrases that give the same information. Keep evidence strength unchanged.

Keep necessary negative constraints and limitations.

## Scope and historical information

Keep current contracts, useful extension boundaries, measured evidence, and key decisions.
Keep daily repair narratives in Git history.
Use `docs/handoff.md` for current status and next work.
Use `docs/audit.md` for open findings and a compact fix index.
Use `docs/acceptance.md` as the sole evidence index.

Put host paths, addresses, process identities, and maintenance details in ignored `LOCAL.md`.
Keep full original evidence in ignored local storage and commit explicit portable-derived summaries.
Keep original hash (SHA-256) distinct from derived-file hashes.
Record the source and configuration of each historical measurement.
Do not treat a past result as current-source acceptance.

## Automated checks and reviewer checks

```bash
scripts/with-env.sh python scripts/check_docs.py
```

The checker examines all owned tracked/unignored Markdown and the current tracked text content.
It checks translation pairs, protected code equality, explicit numeric values, local links/anchors, sentence length, paragraph size, and punctuation.
It validates glossary structure and registered forms, and rejects known unsuitable generic wording.
It rejects host-specific paths, addresses, and GPU identities.
It also checks the ignore rules for `LOCAL.md` and local evidence.

The pre-commit hook uses the same command.

Automation does not determine word meaning, part of speech, paragraph topic, or full translation fidelity.
A reviewer who did not make the change must compare English with the official dictionary and Chinese with the English source.
Examine technical-term necessity and algorithm meaning at the relevant source.
Use the official standard directly. Keep private reference copies in external storage.

A passing checker alone is not a STE conformity certificate.
