# Contributing

Read the root [AGENTS.md](AGENTS.md) and applicable directory instructions before changes.
Use the [development setup](docs/development.md) to install tools and hooks.

## Code

Use names, functions, and module boundaries that make each code fragment understandable in context.
Prefer clear code to explanatory comments. Keep necessary safety, mathematical, and external-contract notes.
Separate functions and module definitions with blank lines. Keep related imports grouped.

Use the repository formatters. Keep development dependencies separate from inference dependencies.

## Documentation

Keep README focused on users. Put shared direction in root guides and local details in directory guides.
Keep each English Markdown document at most 2000 English words.
Update its complete same-directory Chinese companion in the same change.
Follow the [document rules](docs/AGENTS.md). Keep contribution instructions useful to future contributors, without chat-specific history.

## Tests

Use a small set of independent behavior tests. Cover meaningful failure modes instead of copying implementation steps.
Before adding a test, tell the maintainer what it protects and why existing coverage is insufficient.
Agents run applicable existing tests themselves. The maintainer does not have to execute tests manually.
Do not commit test reports, logs, result tables, or pass/fail histories.

## Delivery

Get an independent review and correct its findings before delivery.
Run applicable hooks with the configured development environment. Keep hooks enabled and do not use `--no-verify`.
Stage intentional paths explicitly. Do not use `git add .`.

Use a Conventional Commits title and a concise description of the resulting change.
Update the [handoff](docs/handoff.md) with current work and next steps, without test results.
Commit and push the working branch to GitHub after each delivery.
Stop owned workers and temporary services. Keep host settings in ignored `LOCAL.md`.
