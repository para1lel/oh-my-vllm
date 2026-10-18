# Contribution procedure

Read the root [AGENTS.md](AGENTS.md) and the applicable directory instructions before changes.
Use the [development guide](docs/development.md) to install the tools and hooks.

## Code

Use names that show what each function and module does.
Use a comment only for information that the code cannot give.

Keep necessary safety, mathematical, and external-contract information.
Put blank lines between functions and module definitions. Keep related imports together.

Use the repository formatters. Use different dependency files for development and inference.

## Documents

Put user information in README. Put rules for all directories in the root guides.
Put local details in the related directory guides.

Write English documents in ASD-STE100. Obey the [document rules](docs/AGENTS.md).
Keep each English Markdown document at most 2000 English words.

Supply its full Chinese translation in the same directory and change.
Keep this procedure applicable to other contributors. Do not add chat history.

## Tests

Keep a small set of behavior tests.
Use tests for failure conditions. Do not copy the implementation into expected values.

Before you add a test, tell the maintainer why it is necessary.
Tell the maintainer why the other tests are not sufficient.
Agents do the applicable tests in the repository for the maintainer.

Keep test reports and logs external to the repository.
Update the required performance tables in the two README files before each delivery.
Do not add other result tables or pass/fail histories.

## Delivery

Get a code review from another agent. Correct the errors before delivery.
Do the applicable hooks in the configured development environment.

Keep the hooks active. Do not use `--no-verify`.
Stage the files for the change. Do not use `git add .`.

Use a Conventional Commits title. Give a short description of the change.
Update the [handoff](docs/handoff.md) with the work state and next steps, without test results.

Commit the change. Push the working branch to GitHub after each delivery.
Stop the workers and temporary services that you start. Put host settings in the ignored `LOCAL.md` file.
