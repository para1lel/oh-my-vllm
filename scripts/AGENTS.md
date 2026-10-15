# Repository tools

Put inference code and its dependencies in other directories.
Use the Python standard library unless a parser is necessary for syntax analysis.

Keep each checker on its stated contract.
Use syntax trees to find code definitions. Do not use text patterns for this operation.

Use clear names and small functions to give the logic.

Use the same environment wrapper for hooks and direct commands.
Process owned tracked or unignored files only.

Do not process third-party trees, symlinks, generated files, or paths beyond the repository.
If an optional source directory does not exist, there are no applicable files in that directory.

If an applicable input has an error, stop the check with an error.
If a necessary tool is missing, stop the check with an error.

Obey the [test policy](../CONTRIBUTING.md) before you add tests.
Do not save check results in repository documents.
