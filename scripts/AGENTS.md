# Repository tools

Keep these tools separate from inference code and dependencies.
Use the Python standard library unless an existing parser supplies necessary syntax handling.
Keep checks focused on their stated contract. Reuse syntax trees instead of matching code definitions with regular expressions.
Do not add explanatory comments when clear names and small functions express the logic.

Hooks and direct commands use the same environment wrapper.
Process owned tracked or unignored files. Exclude third-party trees, symlinks, generated files, and paths beyond the repository.
Treat absent optional source trees as having no applicable files.
For an applicable file, report missing tools or invalid input as failures.

Follow the [test policy](../CONTRIBUTING.md) before adding tests. Do not save check results in repository documents.
