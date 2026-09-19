@AGENTS.md

# Claude-specific notes

- Standing commit attribution: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Reply in Chinese when the user writes Chinese.
- Today's date context: session was last active 2026-09-19.
- The pre-commit hook stashes unstaged files before running formatters; always
  stage `Cargo.lock` alongside Rust source changes to avoid stash conflicts.
