@AGENTS.md

# Claude 专用说明

- 每次提交都附带：`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`。
- 用户用中文时，以中文回复。
- 日期上下文：此记录对应的会话最后活跃于 2026-09-19。
- pre-commit hook 会先暂存未暂存的改动，再运行格式化工具；Rust 源码修改伴随 `Cargo.lock` 修改时，两者一起暂存，避免 stash 冲突。

本文件是英文版的中文译本；agent 以英文版和英文 AGENTS.md 为准。
