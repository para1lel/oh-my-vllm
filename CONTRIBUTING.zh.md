# 贡献流程

修改前阅读 [协作规则](AGENTS.zh.md) 和 [当前工作](docs/handoff.zh.md).
只在 `main` 上开发和提交.
保留其他任务已有的修改.
仅在用户明确指示后推送或创建 PR.

## 实现

保持 Rust 对服务, 调度和逻辑缓存的所有权.
GPU 计算保留在 Python 和独立库中.
保留块大小, 数值容差, API 功能和有效验收条件.
选择兼容的稳定依赖, 固定经过验证的组合.
Python 依赖使用 `uv` 安装; 不直接使用 `pip`.

使用 `rustfmt.toml`, Rust 行宽硬限制为 100 字符.
限制包括宏, 字符串, 注释和展开后的制表符.
长 JSON 宏拆成字段, 长字符串使用 `concat!`.
每个 `unsafe` 块都添加 `// SAFETY:` 注释.
使用 Python 3.12 或更高版本及 `ruff.toml`.

自有 CUDA 模块使用 clang-format 23.1.3. 每个后端代码文件至多 800 行.

新增行为添加有实际意义的测试.
无法测试时记录原因.
按正式协议测量性能; 估算值明确标为估算.
临时调优和原始 trace 放在仓库外.

## 审查与检查

每个里程碑后, 安排另一子代理进行审查.
提交前修复审查发现的正确性, 性能和工程规范问题.
完成必需 [检查](AGENTS.zh.md#必需检查) 和适用 [测试](docs/testing.zh.md).
使用 `scripts/with-env.sh` 选择环境.
`.pre-commit-config.yaml` 的本地钩子使用此包装脚本.

英文 Markdown 修改时同步更新完整中文译文.
遵循 [写作规则](docs/writing.zh.md), [术语表](docs/glossary.zh.md) 和 `AGENTS.md` 的文档更新表.
主机配置记入被忽略的 `LOCAL.md`.
原始证据保存在被忽略的本地存储; 仓库提交明确标识的可移植摘要.

## 提交

只暂存属于当前修改的路径:

```bash
git add path/to/changed-file path/to/another-file
scripts/with-env.sh pre-commit run --all-files
git commit
```

提交钩子需要与手动检查相同的配置环境.
保持钩子启用; 不使用 `git add .` 或 `--no-verify`.
标题采用 Conventional Commits, 例如 `docs: make deployment guides portable`.
正文说明最终行为, 验证, 跳过的检查和剩余限制.
每次提交包含以下原样署名:

```text
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

更新 `docs/handoff.md`, 说明当前结果和开放工作.
停止所属 GPU 进程和临时监听器.
用户要求持续运行的服务, 在交接文档记录用途, 在 `LOCAL.md` 记录本机信息.
