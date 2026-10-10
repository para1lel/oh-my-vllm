# 协作规则

## 目标与约束

为单张 B200 GPU 上的 Qwen3.8-27B-FP8 构建 Rust 推理框架.
Rust 负责服务, 调度和逻辑 KV 缓存; Python 负责 GPU 计算.
边界使用 ZMQ DEALER 和 msgpack.
保留全部现有功能和数值容差.

- 生产路径的块大小保持 784 token.
- 保持 16 层 FA 和 48 层 GDN, 使用 `mamba_cache_mode="align"`.
- 支持输入与输出合计 262144 token.
- 普通和 MTP4 边界测试覆盖 batch 1, 2, 4, 无 OOM 和重计算抢占.
- 保留 [需求](docs/requirements.zh.md) 中全部有效验收条件.
- 项目构建, 测试和推理独立于 vLLM 代码, 环境和构建缓存.
- 项目内核使用 CUDA; 保留冻结的 TileLang 比较实现.
- 自有 CUDA 后端代码文件至多 800 行, 使用 clang-format 钩子.
- 正式算子用例, 测试框架和汇总证据保留在仓库内.
- 临时实验和原始 profiling trace 放在仓库外.
- 未来架构, 多 GPU 和其他 NVIDIA 后端只写扩展说明, 不声称未经测试的支持.
- 不要修改 DSpark 权重.

按依赖顺序选择近期兼容的稳定版本.
固定经过验证的组合.
在项目环境中使用 `uv` 安装 Python 依赖.
使用可用的系统 CUDA / 编译器工具或环境内工具; 驱动由主机提供.
TileFoundry 是来自固定 fork 的开发依赖.
简单的 fork 兼容性问题直接修复; 大幅修改先与用户讨论.

## 环境与 GPU 管理

[开发指南](docs/development.zh.md) 说明可移植的环境选择.
运行模型前设置 `OH_MY_VLLM_MODEL` 或传入 `--model`.
Cargo 和 Python 命令通过 `scripts/with-env.sh` 调用.
包装脚本默认按仓库位置设置 `PYTHONPATH` 和 `CARGO_TARGET_DIR`.
实际主机路径和维护记录放在被忽略的 `LOCAL.md` 中.

使用 `scripts/with-gpu.sh` 等待空闲 B200 并绑定其 UUID.
每次运行结束, 失败, 取消或交接后立即停止任务启动的 GPU 进程及子进程.
通过进程记录和 `nvidia-smi` 确认退出.
只停止任务所属的进程.
只有用户明确要求时才保留服务运行.
在 `LOCAL.md` 记录例外, 在交接文档说明用途.

## 任务流程

1. 阅读 `git status --short`, 识别已有修改.
2. 确认当前分支为 `main`.
3. 阅读 [交接](docs/handoff.zh.md) 和适用的目录规则.
4. 阅读下表中的任务指南.
5. 完成已授权修改及相关测试.
6. 每个里程碑后, 安排另一子代理进行审查.
7. 提交前修复正确性, 性能和工程规范问题.
8. 更新交接文档, 说明结果, 验证限制和剩余工作.
9. 只暂存有意修改的路径.
10. 按 [CONTRIBUTING](CONTRIBUTING.zh.md) 的格式提交到 `main`.
11. 确认任务所属 GPU worker 和临时服务端口已释放.

不创建分支或依托分支的 worktree.
已有 TileFoundry 适配分支是用户授权的例外.
不为该 fork 创建上游 PR.
未经明确指示, 不推送或创建 PR.
不使用 `git add .` 或禁用钩子.
不提交秘密, 环境值或原始 GPU trace.

## 必需检查

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
```

按 [测试指南](docs/testing.zh.md) 执行适用的 Python, GPU, 教程和验收检查.
新增功能需要测试, 或记录无法测试的原因.
性能结论必须来自实际测量.
提交正文说明跳过的检查.
Rust 行宽硬限制为 100 字符, 包括宏和字符串.
每个 `unsafe` 块需要 `// SAFETY:` 注释.

## 文档

英文 Markdown 原文及同目录 `<stem>.zh.md` 完整译文在同一次修改中维护.
代理以英文原文为准.
两个语言版本的命令, 标识符, 数字, 证据范围和限制保持一致.
排除第三方子模块和生成或 vendored 依赖.
遵循 [写作规则](docs/writing.zh.md) 和 [技术术语表](docs/glossary.zh.md).

| 变更 | 文档 |
|---|---|
| 需求 | `docs/requirements.md` |
| 架构 | `docs/architecture.md` |
| 关键决策 | `docs/decisions/` |
| 任务完成或阻塞 | `docs/handoff.md` |
| 构建或测试命令 | `docs/development.md`, `docs/testing.md` |
| 提交约定 | `CONTRIBUTING.md` |

| 任务 | 优先阅读 |
|---|---|
| 任何任务 | 本文件和 `docs/handoff.md` |
| 调度或 KV 缓存 | `docs/architecture.md`, `docs/decisions/` |
| Python runner | `docs/architecture.md` 的 Python 职责 |
| 性能 | `docs/requirements.md`, `docs/profiling.md`, `docs/testing.md` |
| 新功能 | `docs/requirements.md`, `docs/architecture.md` |
| 故障 | `docs/handoff.md` 的开放工作和 `docs/testing.md` |
| 审计问题 | `docs/audit.md` |
