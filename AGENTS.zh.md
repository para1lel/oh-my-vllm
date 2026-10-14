# 协作原则

构建面向单张 B200 的 Qwen3.8-27B-FP8 推理框架.
Rust 负责服务, 调度和逻辑 KV. Python 负责 GPU 计算.
进程之间使用 ZMQ DEALER 和 msgpack.

## 不变量

保持现有功能, 数值容差和[有效需求](docs/requirements.zh.md).
保持 block 大小 784, 16 个 FA 层, 48 个 GDN 层, `mamba_cache_mode="align"`, 以及 262144 个总 token.
不修改 DSpark 权重, 不引入 vLLM 依赖, 环境或缓存.

Torch 不提供的算子使用仓库 CUDA 实现. 允许原生 CUDA 库.
开发工具与推理依赖分开.

## 工作流程

读取 Git 状态, 然后读取[交接](docs/handoff.zh.md)和适用的目录指令.
Codex 使用 `main`. OMP 在独立 checkout 使用分配的 Issue 分支.
使用满足契约的简单改动. 为关键行为添加测试, 不测试实现细节.
每个里程碑都由独立 sub-agent review, 修正发现, 更新交接, 然后提交有意修改的路径.

遵循[贡献规则](CONTRIBUTING.zh.md). 保持 hooks 启用.
用户已授权项目 Issue, OMP 分支和 pull request, 以及经审查里程碑的发布.
遵循[协作流程](docs/collaboration.zh.md): rebase, 重跑检查, squash, 再 fast-forward `main`.
其他仓库及发布操作仍需用户明确指令.
现有 TileFoundry 适配分支是已许可的例外.

## 必需检查

使用[开发命令](docs/development.zh.md#构建与检查)和适用的[测试](docs/testing.zh.md).
性能声明须有测量依据. 报告跳过的检查和验证范围限制.

## 所有权

使用环境和 GPU 包装脚本. 每次运行或交接后停止自有 GPU 进程和临时服务.
只有用户明确请求才允许服务继续运行. 在被忽略的 `LOCAL.md` 和交接中记录.
主机配置放在 `LOCAL.md`, 原始 trace 和实验放在外部存储, 当前文档保持简短.

英文文档及其中文配套文档保持相同范围.
