# 仓库协作

Codex 使用 `refactor`. OMP 在不同 checkout 中使用分配的 Issue 分支.
OMP 工作使用项目 Issues 和 pull requests.

每个里程碑都由另一名 agent 做代码审查.
发布前修正错误.

将 Issue 分支 rebase 到 `refactor`.
重新执行适用的检查.
Squash 已审查的变更.
通过 fast-forward 更新将 `refactor` 移到已审查的变更.

维护者允许项目 Issues, OMP 分支和 pull requests, 以及已审查里程碑的发布.
在其他仓库工作或进行其他发布前, 先取得维护者明确指示.
现有 TileFoundry 适配分支是允许的例外.

Git 修订版本, hooks 和文档使用[贡献流程](../CONTRIBUTING.zh.md).
主机设置和临时服务的所有权保存在被忽略的本地配置中.
