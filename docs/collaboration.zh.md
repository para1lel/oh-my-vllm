# Codex 与 OMP 协作

Codex 负责架构, 项目集成, pull request 审查, GPU 执行和端到端验收.
OMP 在独立 checkout 实现分配的细节, 并编写完整的英文与中文文档.
保持[有效需求](requirements.zh.md), 数值容差, checkpoint 字节和绑定源码的证据不变.

## Issue 与审查

发现新问题时创建 GitHub Issue, 记录触发条件, 观测结果, 验收条件和负责的协作者.
每个 OMP Issue 使用一个限定范围的 `omp/issue-N-description` 分支.
共用 GitHub 账号, 评论标注 `[Codex]` 或 `[OMP/deepseek-flash]`.
任务分配, 发现和修正通过 Issue, pull request 及其评论讨论.

OMP 每次里程碑提交前请求 Codex 独立审查.
修正发现, 更新双语交接, 并保持 hooks 启用完成适用检查.
只暂存有意修改的路径. 保留[贡献规则](../CONTRIBUTING.zh.md)要求的提交署名.
推送 OMP 分支, 并创建 draft pull request.

合入前, 将 pull request rebase 到当前 `main`.
Rebase 后重跑受影响的检查. 审查最终 head 和 base 身份.
将其中的提交 squash 为一个新提交, 唯一父提交为当前 `main`.
通过 fast-forward 推进 `main`, 并及时同步 GitHub.
若 `main` 改变, 合入前重新 rebase, 检查和审查.

问题修复经审查, 合入且具有适用证据后关闭 Issue.
尚未完成的最终源码验收保持开放.

## 模型与派发

协作者使用 OMP `18.10.3` 和官方 `deepseek/deepseek-flash`.
默认思考等级为 `high`. 复杂 CUDA 任务可以使用 `max`.
凭据保留在用户凭据库. 不将它们放入源码, 命令参数, 日志, Issue 或 pull request.

文本角色使用官方模型. 禁用跨提供方的重试 fallback.
本开发流程不启用图像, 搜索模型, 语音和听写角色.
主机配置, 代理选择和 checkout 路径保留在被忽略的 `LOCAL.md`.
独立 checkout 使用独立的构建和运行时缓存输出.

API 不设费用上限.
[官方峰时](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)按北京时间确定: 工作日 09:00–12:00 和 14:00–18:00, 法定节假日除外.
[派发策略](../development/collaboration.py)在每个新 Issue 任务开始前立即检查 `Asia/Shanghai` 时段.
工作日节假日未核实前, 保守地维持峰时阻止规则.

OMP 可以跨定价时段完成已有 Issue. 恢复任务必须使用已记录的原生会话.
峰时的新工作由 Codex 承担.
本策略限制任务启动, 不证明提供方如何对跨时段请求计费.
本地模型的 OMP 验收客户端不使用此付费提供方的派发策略.

通过环境包装脚本使用[入口](../scripts/omp-collaborator.py):

```bash
scripts/with-env.sh python scripts/omp-collaborator.py --checkout "$OMP_CHECKOUT" --issue 4
scripts/with-env.sh python scripts/omp-collaborator.py --checkout "$OMP_CHECKOUT" --issue 4 --continue-task "Apply the reviewed corrections on Issue 4."
```

控制器验证仓库和分支范围, 防止并发派发, 并保留进程出生身份和原生会话记录.
GitHub 访问后再次检查范围和干净状态, 启动前再次检查定价时段.
任务自然完成或控制器失败后, 独立监督进程验证后代清理.
代理配置来自环境或 `OH_MY_VLLM_GITHUB_PROXY`, 主机值保留在 `LOCAL.md`.

原始任务日志保留在 checkout 被忽略的 `.local` 目录中.
运行后停止自有 GPU 进程和临时服务. 保留用户明确要求的 CPU 教程预览.
