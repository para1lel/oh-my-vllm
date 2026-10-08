# 当前状态与开放工作

更新: 2026-10-08.

## 当前实现

独立运行时为单张 B200 上的目标模型提供 Rust 服务 / 调度 / 逻辑 KV 和 Python GPU 执行.
CUDA 是项目默认后端; TileLang 冻结用于比较.
Semantic IR 在手动 graph 生命周期内编译 prefill, target decode, MTP draft 和四步 proposal.
Block784, 普通 FP32 状态, MTP BF16 状态和现有功能继续有效.
12 章 Twine 教程覆盖当前核心源码, 按用户要求保留端口 18084 预览.
地址, PID 和维护命令记入被忽略的 `LOCAL.md`.

[验收索引](acceptance.zh.md) 将每份历史测量绑定到对应源码.
最近完整 147-case / 12-row 性能结果适用于 `619c9d9`.
后续 IR 正确性修改通过 549 个 GPU 测试, 70 个 subtest 和 308 个 CPU 测试, 未重新完整采集性能.

## 文档与可移植性重构

项目英文 Markdown 采用 ASD-STE100 Issue9 规则, 配有项目技术术语表和中文译文.
架构包含设计; 本文件包含原计划.
内核开发合并 CUDA/TileLang 指南.
审计保留剩余问题和 48 项修复索引.
验收文档为统一证据索引.
保留关键 ADR 决策和替代关系.
逐日修复经过保留在 Git 历史.

环境选择使用显式 prefix, 已激活 virtual/conda 环境或仓库 `.venv`.
模型位置在执行工具中统一使用 `OH_MY_VLLM_MODEL` 或显式 `--model`.
Rust library launch 在启动子进程前拒绝空模型.
完整 GPU 测试在选择 GPU 前拒绝缺失模型配置.
CPU 模型集成明确跳过; 纯校验测试继续执行.

全部 66 份原跟踪证据有字节不变的被忽略本地副本, SHA-256 校验通过.
跟踪的派生摘要保留数值测量, 版本和原始哈希, 移除主机身份和原始 payload.
正式比较器拒绝派生 envelope, 保留现有门槛.
新服务器必须测量自己的匹配基线.
本机配置和原始证据由 Git 忽略.

## 本轮验证

Rust workspace 通过 124 个测试. 格式, 100 字符行宽, Clippy 和 release 构建通过.
Python, 脚本, 基准, 测试和开发工具的 Ruff 格式 / 检查通过.
最终 CPU 套件通过 329 个测试和 70 个 subtest, 排除 241 个 GPU 测试.
最后一次 checker 修改后, 其 13 个针对性测试再次通过.

未设置模型时, 所选测试通过 6 项和 2 个 subtest, 33 项 checkpoint 集成明确跳过.
两个 GPU smoke test 各从目标 checkpoint 生成 64 个 token.
普通执行通过环境变量选择模型.
MTP4 使用显式 CLI 选择, 同时将模型环境变量设为刻意无效的值.
真实模型执行验证了预期的选择优先级.

教程重建, Node 节选测试, 8 个 Playwright 测试和 Rust trace 测试通过.
Trace 格式和 Clippy 通过.
Browser plugin not available: 使用已配置的 Playwright Chromium 验证浏览器.
桌面 / 移动截图, 两种主题, 源码链接, 导航, 公式, 字段和调度实验检查通过.
按用户要求保留端口 18084 静态预览. 全部 smoke worker 已退出并释放 GPU 资源.

其他代理审查全部 27 对英中文档, 词典含义, 技术词性和实现合同.
还将全部 66 份归档和 64 份派生 JSON 与源字节及哈希核对.
基线提取复现全部 12 行哈希.
提交 hook 检查配对, 受保护命令, 数字, 术语定义, 链接, STE 机械规则和主机信息.
批准含义及完整翻译忠实度仍需审查, 见 [写作规则](writing.zh.md).

完整 GPU 回归套件, 6 项最大上下文运行, 147-case 算子采集和 12-row 性能采集未重新执行.
推理计算未改变. 本轮修改文档, 部署选择, 测试配置和证据处理.
当前源码性能仍需新一轮完整采集. 历史验收保留对应的实测源码与配置范围.

## 开放工作

- 用户审阅扩展教程, 修正阅读中发现的具体教学缺口.
- 按 [审计](audit.zh.md) 的 `PY-04`, 在实测显存压力下测试跨形状 262k graph eviction/recapture.
- 在扩大长期 worker 支持结论前, 证明持续 shape 变化下 compiler storage 有界.
- `PY-06` host overlap 工作以测量为依据; pinned readback 实验没有明显完整调用净收益.
- `KRN-06` 候选继续撤回, 直到新的完整操作证据支持.
- 在独立需求修改中设计 roofline 门槛, 与 vLLM 解耦.

现有 95% 吞吐, 110% TTFT 和 10% spread 门槛继续有效.
后续实现任务前阅读 [需求](requirements.zh.md), [测试](testing.zh.md) 和 [审计](audit.zh.md).
