# 当前状态与开放工作

更新: 2026-10-08.

## 当前实现

独立运行时为单张 B200 上的目标模型提供 Rust 服务 / 调度 / 逻辑 KV 和 Python GPU 执行.
CUDA 是项目默认后端; TileLang 冻结用于比较.
Semantic IR 在手动 graph 生命周期内编译 prefill, target decode, 原生 MTP4 和 DSpark.

Block784 和现有功能继续有效. 普通执行保留 FP32 GDN 状态.
MTP4 与 DSpark 保留 BF16 GDN 状态, 遵守舍入和快照合同.

进程在加载模型前选择 `--speculative-mode none`, `mtp` 或 `dspark`.
原有 `--num-speculative-tokens 0/4` 接口保持兼容.
普通执行每个 decode step 从目标分布采样一个 token.
MTP4 提出 4 个 token 并与目标采样比较, 最多使用 5 个目标 query.
保留匹配的草稿前缀及下一个目标 token.

DSpark 最多提出 7 个 candidate, 最多使用 8 个目标 query.
5 层草稿使用从 0 开始编号的目标 feature `(5,19,33,47,61)`, context injection 和 Markov head.
Greedy 验证保留匹配的草稿前缀及下一个目标 token.

随机采样验证使用完整条件分布和 `min(1,p/q)` 接受.
拒绝采样使用归一化的 `max(p-q,0)`, 全部接受后从目标分布采样一个 bonus token.
Grammar mask 和采样设置适用于每个草稿前缀.
仅保留的行进入持久 DSpark context. 拒绝的目标行回滚已计算进度与 GDN 快照.

`OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model` 指定可选 checkpoint.
加载检查配置, shape, BF16 dtype 及实际配置 / 权重哈希.

用户可以设置 confidence threshold. 使用不同文本 / 工具历史输入选值后, 初始设置为 `0.2`.
设置为 `0.0` 保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.
Confidence 可从首个 candidate 停止, 返回 0 个草稿, 该步由目标执行单 token 解码.

合成输入诊断提供调参证据, 其范围与该选择过程及正式验收不同.

Twine 教程源码有 13 章, 含 DSpark 章节.
按用户要求保留的端口 18084 预览使用当前构建, 包括新增 result mode 字段和 confidence 设置.
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

## Confidence 阈值选值

`calibration-03` 使用项目文本及带源码的工具历史, 与正式合成 token ID 分开.
每个请求有 32768 个输入 token 和 512 个保留输出 token, batch 含 2 个请求.
每项设置使用 greedy 采样, 忽略 EOS, 以及 `thinking=off` 模板.
每项设置在单张 B200 上执行 2 次完整预热和 2 次测量.

| 阈值 | 输出 TPS 中位数 | 接受 / 已调度草稿 |
|---|---:|---:|
| `0.0` | 177.287 | 28.487% |
| `0.05` | 183.097 | 44.957% |
| `0.1` | 184.198 | 53.406% |
| `0.2` | 184.323 | 62.5% |

接受率使用测量行的已接受草稿总数除以实际验证草稿总数.
实际验证草稿是已调度 candidate, 与返回的下一步 proposal 分开计数.

本项目从这组小样本选择 `0.2`, `0.1` 与 `0.2` 的吞吐接近.
选值证据覆盖这 2 个输入和 2 次测量, 正式采集必须测量其他负载的性能及稳定性.
日志, 语料 / 源码哈希和调参记录保留在外部存储.

## DSpark 实现里程碑验证

Rust workspace 通过 130 个测试. 实现过程中, 格式, 100 字符行宽, Clippy 和 release 构建通过.
Python, 脚本, 基准, 测试和开发工具的 Ruff 格式 / 检查通过.
最新 CPU 采集通过 513 个测试和 70 个 subtest, 排除 285 个 GPU 测试.

后续针对性测试覆盖采样, graph family, 源码身份, 测量区间和服务证据.
针对性的 benchmark / context / service 检查通过 120 个 CPU 测试, 排除 9 个 GPU 测试.
外部 supervisor 的 11 个 CPU 合同测试通过.
也覆盖 OMP 源码范围, 响应持久化与信号 handler 恢复期间的取消.

GPU 测试通过 57 项新增及已有 compiled case, 4 项 supplemental reference case 和 3 项 context graph case.
Native attention bucket 的 2 项 case 在 eager 和 compiled 模式下通过, 覆盖变化的 metadata, FP64 参考和 cache canary.
Eager 与 compiled DSpark smoke run 各保留 64 个输出 token.
重复前缀诊断在 100 次重复中保持 784-token prefix hit.
这些检查不能替代完整算子, 上下文, 模型及性能采集.

首次完整 GPU 尝试通过 66 个测试和 4 个 subtest, 有 2 个失败 case.
其中 DSpark batch1 边界实际完成 258048 个输入 token 和 4096 个输出 token, 无 OOM 或重算抢占.
通用 benchmark result 遗漏 mode 字段, 导致校验器拒绝该行.
另一失败来自停止该尝试时中断的普通 batch2 运行.
Benchmark result 现已记录 `speculative_mode`.

第二次完整 GPU 采集通过 285 个测试, 513 个 CPU 测试未选中.
ordinary / MTP4 / DSpark 的 9 项边界 case 在 batch 1, 2, 4 下通过, 没有 OOM 或重算抢占.
GPU runtime 源码保持在实现版本 `46f7529`.

首次 DSpark 服务尝试通过 12 项约束 case 和 Chat 生命周期 case.
OMP Chat client 约 250 秒后以状态 0 退出.
原证据校验器将 OMP 行号范围后缀当作文件名的一部分, 因而拒绝了成功的源码读取.
修正后找到文件名及边界值大于 0 的行号范围, 检查指定的 `schedule()` 声明, 保留全部回传源码读取范围.

原 JSONL 和 HTTP 记录的 CPU 重验通过, 最终回答经人工语义审查.
原失败尝试保持不变, 独立结果绑定原始哈希和修正后校验器哈希.
在重验范围内, 已证明客户端读取源码并将工具结果传回后续模型请求.
首次服务尝试在启动后记录源码身份, 属于诊断证据, 不构成当前源码的服务验收.

新服务 supervisor 在启动前及全部客户端工作与清理后记录源码 / binary 身份.
验证监听器归属和实际 mode / threshold, 保留失败, 检查后代 / GPU / 端口释放.
客户端在后续断言前保存收到的响应.
取消时保留失败结果, 覆盖响应持久化及信号 handler 恢复期间.
首次服务 worker 已退出并释放 GPU 资源及服务端口.

第二次服务尝试在源码 `46f7529` 上通过 12 项约束 case 和 Chat 生命周期 case.
OMP Chat client 完成 12 次工具调用, 以状态 0 退出.
它读取的 scheduler 范围在 `schedule()` 定义之前结束, 源码内容检查失败.
任务文字要求读取 scheduler 定义和 worker class, 必要时使用显式行号范围.
校验器保留源码内容和工具结果检查.

每次尝试保留各自原始记录.
第二次服务释放全部所属进程, GPU 资源和端口, 源码身份保持相同.

第三次服务尝试在源码 `7e2d618` 上通过 12 项约束 case 和 Chat / Responses 生命周期 case.
OMP Chat 通过工具结果检查. OMP Responses 完成 12 次工具调用, 以状态 0 退出.
Responses 校验失败, 因为 OMP 在客户端记录中保留 `call_id|item_id`, 在 HTTP 请求中保留 `call_id`.

校验器把这些 ID 与记录中的工具调用, read 路径和输出比较.
HTTP 请求含有 item ID 时, ID 必须一致.
校验器拒绝能对应多个已记录工具调用的 ID.

原始 Responses 记录通过 CPU 重验, 失败尝试保留原有结果.
重验在客户端运行后计算服务日志 offset.
新服务运行必须在客户端启动前记录 offset.

这次尝试未运行长上下文服务 case.
源码保持相同. 全部所属进程, GPU 资源和端口已释放.
选定服务检查通过 63 个 CPU 测试.

第四次服务尝试在源码 `fb432c6` 上通过 12 项约束 case.
两个短请求的共享 batch 检查失败, 该次尝试的内容, 存储, 取消和释放检查均没有错误.
测试让两个客户端同时开始, 并要求较长输出.
同一 batch, 输出内容, 取消和释放检查继续有效.
使用 GPU 服务运行此测试, 证明共享调度.

源码保持相同. 第四次服务释放全部所属进程, GPU 资源和端口.

较早的 13 章教程构建通过 10 个 Node 测试, 9 个 Playwright 测试和 Rust trace 测试.
Trace 格式和 Clippy 通过.
Browser plugin not available: 使用已配置的 Playwright Chromium 验证浏览器.
桌面 / 移动截图, 两种主题, 源码链接, 导航, 公式, 字段和调度实验检查通过.

最近一次教程构建通过 10 个 Node 测试和 9 项浏览器测试.
37 个源码摘录锚点和 71 个源码覆盖条目与当前源码哈希一致.
桌面 / 移动亮色 / 暗色截图覆盖 confidence 公式, 没有中文字形缺失.
按用户要求保留端口 18084 静态预览. 全部 smoke worker 已退出并释放 GPU 资源.

其他代理审查 mode 接口, checkpoint 加载, 采样, 精度, cache 回滚, graph, 基准和服务证据.
文档 hook 检查配对, 受保护命令, 数字, 术语定义, 链接, STE 机械规则和主机信息.
批准含义及完整翻译忠实度仍需审查, 见 [写作规则](writing.zh.md).

Responses 服务验收, 长上下文服务验收, 230-case 算子采集和 12-row 性能采集尚未完成.
3-row DSpark / MTP4 配对比较和 9 项 release-binary 上下文边界运行也仍待执行.
当前源码性能仍需新一轮完整采集. 历史验收保留对应的实测源码与配置范围.
按用户要求, 这些 DSpark 比较统计没有验收门槛.
原有 12 项 workload 门槛及正确性, 显存, 重算, 稳态和证据合同继续有效.

## CUDA kernel 验证阶段

DSpark CUDA 更新通过 49 项 GPU 测试, 排除 5 项 CPU 测试.
其中包含 20 项新增 layout, 地址和 cache-write 用例.
选定的 fixture, 输出, 比较和 provenance 检查通过 35 项 CPU 测试, 排除 10 项 GPU 测试.
全部 8 项仓库 hook 通过.

另一 agent 检查 BF16 舍入, NeoX 配对, slot 范围, 地址除法以及 cache write 之前的源读取.
诊断显示, 不同目标地址改变 append 带宽.
地址交换前参考实现耗时较少, 地址交换后 CUDA 耗时较少.
Append 采集器现先校验独立缓存, 再使用相同缓存计时.
冻结参考, 最初 147 项用例 ID 和 3 轮统计门槛保持不变.

此前基于 `1bcf5d5` 的服务采集通过约束, Chat / Responses 生命周期, 两个 OMP 客户端和长上下文前缀复用.
全部任务 worker 和服务端口已释放, 语义审查在外部证据中记录回答错误及引用限制.
后续完整采集将使用已提交的 CUDA 更新.

## 设备计时验证

基于 `353c94a` 的完整算子采集通过 230 项数值检查.
两项统计速度检查未通过. Append 的 CUDA 耗时在三轮均较少.
Q/K 的 CUDA 耗时在第一轮较多.
完整失败采集保存在外部目录.

`353c94a` 的计时器通过不同主机调用在 graph 提交前后记录 event.
主机提交停顿可能进入耗时, 已记录的尖峰尚不能确定其原因.
新计时器在每个 graph 中的全部 100 次操作前后记录 external event.

对照在 replay 前后各插入 10 ms 主机停顿.
旧计时器的单次操作耗时从 2.18576 变为 202.39872 微秒.
新计时器从 1.63360 变为 1.64576 微秒.
Graph 检查发现两个 event node 位于完整的 100 个 kernel node 链两端.

首次诊断的 node counter 使用错误的 DOT 标签模式, 断言失败.
独立校验保留原始记录, 核对 graph 链和哈希.
两项 GPU 测试通过, 分别在 replay 前或后插入 20 ms 主机停顿并检查完整 graph node.

Runtime 文件和已加载 CUDA 模块保持与 `353c94a` 相同.
该版本服务采集通过全部六组客户端, 包括两个 OMP 工具循环及长上下文前缀复用.
全部服务进程和服务端口已释放.
后续正式计时将使用包含修正采集器的 clean commit.

## 短输入 Q/K 验证

基于 `49579b4` 的算子采集通过 230 项数值检查和 229 项速度检查.
一个 Q/K 案例一轮的 CUDA 中位耗时比 TileLang 中位耗时多 1.12 纳秒.
完整失败采集保存在外部目录.

新的短输入 Q/K 路径使用第一组乘法结果, 不再先与零相加.
后续 FP32 累加及 BF16 输出舍入保持相同.
长输入路径和 fallback kernel 保持相同.

全部六个 Q/K 诊断案例使用生产模块通过数值和速度检查.
新的数值检查包括 624, 1023 和 1024 行及负零.
二十项 GPU 检查通过, 包括 FP32 下溢, 溢出, 非有限值, 以及 FP64 Q/K 和 GDN 参考.
完整算子与服务采集将使用新提交源码和已加载模块.

## Target graph 预算与 Q/K 计时

Batch 4 成对采集中的共享 32 条目 target graph cache 出现驱逐和再次 capture.
中止的采集保留原始记录. 源码未变, 所有任务 worker 已释放.

普通 / MTP target 预算保持 32 条目. DSpark 使用 64 条目. 比较 worker 使用 96 条目.
比较 family 下限分别为 target 16 和 target feature 32. 这些下限表示最低保护数量.

显存余量检查保持 4 GiB. 新正式测量必须证明测量阶段没有 capture 或编译.

Q/K 计时先用不同分配检查数值输出.
完整无状态函数随后使用同一个 graph pool 和 stream, 每次捕获调用的对应目标地址相同.
每次完整 Q/K 调用分配两个输出. 采集器检查输出与上一次输出的 alias, 以及输入存储字节.
Graph 和 stream 保留到最终同步完成. 其他操作保留原有计时合同.

6 种正式 Q/K 形状通过诊断数值与统计比较.
CPU 集合通过 539 个测试和 70 个 subtests. 重点 graph 计时 GPU 测试和全部 8 个 hooks 通过.
中止的完整 GPU 选择不作为验收证据. 任务子进程已停止.
新的服务, 成对比较, 算子, 边界和框架采集仍待使用提交后的修正完成.

## 草稿和 context graph 预算

下一次成对采集的 batch 1 和 batch 2 通过测量审计.
Batch 4 没有 target graph 重捕获. MTP draft / proposal 和 DSpark context cache 仍在测量阶段重捕获.
完整失败采集保留全部样本和计数. 所有任务 worker 已释放.

DSpark context graph 改为 32 条目, 覆盖全部合法的 1 至 32 行数.
MTP 比较 worker 使用 64 条目 draft / proposal, 下限分别为 32 和 8.
仅运行 MTP 的 worker 保持 32 条目, 下限分别为 16 和 4.
已有显存, 准入, pool 与 cooldown 合同继续生效.

重点 CPU 测试通过 137 项. 新测试覆盖全部 context 行形状, 比较 family 保留和无效预算.
重点 GPU graph 检查与全部 8 个 hooks 通过. 完整验收仍待使用提交后的修正完成.

## 开放工作

首组成对采集完成了 batch 1, 随后在绑定 batch 2 的 IPC 地址时失败.
已关闭的 Unix 监听器留下了文件系统条目. 每个 batch 现在使用临时目录中的不同地址.
CPU 回归测试依次绑定并关闭三个监听器, 保留各自的文件系统条目.
失败的采集记录保存在外部目录. 完整重测须使用已提交的修复.

- 用户审阅扩展教程, 修正阅读中发现的具体教学缺口.
- 添加选值的派生证据, 保留文本 / 工具历史来源, 哈希和已调度 / 接受草稿数.
- 在源码保持不变时完成剩余服务, 算子, 性能, 配对比较及 release-binary 上下文采集.
- 按 [审计](audit.zh.md) 的 `PY-04`, 在实测显存压力下测试跨形状 262k graph eviction/recapture.
- 在扩大长期 worker 支持结论前, 证明持续 shape 变化下 compiler storage 有界.
- `PY-06` host overlap 工作以测量为依据; pinned readback 实验没有明显完整调用净收益.
- `KRN-06` 候选继续撤回, 直到新的完整操作证据支持.
- 在独立需求修改中设计 roofline 门槛, 与 vLLM 解耦.

现有 95% 吞吐, 110% TTFT 和 10% spread 门槛继续有效.
后续实现任务前阅读 [需求](requirements.zh.md), [测试](testing.zh.md) 和 [审计](audit.zh.md).
