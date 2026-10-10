# 需求

本文件定义当前有效范围.
[验收](acceptance.zh.md) 记录实测源码身份, [审计](audit.zh.md) 记录尚存限制.

## REQ-GOAL-001: 推理目标

为单张 B200 GPU 上的 Qwen3.8-27B-FP8 提供 Rust 推理框架.
保留全部现有离线和 HTTP 功能.
每个性能工作负载必须满足 `REQ-PERF-001` 和 `REQ-PERF-002`.

## REQ-ARCH-001: 职责

Rust 负责服务, 请求队列, token 预算, 逻辑 KV 分配, 前缀复用和抢占.
Python 负责模型加载, GPU forward 计算, 内核和手动 CUDA Graph.
使用一条 ZMQ DEALER 通道传输 msgpack.
Tensor 保留在 Python 侧.

## REQ-MODEL-001: 目标

使用包含 48 层 GDN 和 16 层 FA 的 Qwen3.8-27B-FP8.
所有生产路径使用块大小 784.
FA group 为 0, GDN group 为 1, 使用 `mamba_cache_mode="align"`.
Checkpoint 位置通过 `OH_MY_VLLM_MODEL` 或 `--model` 设置.

## REQ-EXEC-001: GPU 执行

模型计算使用 CUDA Graph, 多个 CUDA stream 和 Programmatic Dependent Launch (PDL).
通过语义操作和 tensor / state 读写定义图依赖.
独立分支可以并行执行.
PDL consumer 必须先等待, 再读取 producer 输出.

保留 graph capture 状态恢复, 缓存所有权和有界的 graph 存储.
通过完整算子与框架测量调优执行.
记录每项选定优化的配置, 数值检查, 显存和时延.
通过单独测量区分 graph, stream 和 PDL 的效果.
阶段时延与算子门槛继续有效.

## REQ-PERF-001: 阶段时延

| 模式 | 输入 token | 输出 token | Batch sizes |
|---|---:|---:|---|
| ordinary, MTP4, prefix-hit, DSpark | 32768 | 4096 | 1, 2, 4 |
| ordinary | 131072 | 4096 | 1, 2, 4 |

这些组合构成十五行.
使用合成 token IDs, greedy 采样和固定输出数量, 忽略 EOS.
每次重复前重置前缀复用.
prefix-hit 行为每个请求预置 32144 个可复用 token.
整个负载保持驻留, 重算抢占次数为零.

为每个请求记录提交, 首个保留 token 和末个保留 token 的时间.
Prefill 从提交到首个 token. Decode 从首个 token 到末个 token.
每次重复分别取所有请求中最长的 prefill 区间和最长的 decode 区间.
Batch 内阶段交错时, 使用各请求自己的边界.
计入注册, 排队, 调度, Python, 传输和采样.
排除加载, 分词, HTTP 和预热.
独立报告整批耗时, 清理耗时, TPS, 接受率和 GPU 诊断.

除 prefix-hit prefill 外, 每个阶段的墙钟时延中位数最多为理论下界中位数的三倍.
墙钟时延的相对极差最多为 10%.
记录 prefix-hit prefill 的时延倍率, 墙钟时延和理论下界, 不设时延门槛.
其 10% 波动检查继续有效. Prefix-hit decode 保持三倍限制.

## REQ-PERF-002: 理论模型与测量

使用固定语义 DAG, 保留数据, token, 层和持久状态依赖.
独立分支可以并行.
CUDA stream 顺序不增加依赖.
为每个执行 step 记录有效 query, KV 长度, 状态访问, 草稿, 验证和回滚.
Kernel 拆分或融合时, 规范成本规则保持一致.

理论下界取以下限制中的最大值:

- 语义 DAG 的最长路径.
- 必需 HBM 流量除以官方单卡带宽.
- 每类共享执行资源的必需工作除以其官方峰值.

计入参数读取, scale, 模型计算, GPU 采样, KV / state 更新, 草稿和被拒绝的验证行.
使用各项运算的精度和执行资源.
算法执行七行 DSpark backbone 时, 七行全部计入.
Padding, 重复传输和无用计算单独记录为开销诊断.
这些工作不得增加理论下界.

使用单卡非稀疏 B200 峰值: FP8 Tensor 4.5 PFLOPS, BF16 Tensor 2.25 PFLOPS, FP32 75 TFLOPS 和 FP64 37 TFLOPS.
使用 HBM 带宽 7.7 TB/s.
其他已知执行资源使用公开指令速率.
记录硬件身份, 峰值来源, 运算规则和模型版本.
实测速率和经验效率系数不得增加下界.

允许理想缓存复用和融合后中间值驻留片上.
使用有限存储容量推导最少必需 HBM 流量.
在该放松模型中, 未公开的片上带宽视为无限.
识别全部 GPU 运算. 未识别的运算阻止验收.

每行执行两次完整预热和五次正式测量.
根据每次重复的执行调度重新计算下界.
Prefill 与 decode 独立应用以下两项检查:

```text
median(wall_time) <= 3 * median(theoretical_lower_bound)
(max(wall_time) - min(wall_time)) / median(wall_time) <= 0.10
```

Prefix-hit prefill 记录第一项检查的结果, 该结果不作为失败门槛.
第二项检查适用于每个阶段.

保留完整失败集和中断尝试, 附上原因.
调查失败后重新执行整组测量.
不得筛选样本.
测量中的编译, 图捕获, 干扰或源码变化使该测量失效.
每次尝试绑定源码, 二进制, 环境, CUDA module, 配置和 checkpoint 身份.
正式验证使用完整原始记录, 可移植摘要不能替代原始记录.

## REQ-PERF-003: DSpark 与原生 MTP4

比较同一最终源码和二进制中的可选 DSpark 与原生 MTP4.
使用 32768 个输入 token, 4096 个保留输出 token, batch 为 1, 2, 4.
使用相同的合成 token ID, greedy sampling 和固定输出数量, 忽略 EOS.
框架批吞吐包含注册, prefill, 调度, 传输, 采样和清理.

每个 batch 的比较 worker 共享 target 权重和物理 target 缓存.
预热前加载两种草稿模型, 每次尝试前重置前缀复用.
每个 batch 执行 3 轮.
每轮每种模式执行 2 次完整预热, 随后执行 5 个测量配对, 交替改变模式顺序.

每个 batch 报告以下统计:

- 每轮每种模式的吞吐中位数及波动.
- 每轮 DSpark 中位数与 MTP4 的差值.
- 吞吐增益的 hierarchical paired-bootstrap 单侧 95% 置信下界.

报告 TTFT 及其波动.
DSpark 速度, 波动及置信界限不增加验收门槛.
保留每次失败或中断尝试及完整原始记录.
测量期间发生编译或 graph capture 会使该次尝试失效.

记录绑定实际源码文件, 二进制, Python 环境, 加载的 CUDA 模块, 运行配置及 checkpoint 字节.
加载的 DSpark 配置和权重哈希必须与指定 checkpoint 匹配.
记录容量, allocated / reserved 显存峰值, compile / capture 审计和实际验证 / 接受的草稿数.
接受率的分母是已调度草稿 token, 下一步返回的 proposal 使用另一计数.

源码身份及测量范围见 [验收索引](acceptance.zh.md).
采集流程见 [测试](testing.zh.md#dspark-比较).

## REQ-CONTEXT-001: 上下文与显存

支持输入与输出合计 262144 token.
普通和 MTP4 模式, batch 1, 2, 4, 完成输入 258048, 输出 4096 的用例.
这些运行不得发生 OOM 或重计算抢占.
保持 32768-token 步预算和 784-token 块.
记录性能和 GPU 显存峰值; 不增加额外比率门槛.
覆盖长上下文前缀恢复和约束解码.
DSpark 在 batch 1, 2, 4 执行相同边界用例.
保留 6 个普通 / MTP4 用例作为独立门槛.

## 功能需求

| ID | 合同 |
|---|---|
| REQ-FUNC-001 | 在 32768-token 预算内分块 prefill; 保存对齐的 GDN checkpoint. |
| REQ-FUNC-002 | 每步先调度活跃解码, 再调度新 prefill. |
| REQ-FUNC-003 | 使用链式哈希前缀复用, LRU 淘汰和协调的 FA / GDN group. |
| REQ-FUNC-004 | 优先抢占较晚接纳的运行请求; 重计算保留已接受历史, 清除未验证草稿. |
| REQ-FUNC-005 | 提供 MTP4, 显式返回保留 token 和新草稿 ID; 保持 BF16 GDN 状态和 block784. |
| REQ-FUNC-006 | 提供可选 DSpark, 使用本地 BF16 checkpoint, 至多 7 个草稿, target 验证和 block784. |

DSpark 使用从 0 开始编号的 target 层输出 `(5,19,33,47,61)`, 以及 5 层草稿 GQA.
保留原生 MTP4, 每个 worker 选择一种草稿模式, 不按请求切换.

DSpark 累计 confidence 阈值的初始设置为 `0.2`, proposal 数量为 0 至 7.
首项 confidence 乘积低于阈值时, 该步由目标执行单 token 解码.
设置 `--dspark-confidence-threshold 0.0` 可保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.

随机采样使用条件 proposal 概率及 target 接受概率 `min(1,p(x)/q(x))`.
拒绝时从归一化的 `max(p-q,0)` 采样, bonus token 从 target 分布采样.
每个条件分布都应用用户采样参数和 grammar mask.
推测模式保持 BF16 GDN 状态及既有半精度舍入约束.
只有接受的输入行才能提交 target 状态或 DSpark context.

## REQ-ACC-001: 数值精度

实际 GQA 和 GDN 推理路径与独立 CPU FP64 参考比较.
覆盖 prefill, decode, 分块边界和 recurrent state.
使用实际舍入后的输入.
BF16 输出使用 `atol=rtol=0.03`.
Recurrent state 的 NRMSE 至多 1%, 最大绝对误差至多为参考峰值的 2%.
保留路径相关容差和现有测试.
展示连贯文本, MTP 和前缀命中.
完整模型 token 相等不作为门槛.

## REQ-RUNNER-002: 独立运行时

使用项目实现和独立库.
项目构建, 测试和推理不得安装, 导入或链接 vLLM.
不得依赖其 checkout, 旧环境或编译缓存.
旧 adapter 已移除; [ADR-002](decisions/ADR-002-gpuworker-adapter.zh.md) 和 [ADR-005](decisions/ADR-005-v2-model-runner.zh.md) 记录替代关系.
`REQ-RUNNER-001` 已退役, 由本需求替代.
按依赖顺序选择兼容的稳定版本, 固定已验证版本.
保留移植代码的来源和许可证.
未来架构, 多 GPU 和其他 NVIDIA GPU 只写简要扩展说明.

## REQ-OBS-001: 诊断

提供 UTC 时间戳, 单调时长, run / request / step 关联和可配置日志级别.
默认级别避免逐步 I/O.
Debug 日志暴露调度, 缓存, 传输和 worker 时长.
区分主机时间与 CUDA 内核时间.
Profiling 需显式选择.

## 服务需求

| ID | 合同 |
|---|---|
| REQ-SERVE-001 | 提供文本 Chat Completions 和 Responses, 流式响应, 工具, 历史, usage, 模型发现, 查询和删除. |
| REQ-SERVE-002 | 提供 off/low/medium/high/xhigh 思考; 默认 medium, high 映射为 xhigh, none 映射为 off. |
| REQ-SERVE-003 | MTP4 和 DSpark 分别通过两种 API 完成真实 oh-my-pi 任务, 实际读文件并将结果传入后续模型输入. |
| REQ-SERVE-004 | 采样前施加 JSON / Schema / strict-tool mask, 包括每种模式的草稿和 bonus token; 拒绝不支持的 schema. |
| REQ-SERVE-005 | 检查排队, 准备, TTFT, 输出速率, 缓存和草稿计数, 不增加 HTTP 吞吐门槛. |

[服务合同](serving.zh.md) 定义支持的参数, XML 限制, 存储限制和错误.
单独使用 scripted worker 不能证明真实模型验收.
每种草稿模式都测试两种 API, 约束, 取消, 混合 batch 和长上下文前缀复用.
OMP 证据必须显示两次源码读取, 结果回传, 继续生成及使用读取结果的答案.
通过 response ID 与服务日志核对实际模式及已验证草稿.

## REQ-KERNEL-001: 冻结 TileLang 比较

项目内 Triton 到 TileLang 的迁移已完成.
保留已验收的 TileLang 实现, 作为 `REQ-KERNEL-002` 的冻结比较对象.
第三方内部实现不属于迁移范围.
固定的 TileFoundry fork 用于开发.
大幅修改 fork 前先讨论.
临时调优放在仓库外.
旧 Triton 后端不提供验收门槛.

## REQ-KERNEL-002: CUDA 与 PTX

B200 使用项目内 CUDA C++ 内核, 可包含 inline PTX.
允许 CUTLASS.
CUDA 为默认后端; TileLang 需显式选择, 不做静默回退.
保留独立精度, 框架, 上下文和真实服务门槛.

在 15 个工作负载内静态推导每个实现路径的最大合法调用.
相同配置去重.
实现路径不同的阶段分别保留.
不独立最大化无法同时出现的维度, 不以动态 tracing 选择正式用例.
每个用例必须稳定快于冻结 TileLang; 无最低百分比增益.
至少 3 轮, 每轮 20 个交错配对.
CUDA 每轮中位数必须更快.
配对节省时间的单侧 95% 置信下界必须为正.
等价融合链包括所有必要复制和 reduction.

正式用例, 测试框架和汇总证据保留在仓库内.
原始 trace 和临时调优保留在仓库外.
Profiling 与验收计时分开.
缺少计数器时明确报告, 不豁免门槛.
使用 [内核流程](kernels.zh.md).

DSpark 增加独立推导的补充算子用例.
原用例集合及冻结 TileLang 源码哈希保持不变.
新增 TileLang 参考与冻结实现保存在不同文件中.
每个新用例使用相同精度, 3 轮, 20 个配对和置信下界门槛.

## REQ-IR-001: Semantic IR

拥有 Qwen target 和 MTP 路径中项目 CUDA 及关键 FlashInfer 算子的语义和 provider 选择.
普通 PyTorch 算子, embedding, reshape 和 `F.linear` 不在此算子清单内.
每个算子具有 PyTorch 参考, shape / dtype 合同, mutation schema 和静态选择的 provider.
Provider 选择使用 phase, shape, dtype, layout 和 device metadata.
Provider 失败即报错; reference 只作为显式 debug 选项.

使用 `torch.compile(fullgraph=True)` 编译 prefill, target decode, MTP draft 和四步 proposal.
DSpark target feature, context injection, backbone, Markov step 和 greedy proposal 使用相同 fullgraph 合同.
Provider lowering 前保留语义节点.
主机规划, 分配, 调度和 ZMQ 保留在这些单元外.
手动 CUDA Graph 包含编译单元; 禁用编译器管理的 graph.
保留持久写入顺序, capture 恢复和显存保护.
Activation donation 需要对指定临时值的证明; 持久缓存不能被 donation.
Graph rewrite 需要等价测试和注明例外原因的调用点清单.

验收包括全部现有精度测试, 普通 / MTP4 上下文用例, 正式算子用例和 15 个性能行.
增加 DSpark 边界, 服务, 补充算子和配对性能用例.
真实模型测试必须覆盖全部 4 个编译单元, 以及 metadata 变化后的 graph replay.
不增加编译加速百分比门槛.

## REQ-LEARN-001: 交互教程

在 `code-journey/` 维护真正的 Twine 教程, 包含 13 个完整中文章节.
通过实际源码讲解核心推理逻辑, 决策和实现.
阅读选项用于了解兴趣和处理内部前置知识.
解释展示的字段, 参数和变量; 默认读者熟悉基础 Rust 和 Python 语法.

使用 LXGW WenKai, Fira Code Nerd Font, KaTeX, 正文字号为桌面 / 移动端 17/16px.
保留主题选择和阅读 / 实验进度.
中文使用半角标点及指定空格; 应用 Humanizer-zh.
按用户要求在端口 18084 提供预览.
服务器地址和进程信息记入 `LOCAL.md`.

所有页面跳转使用条件无序列表.
只有显式完成选项会完成章节.
复习 / 地图访问保留阅读进度; 旧短 passage 迁移为访问记录.
从当前源码提取附带文档, 属性和 decorator 的片段.
删除公共前导缩进; 保留相对缩进, 行号和完整文件哈希.
展示与复制的片段一致, 提供清晰高亮, 完整源码页和键盘滚动.
构建时拒绝缺失的源码覆盖, anchor 和记录字段解释.

测试 13 条路线, 前置知识, 选项, 完成, 迁移, 重启, 刷新和后退历史.
测试 5 个真实 Rust CPU trace: 784, 785, 1568, 1569, 32768 token.
测试字体, 公式, 字段, console 健康, 桌面 / 移动布局和两种主题.
测试片段文档, 缩进规范, 剪贴板路径, 源码哈希, 行 anchor, URL 和至少 4.5:1 的代码对比度.
截图需对照要求的阅读样式检查.
CPU 合成反馈展示调度; GPU 证据使用独立验收记录.

## REQ-DOC-001: 文档与可移植性

项目自有英文 Markdown 遵循 ASD-STE100 Issue 9 规则和词典.
维护完整中文译文, 技术术语表, 自动检查和独立语义审查.
保留有用的当前信息和关键历史决策.
主机信息记入被忽略的 `LOCAL.md`, 原始证据放在被忽略的本地存储.
Rust, Python, 脚本和测试统一使用环境变量和 CLI 参数.
清理当前跟踪内容; 保留 Git 历史.

## REQ-OUT-SCOPE-001: 后续范围

当前范围不包括多模态输入, CPU KV swap, 多 GPU, 超过 1 的 tensor parallelism, gRPC, LoRA 和生产打包.
仅记录扩展点, 不增加空接口或未经测试的支持声明.
