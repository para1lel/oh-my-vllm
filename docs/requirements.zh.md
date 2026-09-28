# 需求 — oh-my-vllm

_确认状态：除另有说明，均经用户确认。最后审阅：2026-09-23。_

当前验收状态见 acceptance.zh.md；对下文部分“已实现”表述有所限定的未修复代码审计问题见 audit-2026-09-23.zh.md。

## REQ-GOAL-001 — 主要目标

**状态：** 用户已确认
**优先级：** 必须

构建以 Rust 为主的推理框架，在单张 B200 GPU 上运行 Qwen3.8-27B-FP8，在下述工作负载达到 vLLM EngineCore 吞吐的至少 95%。Rust 负责服务、调度和逻辑 KV，Python 负责 GPU 计算。

**验收：** `benchmarks/ttft.py` 对冻结基线为 REQ-PERF-001 的每一组记录 passed=true（采用 REQ-PERF-002 协议）。

## REQ-ARCH-001 — Rust/Python 分工

**状态：** 用户已确认
**优先级：** 必须

Rust 负责请求队列、token 预算、KV 块分配、前缀缓存和抢占。Python 负责模型加载、前向传播、注意力 kernel 和 CUDAGraph 捕获。边界为单个 ZMQ DEALER socket，不共享内存，不传递 Tensor 数据。

**约束：** 不把调度逻辑移到 Python，不调用 vLLM Python 调度器。这一分工是设计本身，不是实现便利措施。

## REQ-MODEL-001 — 目标模型

**状态：** 用户已确认
**优先级：** 必须

模型：`/data0/shared/Qwen3.8-27B-FP8`（Qwen3.8-27B，FP8 量化）。架构：48 层 GatedDeltaNet（Mamba）+16 层全注意力。块大小：**784 token**，混合架构的硬约束。KV 分组：FA 组（`group_id=0`，16 层）和 Mamba 组（`group_id=1`，48 层，`mamba_cache_mode="align"`）。

**不可更改：** block_size。代码可以将其他值参数化，但默认值和全部生产路径均为 784。

## REQ-PERF-001 — 吞吐目标

**状态：** 用户已确认
**优先级：** 必须

| 模式 | 输入 token | 输出 token | batch |
|---|---|---|---|
| ordinary / MTP4 / prefix-hit | 32768 | 4096 | 1, 2, 4 |
| ordinary | 131072 | 4096 | 1, 2, 4 |

全部 12 组要求吞吐 >= 新冻结官方 vLLM main 基线的 95%。用户于 2026-09-22 授权更新隔离 checkout/环境并重测全部组。旧基线产物保留为历史。使用相同 token 输入、采样、输出数量及可比较的有效缓存容量，明确记录配置和源码/环境身份。

## REQ-PERF-002 — EngineCore TTFT

全部 12 组同时要求 TTFT <= 新基线的 110%。输入预先 token 化，从共同 batch 提交、请求注册之前启动单调时钟；各请求在调用方收到第一个保留输出 token 时停止。包含引擎排队、调度、传输和采样，不包含 HTTP、tokenization、模型加载及预热。保留每请求 TTFT。每组统计量为各次重复最大请求 TTFT 的中位数。纯 GPU prefill 计时仅供诊断。

至少完整预热两次、测量五次。普通/MTP 测量重置前缀复用；prefix 模式显式预填受控命中。正式运行出现编译或新图捕获则失效。TTFT 或吞吐的 (max-min)/median >10% 时，调查并重跑完整测量集，保留每次尝试和排除原因，不能挑选重复结果。用户于 2026-09-22 将两种指标、两个引擎的稳定性上限从 5% 调整为 10%。复核完整原始测量集时保留原产物并记录新规则，吞吐及 TTFT 比例门槛不变。冷时延单独报告，无硬门槛。修复失败，未经用户授权不放宽阈值。默认 CUDA 后端下全部 12 组均已通过，见 acceptance.zh.md 及保留的原始证据。

## REQ-CONTEXT-001 — 最大上下文与显存

支持输入/输出合计 262144 token。边界工作负载为输入 258048、输出 4096，普通模式和 MTP4，batch 1/2/4，要求无 OOM、无重计算抢占。记录性能和峰值 GPU 显存，不在 12 组矩阵外增加比例门槛。覆盖长上下文前缀恢复和约束解码。保持每步 32768 token 预算和块大小 784；按需优化布局/workspace，保持数值容差与功能。

## REQ-FUNC-001 — 分块 prefill

**状态：** 用户已确认；**已实现**（代码观察）

长 prompt 按 `max_num_batched_tokens=32768` 上限分块。调度器应用 token 预算和显式 aligned_prefill 拆分，在块边界生成 Mamba checkpoint。

## REQ-FUNC-002 — 连续 batching

**状态：** 用户已确认；**已实现**（代码观察）

同一步内先调度 decode 请求，再调度新 prefill。见 `crates/scheduler/src/lib.rs` 的阶段一运行队列和阶段二等待队列。

## REQ-FUNC-003 — 前缀缓存

**状态：** 用户已确认；**已实现**（代码观察）

链式 hash 前缀缓存，SHA-256 截断为 64 位，采用 LRU 淘汰。跨两组 KV 协调，见 `crates/kv-cache/src/`。

## REQ-FUNC-004 — 重计算抢占

**状态：** 用户已确认；**已实现**（代码观察）

运行中请求需要的块超过可用量时，调度器从运行队列尾部抢占优先级更低（更晚接纳）
的请求，并重试高优先级请求。如果没有更低优先级的请求，则抢占当前请求。被抢占者
按接纳顺序返回等待队列，`num_computed_tokens=0`；已接受历史保留，未验证 draft
清空。见 `crates/scheduler/src/lib.rs`（审计 SCH-06）。

**明确推迟：** 基于 swap 的抢占（CPU KV offload）。它需要 Python 端 CPU tensor 管理，不在当前计划中。

## REQ-FUNC-005 — MTP 推测解码

**状态：** 用户已确认，已实现并端到端运行。

以 num_speculative_tokens=4 启用 MTP4。自有 Worker 在同一协议响应中返回全部保留 target token 和明确的 new_draft_token_ids。Rust 调度 draft、预留 target 状态并回滚已调度的拒绝项。ADR003 记录为保留 block784 而使用 BF16 SSM。连贯 MTP 文本、实际路径 FP64 和功能组合已通过；历史 MTP 性能在 batch 1/2/4 均通过。

## REQ-ACC-001 — 实际路径 GQA/GDN 精度

将真实推理的 GQA/GDN kernel 与独立 CPU FP64 参考比较，覆盖 prefill、decode、分块边界和递归状态。记录随 dtype 变化的容差。已有独立 GQA 测试不能证明实际路径覆盖。还需展示连贯真实文本推理，包含 MTP 和前缀命中。完整模型 token 相等不是验收门槛。

## REQ-RUNNER-002 — 独立运行时

用本仓库维护的代码和独立库替代 vLLM 实现依赖。保留 Rust 服务/调度/逻辑 KV 及 Python GPU 执行。最终构建、测试和服务不得安装/导入/链接 vLLM，也不依赖其源码、conda 环境或编译缓存。V2 适配器已移除。运行时独立性和正确性通过；当前性能验收为 REQ-PERF-001/002 的十二组矩阵。

按依赖顺序选择较新且兼容的稳定版本，验证后固定。高层依赖在 conda oh-my-vllm，可用正常工作的系统 CUDA/编译工具。可移植选定代码并保留来源和许可证，但不能整套复制框架。

保留全部功能及 FP64 容差。每个里程碑运行针对性回归。（原始 2026-09-19 基线的九组门槛已于 2026-09-21 达成，并由 REQ-PERF-001/002 取代。）未来模型架构、单机多 GPU、其他 NVIDIA GPU 和本地 DSpark 只需简要扩展文档。用户确认的边界见 ADR-006。

## REQ-RUNNER-001 — V2 Model Runner（已完成的前序阶段）

只用 GPUWorker 的 V2 Model Runner，不保留旧实现或回退。保留全部离线和服务功能，包括 MTP4 约束解码、前缀缓存、重计算抢占和取消。不改 vLLM 源码，所需适配在本项目实现。Rust 拥有已接受历史，可在接纳/恢复时发送完整历史；普通 decode 保持增量，draft 单独处理。

该迁移复用 bench/baseline/2026-09-19-acceptance.json 中冻结的九组 EngineCore 基线，不重跑 vLLM。保留 95% 门槛、已有 FP64 容差和 HTTP 可观测检查。当前 editable vLLM revision、包版本和历史基线身份分别记录，见 ADR-005。

## REQ-OBS-001 — 带时间戳的诊断日志

两进程需要 UTC 时间戳、单调计时、run/request/step 关联及可配置级别。默认日志避免逐步 I/O；debug 展示 scheduler、KV、传输和 worker 计时。主机执行时间不能作为 CUDA kernel 时间报告。profiling 显式开启。

## REQ-SERVE-001 — OpenAI 兼容本地服务

**状态：** 用户于 2026-09-21 确认，已实现，真实 MTP4 验收通过
**优先级：** 服务扩展必须具备

同时支持 Chat Completions 和 Responses，包含流式/非流式响应、模型发现、函数工具调用、工具结果历史、usage、终止及错误语义。服务和 oh-my-pi 在本机运行。HTTP 入口、协议适配和生命周期优先 Rust，保留 Rust scheduler/KV 所有权及 Rust/Python GPU 分工。这扩展了此前 HTTP 排除范围，但不替代 EngineCore 性能要求，也不声明 HTTP 性能验收。

Responses 支持完整历史、存储型响应的 `previous_response_id`、查询和删除。使用有界、可过期内存，无需跨服务重启续接。默认 store=true、TTL 一小时、1,000 条记录、256 MiB 序列化响应/历史载荷。可配置限制、错误及兼容性见 serving.md。

## REQ-SERVE-002 — 可配置思考强度

**状态：** 用户于 2026-09-21 确认，已实现，真实 MTP4 验收通过

提供 off/low/medium/high/xhigh，默认 medium。high 是原生 xhigh 的别名，OpenAI none 是 off 的别名。使用原生模板，将 reasoning 与正文/工具参数分离。OMP 映射、回放和校验后的模板扩展见 serving.md。这些是 prompt 控制，不是硬性思考预算。

## REQ-SERVE-003 — 两种 API 的真实 oh-my-pi 验收

**状态：** 用户于 2026-09-21 确认，两种真实 MTP4 agentic 任务均通过

在本仓库分别对两种 API 运行 oh-my-pi。要求读取 README、架构文档和必要源码，然后引用文件介绍项目目标、架构、运行方式和当前完成状态。必须有真实工具调用、工具结果回传模型以及基于文件的最终回答；该任务中不修改仓库。工具由 oh-my-pi 执行。测试两种 API，以及非流式、取消、错误和思考映射。详细覆盖见 [serving.md](serving.zh.md)。

实现请求取代此前仅文档讨论。两次真实 agentic 运行都启用 MTP。脚本化 worker/客户端协议测试不能代替真实模型验收。早期主机 CUDA/NVML 故障已恢复；真实证据和局限记录在 acceptance.md、handoff.md。

## REQ-SERVE-004 — 兼容 MTP 的约束解码

通过 token 级 mask 支持 JSON object、JSON Schema 和严格函数参数，包含 MTP 验证路径。不静默回退到非 MTP，不用仅生成后校验替代。不支持的 schema 显式拒绝。支持 tool choice auto/none/required/named 和 parallel_tool_calls=false。支持 schema 和 XML 编码限制见 serving.md。

## REQ-SERVE-005 — 服务可观测性

不添加严格 HTTP 吞吐门槛。检查真实排队/准备耗时、TTFT、输出速率、步骤/缓存行为和 MTP proposed/accepted 计数。诊断异常日志及资源滞留。不能用旧 EngineCore 结果作为服务性能证据。既有 REQ-PERF-001 保持不变。

## REQ-OUT-SCOPE-001 — 明确排除

**状态：** 用户已确认

- 多模态输入：当前仅文本，架构保留扩展可能。
- 基于 swap 的抢占（CPU KV offload）。
- 多 GPU / tensor parallel >1。
- gRPC server；OpenAI 兼容 HTTP 属于范围内，见 REQ-SERVE-001。
- LoRA adapter。
- 生产部署或容器化。

## REQ-KERNEL-001 — TileLang 自定义 kernel 与 TileFoundry 工作流

**状态：** 已实现并验收（2026-09-22）；现为 REQ-KERNEL-002 的冻结对照
**优先级：** 必须

逐步将所有项目自有 Triton kernel 替换为 TileLang，保留每个现有功能及数值容差。第三方库内部不在替换范围内。最终生产代码不保留旧自定义 Triton 回退。TileFoundry 是仅供开发的工具，从 3rdparty 下固定的个人 fork/submodule 在现有 conda 环境中源码安装。取消 Transformers 上限，简单兼容问题在 fork 修复，重大修复先讨论。可降级为兼容已发布 TileLang/OR-Tools，但不在本地维护这两个库。

保持已有正确性和冻结 vLLM 的 12 组验收标准。不设相对旧 Triton kernel 的性能或正确性门槛。全部 12 组必须通过吞吐、TTFT 和稳定性后才算完成。结合 TileFoundry 分析和实际模型测量，在所需多种 shape 上调查并调优；解释不能豁免失败的性能门槛。临时算子测试和记录必须放在仓库外。最终保留测试/证据仅限用户要求和已有文档规定的验收，见 tilelang-development.md。

## REQ-KERNEL-002 — CUDA/PTX 迁移（2026-09-22）

**状态：** 已实现并验收；CUDA 为默认后端（2026-09-23）。

设计讨论后用户授权实施。将全部项目自有 TileLang kernel 替换为 CUDA C++ 和必要的内联 PTX，仅针对 B200 优化，允许使用 CUTLASS。冻结并保留已验收的 TileLang 后端，可显式选择；CUDA 验收禁止静默回退。第三方库算子不在范围内。保持已有独立正确性参考与容差、12 组冻结 vLLM 性能门槛（吞吐 95%、TTFT 110%、极差/中位数 10%）、最大上下文以及 MTP 服务/agentic 检查。

针对不同实现路径/配置，静态推导每个算子在 12 组工作负载中的最大合法调用，合并相同配置，不做动态 shape 采集。仅在实现路径不同时区分 prefill/decode 或普通/MTP；不将实际不能同时出现的各维度最大值拼成测试。所有去重配置均须稳定超过冻结 TileLang，不设最低提速百分比。允许比较等价融合链，包含所有必要复制和归约。至少三轮独立交错成对计时，每轮 CUDA 中位耗时更低，且成对节省耗时的单侧 95% 置信下界为正；无法确定的差异不算通过。不新增相对 TileLang 的端到端性能比例门槛。

项目开发工具集成 TileFoundry 语义/静态分析、CUDA Event/Graph 计时和 Nsight/编译器指标，明确区分估计与实测。硬件分析独立于验收计时。先检查计数器权限，若缺失则保留可用证据并报告，不豁免性能门槛。TileFoundry 仅为开发依赖。正式配置、测试程序和汇总证据入库，临时调优及原始 trace 留在仓库外。见 cuda-development.zh.md。
