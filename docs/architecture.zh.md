# 架构 — oh-my-vllm

Rust 负责 HTTP 服务、请求调度、已接受 token 历史及逻辑 KV 分配。Python 负责单设备 GPU 计算。两进程通过 ZMQ DEALER 和 msgpack 通信；不存在 vLLM 运行时、调度器或模型适配器。

## 执行流程

1. Rust 前端归一化 Chat/Responses 请求。Python ServingAdapter 应用 checkpoint 的 chat template、tokenizer、采样配置和 XGrammar 约束，返回 prompt ID。
2. Rust 接纳请求、解析共享前缀并分配 FA/Mamba 表。调度器将 prefill、decode 和实际 MTP draft ID 组成 batch。
3. Python 验证已接受历史，并按各 tensor 容量验证物理地址（包括递归状态别名和跨请求 FA 页写入），执行 Qwen 模型，并在逐 draft grammar mask 下从 target 分布采样。
4. Python 只提交保留的输出，选择被接受的递归快照，保存跨块 checkpoint，再生成新的 MTP proposal。
5. Rust 更新已接受历史、回滚被拒绝的推测位置并释放已完成请求。即使没有请求被调度，仅包含完成信息的通知也会释放 Python 状态。

消息字段和调度器/KV 数据结构见 [design.md](design.zh.md)。

错误处理：prepare 回复携带 validation/internal 类别；HTTP 适配层只将明确的输入验证映射为 400，未知 worker 故障映射为 500。RPC 回复携带 ID；超时的准备会取消，迟到的回复会被丢弃。execute 结果可报告请求局部错误，只移除对应请求；CUDA/设备执行故障仍会停止引擎。Rust 父进程死亡时，自有 Python worker 会退出。

## Rust 模块

- `crates/kv-cache`：共享逻辑块池、链式 hash 前缀、对齐 Mamba checkpoint 和推测预留。两阶段分配在分配新页前先触达复用前缀。
- `crates/scheduler`：FCFS 等待/运行队列、分块 prefill、token 预算、错峰接纳、重计算抢占和已接受 draft 计数。kv-cache crate 默认使用一个共享池，指定 `--mamba-blocks` 时 FA/GDN 使用独立池（ADR-007）。
- `crates/zmq-worker`：CLI、OpenAI 兼容 HTTP API、模型进程生命周期、取消、ZMQ 传输和关联日志。

目标是在一张 B200 上运行 `/data0/shared/Qwen3.8-27B-FP8`。每个逻辑 FA 页包含 784 token，共 16 层 FA、48 层 GDN。保留 CLI 的 `num_gpu_blocks` 容量单位以兼容冻结基线：ready 报告 floor(value/3) 个逻辑块。Python 现在直接按逻辑容量为每层分配 tensor，不再有旧三路物理 stride 或混合布局存储。

## Python 模块

- `worker/protocol.py`：框架自有 wire dataclass。
- `worker/model_runner.py`：具体 Worker、持久请求/采样状态、缓存 tensor 和执行编排。
- `worker/batch_plan.py`：主机校验、已接受状态选择及 checkpoint 复制，不分配逻辑页。
- `models/qwen.py`：直接加载 safetensors、融合投影权重映射、64 层 target 模型和共享 embedding/output 权重的单层 MTP head。
- `worker/sampler.py`：惩罚、top-k/top-p、贪心/随机 target 采样和确定性贪心 proposal 的精确验证。
- `worker/serving.py`：Transformers/Tokenizers 准备、增量解码、XGrammar mask、EOS/stop/长度处理及 reasoning 计数。
- `worker/mtp.py`：偏移 MTP 缓存、边界特征和 proposal 生成。
- `worker/decode_graph.py`：target/MTP CUDA Graph buffer、捕获和回放。
- `worker/runtime.py`：依赖身份及已加载模块/库审计。

## 混合缓存与 MTP

GDN 状态形状为 [slot,48,128,128]，普通模式为 FP32，MTP 为 BF16。因果卷积状态为 BF16 [slot,10240,3]。decode 读取最后提交的物理槽，并独立写候选快照。被拒绝候选绝不会成为缓存前缀。token 跨越 784 边界时，将该位置的精确快照保存到对应 checkpoint 槽，即使后面的 draft 也被接受。零源槽不可变。batch 规划禁止写/写及其他请求写/源别名，包括 checkpoint 复制。

MTP 行 p 组合 target hidden[p-1] 和 input token[p]。统一 +1 RoPE 偏移保留相对注意力，位置零不存在。由此每行与 Rust 前缀 hash 一致，避免共享边界行依赖未缓存后缀。每个 784 边界的 target hidden 与 FA 页一起保存。若 Rust 尚未分配下一页，proposer 推迟边界行并缩短 draft，绝不绕过 Rust 分配页。抢占丢弃请求局部进度；恢复时由已接受前缀和保留的边界特征重建。

贪心 proposal 依次对照各 draft 前缀下的 target 分布采样结果验证。匹配则接受；首次不匹配或 bonus sample 结束验证。这保留随机 target 采样，不假定 draft 概率。grammar 状态只随保留输出前进。

## 设备 kernel 与计算图

独立 FlashInfer TRT-LLM FP8 GEMM 处理 <=32 行，使用列主序 activation scale 和行主序 checkpoint scale。更大输入使用行主序 scale 的 CUTLASS。FlashInfer 0.6.18 的 CUTLASS SM100 17..32 行路径在真实模型宽度下不确定，因此永不选择。实际宽度 FP64/重复测试覆盖 16/17/20/24/32/33 行。量化保留逐行、每 128 元素的 scale。

项目自有 CUDA kernel（保留冻结 TileLang 对照）实现 GDN 递归、因果卷积、分页 split-KV GQA decode、归一化、部分 NeoX RoPE 和逐元素融合。长 GDN/FA prefill 由 FlashInfer 实现。GDN prefill 显式归一化 q/k，因为所选版本声明的归一化标志实际未使用。

decode 图使用固定 token/请求 shape，以及动态位置、块表和状态地址。预热/捕获保存每个 FA/状态写目的地，正式回放前恢复。target 和 proposer 各保留最多 32 种图 shape；prefill 或超过上限的未缓存 shape 使用 eager。图输出在复用前消耗，递归 MTP 输入复制到独立 buffer。`OH_MY_VLLM_ENFORCE_EAGER=1` 仅用于诊断。最终性能需要测得相关 shape 的覆盖，不能仅凭图捕获成功。

## 依赖与未来范围

全部高层依赖和 Rust 工具放在 conda `oh-my-vllm`；允许宿主机驱动及 CUDA/编译工具。独立 FlashInfer/TileLang/第三方 Triton 缓存根目录避免复用旧框架产物。启动记录精确运行时身份，关闭时检查导入模块和映射库中的旧依赖。

[ADR-006](decisions/ADR-006-independent-runtime.zh.md) 说明未来模型/NVIDIA 后端扩展、单机多 GPU collective 和本地 DSpark draft checkpoint。真正开展时再实现具体能力；当前运行时没有空接口、PD 分离路径或多机组件。历史 vLLM/V2 决策保留在早期 ADR 和验收数据中。

### 分组推测验证注意力

图中的 target 验证让同一请求最多五个连续 query 共享一次分块 KV 读取。每个 query 保留各自因果长度；请求之间不共享块表或注意力归一化。CUDA Graph 回放复制 ragged start、块表及长度，因此同一图支持不同逐请求数量。普通单 query decode 保留单 query 路径；draft-head 验证使用相同分组路径，以 token 数和请求数作为 key。分组 FP64 测试覆盖两种 split 数、页边界、偏移后的首个有效位置和回放时重新分组。

无 mask、无 penalty 的贪心 batch 只传输选中 token 和行有效性，所有请求一次传完，然后各自验证 draft 前缀。其他采样配置保留完整 target 分布构造。小型私有 CPU 元数据 tensor 在 GPU consumer 的同一 stream 上执行 nonblocking copy，去除显式逐次等待；不假定 pageable 传输必然与计算重叠。

所有活跃 MTP 请求都有额外三次缓存写空间时，一个 proposal 图计算初始贪心 token 和三个自回归 draft/head 步骤，中间不与主机同步，四个 proposal 一次传回。接近分配/上下文边界时，逐步 proposer 仍会裁剪活跃集合和数量。预热/捕获恢复三个推测 FA 目的地；持久行索引、位置、块表、hidden 输入、模型和缓存引用保持存活供回放使用。两类 proposer 图共享原有 32-shape 预算。

### 独立 FA/GDN 容量（2026-09-22）

`--mamba-blocks N` 为 Rust coordinator 提供独立 GDN 池。FA 容量仍由 worker 报告；Python 按 N 槽分配递归 tensor，按 FA 容量分配 FA/MTP 注意力 tensor。ID 在各缓存组内有效，数值可相同。接纳检查两个池，前缀查找协调两者命中，worker 按各自 tensor 容量验证地址。不指定时保留共享容量配置。理由和长上下文验收协议见 ADR-007。

### 长 prefill 注意力

batch 中 query span 至少为 1024 token 时，Python 只收集活跃 FA KV token 到临时连续 tensor，调用独立 FlashInfer ragged TRT-LLM 注意力，softmax 使用 FP32。Rust 分配和持久 `[page,2,784,heads,dim]` tensor 不变。query span 为 128..1023 且 KV 至少 4096 token 时，使用原生分页 context attention、与 target decode 相同的零拷贝子页视图，以及 FP32 softmax。其他小 query span 使用分页 FA2。混合 batch 共享该决策，因此长 prefill 也会收集活跃 decode 序列。最大上下文 batch4 正确性检查包含由此产生的临时显存；Mamba 精度和数值容差均不改变。

长 prefill 的逐元素计算使用八 token 卷积 tile 和更大 SiLU block；短 decode 保留原 tile 大小。大 gate/up 投影使用独立 CUTLASS dual-SM GEMM。BF16 舍入和逐 token FP8 scale 不变，并与 FP64 参考比较。

卷积、GDN 递归和 RMS 归一化接受带明确 token/head stride 的 packed 投影视图。输出仍为 dense，递归源快照保持隔离。每个 target/draft 组的主机整数元数据通过一个 buffer 复制，设备视图保留底层存储。图输入仍复制到持久 buffer；GDN prefill start 在执行各层前只转换一次 int32。

小 batch BF16 词表投影使用独立 FlashInfer CuTe-DSL GEMM。残差相加和 RMS 归一化共用一个 kernel，在 FP32 归一化前保留 BF16 求和。MLP SiLU/乘法和 FP8 量化共用一个 kernel，保留两处 BF16 舍入和原 scale。融合 RMS 与注意力准备 kernel 使用 epsilon `1e-6`；Qwen loader 在加载权重前验证 checkpoint 的该配置。支持其他 epsilon 需要相应修改 kernel 契约。冻结对照中，Q/K RMS 归一化与部分 NeoX 旋转融合在 TileLang kernel 中，保留中间 BF16 舍入及 packed 投影 stride。固定 256 维 head、64 维旋转和 theta10000000，与校验后的 Qwen checkpoint 一致。相位先用 FP64 计算并约化，再执行 FP32 sin/cos，避免最大上下文附近频率/角度舍入误差放大。冻结 TileLang 的 GDN 递归在至少四条序列时使用 32-value tile，否则为16。原生 CUDA 为模型布局使用带对齐/别名检查的向量状态更新，其他布局保留通用 CUDA 路径。

CUDA 后端将完整的全注意力准备链融合：Q/K RMS/RoPE、V 布局转换和物理 KV 写入。主模型与 MTP 对 packed14336 投影调用同一 `attention_prepare.prepare_attention` 入口。返回的 Q 连续，每个 512 维 Q head 的 gate 半区不变。缓存不得与输入重叠；负 slot 跳过 KV 写入，但仍产生 Q。显式 TileLang 对照后端保留原有完整冻结链。

target decode 图把现有 784-token 页作为 49 个 16-token 子页提供给原生 TRT-LLM 注意力。K/V 偏移视图和 `page * 98 + subpage` 块表避免 KV 复制，并保留 Rust 页所有权。块表展开和 query/KV 长度选择在每次模型执行的图捕获内只做一次，回放使用当前请求元数据。draft 注意力保留项目 kernel，以排除 decode 中不存在的位置零。draft prefill 的 query 至少 128 token 时，从位置一开始收集有效 KV，使用独立 ragged TRT-LLM 注意力及 FP32 softmax；中间 query span 使用 FA2。规划阶段先验证 query 非空连续及 KV 长度，再跳过原生算子的冗余活跃行检查。GDN prefill 用单个 strided kernel 归一化 Q/K，FP32 norm、epsilon 1e-6、BF16 输出，避免多个大临时 tensor。缓存精度和容差均不改变。

## CUDA 后端与冻结对照

自有 kernel 工厂支持显式的进程级后端选择。冻结 TileLang 实现位于 kernels/tilelang_reference，并记录源码清单。原生 CUDA 使用独立 TVM FFI 和调用方 CUDA stream，缺失入口明确报错。算子、框架和功能验收通过后，默认使用 CUDA；启动前设置 OH_MY_VLLM_KERNEL_BACKEND=tilelang 可选择冻结对照。运行时身份使用同一进程级选择。TileFoundry 仍仅用于开发。
