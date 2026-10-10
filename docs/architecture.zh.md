# 架构

Rust 负责请求生命周期和逻辑内存.
Python 负责 GPU tensor 和计算.
[需求](requirements.zh.md) 定义验收行为; [决策](README.zh.md#决策记录) 记录关键替代方案.

## 请求流程

1. Rust 校验 HTTP 请求并保留接纳容量.
2. Python 在后台线程应用 tokenizer / template, 准备采样和 grammar 状态.
3. Python 主线程安装准备好的状态.
4. Rust 接纳 prompt, 查找一致前缀并调度 token 工作.
5. Rust 分配逻辑 FA / GDN slot, 发送执行 metadata.
6. Python 构建 batch plan, 启动 target 和所选草稿模式的 GPU 单元.
7. Python 返回保留 token, 下一批草稿和可选服务输出.
8. Rust 在状态变化前校验完整回复.
9. Rust 提交已接受历史, 结束或重新调度请求, 提供输出.

取消在已完成的引擎操作之间生效.
Finished-only execution 会刷新待处理的 Python 清理, 包括没有剩余工作时.
请求局部失败只停止该请求; worker / device 失败停止引擎.
Rust 子进程管理和 Linux parent-death signal 防止遗留 Python worker.

## Rust 模块

| 模块 | 职责 |
|---|---|
| `crates/kv-cache/src/block.rs`, `pool.rs`, `free_queue.rs` | 块引用, 缓存 metadata 和 LRU 空闲队列. |
| `crates/kv-cache/src/hash.rs`, `group.rs`, `coordinator.rs` | 前缀哈希, group layout 和协调分配. |
| `crates/scheduler/src/request.rs`, `output.rs`, `lib.rs` | 已接受历史, 草稿调度, 校验和队列转换. |
| `crates/zmq-worker/src/client.rs`, `protocol.rs`, `error.rs` | 子进程生命周期, 关联 RPC, 类型化消息和错误边界. |
| `crates/zmq-worker/src/main.rs` | CLI, 离线驱动, 基准计时和运行配置. |
| `crates/zmq-worker/src/serving/` | Axum HTTP, 类型化事件, 工具解析, 响应存储和在线接纳. |

## Python 模块

| 模块 | 职责 |
|---|---|
| `worker/zmq_bridge.py`, `protocol.py` | 消息处理, 后台准备, 主线程安装和清理. |
| `worker/model_runner.py`, `runtime.py`, `logging_utils.py` | 独立初始化, 源码 / 库身份, 执行和诊断. |
| `worker/batch_plan.py`, `tensors.py` | 校验后的 CPU plan, slot 选择, 批量 metadata 传输和 tensor view. |
| `worker/mtp.py` | Proposal, 验证, 已接受状态保留和 checkpoint 写入. |
| `worker/dspark.py`, `worker/dspark_graph.py` | DSpark context, 条件 proposal, confidence 限制和独立 proposal graph. |
| `worker/decode_graph.py`, `graph_cache.py` | 持久输入, 事务式 capture, 有界 graph 条目和 replay. |
| `worker/serving.py`, `sampling.py`, `sampler.py` | Tokenization, XGrammar, 采样, detokenization 和输出计数. |
| `models/qwen.py` | Checkpoint 加载, 64 层 target, MTP 层, FP8 / 普通 projection 和 forward 单元. |
| `models/dspark.py` | 本地 BF16 草稿 checkpoint, 5 层 GQA, target feature projection 及 Markov / confidence head. |
| `ir/` | 语义算子, 参考, mutation schema, provider 选择, lowering 和覆盖. |
| `kernels/` | Attention, GDN, convolution, normalization, elementwise 算子和后端选择. |
| `kernels/cuda_backend/` | B200 CUDA 实现和加载模块来源. |
| `kernels/tilelang_reference/` | 冻结比较实现. |

全部 Python 模块路径相对于 `python/oh_my_vllm/`.

## 传输合同

Rust 绑定 `ipc://<socket_path>`; 唯一 Python worker 连接它.
消息为带 `type` 字段的 msgpack dictionary.
Prepare 和 execute 回复回显 `rpc_id`; 丢弃迟到的已取消 prepare 回复.
`register` 和 `abort` 无回复.

| 消息 | 字段和作用 |
|---|---|
| `init` | 模型路径, 容量, `block_size`, `tensor_parallel_size`, `max_model_len`, 推测模式 / 数量和 DSpark confidence threshold; 分配 worker 状态. |
| `ready` | `logical_num_blocks`, `mamba_blocks`; 返回实际设备容量. |
| `register` | `request_id`, `prompt_token_ids`; 注册离线输入. |
| `prepare` / `prepared` | `rpc_id`, `request_id`, `request` / 准备好的 `prompt_token_ids`; 创建服务状态. |
| `execute` | `rpc_id`, `step_id`, `scheduled`, `finished_request_ids`, `preempted_request_ids`, `num_batched_tokens`; 计算选定工作. |
| Scheduled row | `request_id`, `token_ids`, `num_computed_tokens`, `prefill_token_ids`, `fa_block_table`, `mamba_block_table`. |
| `execute_result` | `rpc_id`, `outputs`; 返回 output row. |
| Output row | `request_id`, `token_ids`, `num_accepted_draft_tokens`, `new_draft_token_ids`; 可选 `error`, `text`, `finish_reason`, `reasoning_tokens`. |
| `error` | `message`, 可选 `rpc_id` 和 `kind`; 区分 validation 和 internal 失败. |
| `abort` / `shutdown` | Abort 带 `request_id`; 释放请求状态 / 停止 worker. |
| `set_speculative_mode` / `mode_changed` | 带关联 ID 的私有比较 RPC, 仅在空闲比较 worker 中切换 MTP / DSpark. |

`prefill_token_ids` 在接纳或重计算时包含完整已接受历史.
普通解码保持增量; 草稿单独保留.
初始化错误不带关联 ID 和 kind; 执行错误的 kind 默认 internal.
缺失注册产生请求局部执行错误.
精确 wire 名称和默认值以 Rust 和 Python 协议定义为准.

## 调度与分配

调度器在 sequence 和 token 预算下先处理运行请求, 再处理等待请求.
对齐 prefill 在 784-token 边界保存可复用 GDN checkpoint.
32768-token prompt 分为 32144 和 624 token.

成功分配至少一页的 prefill 片段后, 后续因对齐缩为一个 token 的 prefill 可以等到下一轮.
仅在剩余预算无法到达下一个 checkpoint 或 prefill 终点时采用这一条件.
纯 decode 轮次, 草稿验证, 可到达的 checkpoint 和最后尾部保持推进.
配置 token 预算不足一页时也保持推进.

所需逻辑容量无法容纳的请求会被拒绝.

分配先移除过期对齐状态, 只读统计所需容量.
容量检查包含 watermark.
任何 group 分配新块前, 先 touch 全部复用块.
随后分配新块, 注册完整缓存块.
这一顺序防止空闲队列重复计数复用块.

容量不足时, Rust 从队尾抢占较晚接纳的运行请求, 重试较早请求.
没有更晚的 victim 时, 抢占当前请求.
Victim 按接纳顺序返回等待队列.
重计算重置 computed cursor, 保留已接受历史并清除草稿.
事务提交前校验输出 ID, 调度数量, token 范围, 接受草稿和错误.

## 缓存 group 与前缀复用

FA group 保存 16 层 784-token page.
GDN align 模式保存运行 / checkpoint 状态, 受保护前一状态和 speculative state slot.
Null placeholder 保留逻辑位置.
未使用的缓存 checkpoint 可以保留到被淘汰.

默认 group 共享一个逻辑池.
`--mamba-blocks N` 为 GDN 提供独立池, ID 在 group 内有效.
FA 容量为 `floor(num_gpu_blocks/3)`; 块 ID 直接索引 tensor slot.
旧 physical-stride 映射已退役.
GDN slot 0 是只读零初始状态, 不作为写入目的地.
FA page 0 为 null page.

概念上的前缀哈希为 `SHA-256(parent_hash || token_ids || extra_keys)[0..8]`.
字节编码包含 parent 存在标记及 token/extra-key 长度.
整数和 digest 前 8 字节使用 little-endian 编码.
每个哈希依赖前序内容; resident 块仍需形成连续 FA 前缀.
协调器将该前缀与可用 GDN checkpoint 协调.
结果为兼容 FA / GDN 命中长度的最小值.
引用保护活跃块; 空闲队列淘汰最久未使用的缓存条目.

普通 GDN 状态为 FP32; MTP 和 DSpark GDN 状态为 BF16.
状态 layout 为 `[slot,48,128,128]`; convolution 状态为 BF16 `[slot,10240,3]`.
参见 [ADR-003](decisions/ADR-003-mtp-state-slots.zh.md) 和 [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.zh.md).

## Target 计算与 attention

Loader 校验 head dimension 256, rotary dimension 64, theta 10000000 和 RMS epsilon `1e-6`.
FP8 activation 使用逐行 128-value scale.
最多 32 个 FP8 projection row 使用第三方库的 TRT-LLM GEMM.

更大的 FP8 batch 使用自有 SM100 CUTLASS 启动入口.
它保留任意 FP32 block scale, FP32 累积和 BF16 输出.
启动入口在调用者 stream 上选择 PDL.
至少 2048 行的宽 gate/up 矩阵使用八 tile swizzle, 改善 cache 局部性.
至少 32144 行的 34816 by 5120 gate/up 矩阵在 two-SM 路径使用 swizzle 16.
该路径使用 256 元素的 K tile, 三级流水与列主序激活和权重 scales.
Scale 分组仍为 128 元素. K-major 布局参考保持 128 元素的 K tile 与五级流水.
其他大型投影保持 K-major scales 与自动级数选择.

在 624 到 2496 行范围内, 五种 one-SM 投影形状使用沿输出列维度含两个 block 的 CTA cluster.
N/K 组合为 34816/5120, 16384/5120, 14336/5120, 5120/17408 和 5120/6144.
TMA multicast 将同一激活 tile 发送到两个 block.
每个 block 计算不同的输出列. scale, 累加, 舍入, PDL 和调用者流保持原契约.
理论计算量保持相同的有效行数和参数范围.

这五种形状使用 M128/N128/K128 tile 与五级输入流水.
自有 `PairedGroupwiseAccum` 继承固定版本 CUTLASS 主循环.
它先发出两次独立输出 tile 的 TMEM load, 再执行一次 `tcgen05.wait::ld.sync.aligned` 等待.
每个输出保持之前的 K 分组顺序, FP32 scale 乘法与 FP32 累加.
最后的 TMEM 等待先于累加槽释放.

它们的 epilogue 直接将累加值按 round-to-nearest-even 转为 BF16.
之前的线性组合固定 alpha 为 1, beta 为 0.
其他 GEMM 形状保持之前的主循环与 epilogue.
项目 `.cuh` 摘要进入构建身份, 加载库来源记录, 已审查运行契约及算子源码清单.
加载器在编译前后核对头文件字节.

不带 scale 的普通 projection 使用 `F.linear`.

这避开已观察到的 FlashInfer CUTLASS 在 17 至 32 row 的不稳定性.
小批 BF16 vocabulary projection 使用 FlashInfer CuTe-DSL GEMM.

Residual / RMS 和 SiLU / FP8 融合保留已有 BF16 舍入点.
MLP gate/up 路径在超过 32 行时, 用一个 CUDA kernel 合并残差 RMS 与激活量化.
它写出 BF16 残差和, 在寄存器中保留归一化后的 BF16 值, 再转换为 FP8.
投影保持之前的 GEMM provider 与数学上的 scale 值.
至多 32 行使用之前的 CUDA 残差 RMS 与量化链.

带 scale 的 GDN 输出先融合 gated RMS 与 FP8 量化, 再使用之前的 GEMM.
每个 token 有 48 个 head, 每个宽度为 128. Kernel 在转换为 FP8 前保留 gated RMS 的 BF16 舍入.
它接受 packed token stride, 省去 BF16 中间数组.
列主序激活 scales 每 warp 处理一个 head 行.
K 主序 scales 在少于 32144 个 token 行时每 warp 处理两个 head 行, 达到该边界时处理四个 head 行.
PDL 在读取输入前等待. 调用者流负责 launch 与输出生命周期.
不带 scale 的 GDN 输出保持 BF16 归一化与普通投影.

大型 gate/up 路径将物理行容量向上取整到四的倍数, 满足 scale 对齐.
融合 kernel 将实际行直接写入该存储.
每次调用清零至多三行尾部, 并打包 43520 字节的 checkpoint scales.
Graph 重放读取变化后的 checkpoint scales. 输出 view 只保留实际行.
Padding 与 scale 打包属于实现开销, 不增加理论下界.

Convolution, GDN 和 RMS 接受 packed projection stride, 返回 dense output.
GDN prefill 显式归一化 Q / K, 使用 FP32 norm 和 BF16 输出.
整数 metadata 按每个 target / draft group 使用一个传输 buffer.
Device view 保留 backing storage.

CUDA attention preparation 融合 Q / K RMS, partial NeoX RoPE, V 转换和 KV 写入.
接受 packed 14336-value projection, 返回连续 Q.
每个 512-wide Q head 的 gate 半部保持不变.
FP64 phase 计算先于 FP32 三角函数.
负 slot 跳过 KV 写入并保留 Q; cache 不能与输入 alias.

Target query span 至少 1024 时, gather 活跃 KV 并使用 ragged TRT-LLM attention.
Span 128 至 1023 且 KV 至少 4096 时使用 native paged context attention.
其他小 span 使用 paged FA2.
路径阈值使用 batch 的最大 query 和 KV 长度.
混合 batch 共享所选路径.
最大上下文显存检查包括 gather KV 的临时内存.

Target `DecodeGraph` 使用 native decode, 将每个 784-token page 视为 49 个 16-token subpage.
未选择 graph 的短 target decode 使用 paged FA2 路径.
K / V offset view 和 `page * 98 + subpage` 表避免 KV 复制.
Graph capture 内的表展开和长度使用当前执行 metadata.
Draft prefill 从位置 1 开始; span 至少 128 时使用 ragged attention.
更小 draft span 使用 FA2; draft decode 使用项目内核.

## MTP

Draft row `p` 组合 target hidden row `p-1` 和 token row `p`.
Row 0 不存在.
统一 RoPE offset 保留相对位置.
Rust 分配 draft / state 容量; Python 不自行创建逻辑块.
784-token 边界 hidden state 在所需 page 不可用时等待.

MTP4 提出 4 个草稿, 对照 target sampling 验证.
Grouped verification 最多 5 个 query, 每个 query 有自己的 causal length.
Single-token decode 路径独立保留.
Worker 保留接受的 target output 和正确 GDN / convolution snapshot.
被拒绝的已调度草稿回滚 computed progress; 未调度草稿不需要回滚.
Grammar simulation 使用各 speculative prefix, 提交保留输出前先回滚.

## DSpark

可选模式通过项目内代码加载本地 DSpark checkpoint.
执行前校验配置, tensor 名称, shape, BF16 dtype 和稳定的源文件身份.
加载的配置和权重哈希标识实际提供 tensor 的文件字节.
不执行 checkpoint 内的 Python 代码.

草稿使用 5 层 GQA, 以及从 0 开始编号的 target 层输出 `(5,19,33,47,61)`.
每个 feature 是下一次 normalization 前的 BF16 层输出.
原生 MTP 和普通计算不提取这些 feature.
共享的 target embedding 和 vocabulary head 提供 DSpark token 表示及 base logits.
草稿将拼接的 target feature 投影为其 5 层的 context KV.

DSpark context page 使用相同的 Rust 分配 FA page ID 和 784-token layout, 保存在独立物理缓存中.
只有已提交的 target 输入行进入 context.
被拒绝的输入行可以保留在 committed cursor 之后的 target FA 中, 后续使用前会被覆盖.
DSpark context 只接收保留输入行.
7 个 proposal row 使用双向草稿 attention 和临时 KV, 不进入持久 context.

Markov head 用前一个 candidate token 修正每行 base logits.
Confidence head 通过累计条件 confidence 限制 candidate prefix.
`--dspark-confidence-threshold` 初始设置为 `0.2`.
首项 confidence 乘积低于阈值时返回 0 个 candidate, 该步使用一个目标 token.
设置为 `0.0` 可在固定数量实验中保留至多 7 个合法 row.
输出和上下文预算可以缩短这个前缀.

Draft block 7 和 training block 16 独立于 target block784.

Greedy verification 将 candidate 与 target argmax token 比较.
随机输出保留每个 candidate 的完整条件 proposal 分布 `q`.
Target 分布 `p` 和 proposal 分布 `q` 都使用配置的 temperature, penalty, top-k / top-p 和 grammar mask.
以概率 `min(1,p(x)/q(x))` 接受 candidate `x`.
拒绝后从归一化的 `max(p-q,0)` 采样, 全部接受后从 `p` 采样 bonus token.
草稿和 target 采样使用各自独立的随机数生成器.

既有 GDN snapshot / rollback 逻辑保留已提交输入后的状态, 验证行数至多 8.
推测状态为 BF16, target 和草稿保留规定的 BF16 舍入点.
Grammar simulation 先回滚, 再由已提交输出更新持久 grammar 状态.
取消和抢占释放 target 与草稿的请求状态.

私有比较 worker 在计时前加载 MTP 与 DSpark, 共享 target 及物理 target 缓存.
仅空闲时允许 mode RPC 将活跃草稿数量在 4 和 7 之间切换.
调度器在每次尝试之间重置前缀 metadata.
HTTP worker 在初始化时选择一种模式, 拒绝此 RPC.
参见 [ADR-009](decisions/ADR-009-dspark-optional-mode.zh.md).

## Semantic IR 与 graph

`torch.library` 算子提供 reference, fake implementation, mutation schema 和 provider 注册.
选择使用静态 metadata, 仅在选择 predicate 需要时转换 symbolic dimension.
固定 PyTorch lowering 在 provider 替换前保留语义节点.
生产 provider 失败报错; debug reference / eager 模式需显式选择.

普通 / MTP4 fullgraph 单元覆盖 prefill, target decode, MTP draft 和四步 proposal.
DSpark 增加 target feature, context injection, backbone, Markov step 和 greedy proposal 单元.
主机规划, 逻辑分配, ZMQ, 采样和普通 PyTorch 算子保留在算子清单外.
手动 CUDA Graph 包含编译单元; 禁用 Inductor graph.
持久 cache 写入保留顺序; 当前没有 activation donation.
单次使用的 BF16 SiLU-to-FP8 rewrite 有等价测试.

普通 / MTP4 target graph 使用 32 条目. DSpark target feature graph 使用 64 条目.
比较 worker 使用 96 条目, target family 下限为 16, target feature family 下限为 32.
两种模式使用独立 family key 和内存 pool. Capture 保留 4 GiB 显存余量检查.

DSpark proposal graph 使用独立的 16 条目 cache 和 pool.
至多 32 row 的 DSpark context injection 使用独立 pool 和 32 条目 cache.
预热和 capture 备份目标 KV slot, 在 `finally` 恢复; replay 提交当前输入.
更大的 context injection 使用编译单元, 不进行手动 capture.
MTP draft / proposal 的默认共享预算为 64 条目, 下限分别为 32 和 8.
此预算同时适用于独立运行和比较 worker.

长持久化 MTP context 使用独立的八条目 cache 与 pool, 保留 12 GiB capture 显存余量. 选中末行使用起点为 1 的自有 decode attention. Capture 备份将写入的 FA 槽, 在预热和捕获之后恢复. Replay 在任何复制之前验证全部 tensor 形状, 类型和 device. 只建立缓存的图可在不同位置与页地址间复用相同 query 形状. Capture 日志与 `mtp_context` 计数器标识这些活动.

普通和 DSpark prefill 通过 `Batch.output_indices` 与 `Batch.output_attention` 选择目标最终输出. 第 63 层保留全部必需 K/V 行, 对选中行计算 attention 与 MLP. MTP 保留供 context 和边界特征使用的目标 hidden. 第 63 层 feature tap 保留完整层输出. 语义操作 `prepare_context` 和 `prepare_query` 分开描述持久写入与有效查询, CUDA 实现保持完整准备 kernel 的 BF16 舍入与 FP64 相位计算.
缓存预算已满时, 替换接纳需要 4 次观察.
衰减计数必须超过最冷可淘汰条目的 2 倍.
预算未满时允许首次 miss 就 capture.
计数每 512 次观察衰减.
空闲 GPU 显存少于 4 GiB 时, cache 跳过新 capture.
Worker 仍执行编译单元.
Capture 失败后退避 64 次观察.
同一 MTP shape 在 4096 次 cache decision 内再次 capture, 会启动共享的 32768-decision capture cooldown.
Resident graph 继续 replay.

各 graph family 内共享 pool, target / draft / proposal family 独立, replay 不重叠.
Capture 恢复持久状态和 FA 写入; 输出复制先于 pool 复用.

Target prefill 为普通和 DSpark 模式各设独立的八条目 family.
这些 family 捕获 query/context 大小固定的 native 或 ragged attention, 并保留至少 32 GiB capture 显存余量.
KV page, state slot, token 和 attention metadata 保持动态.
预热与 capture 在 `finally` 中恢复所有目标 FA/GDN 值.
Replay 先验证全部 metadata 替换, 在复制前恢复捕获的绑定.

FA 2 prefill 使用编译路径.
Dynamo 限制为 4096, 在 256 时警告.
这些限制不能证明无限 shape 变化下编译器内存有界.
参见 [开放审计工作](audit.zh.md).

## CUDA stream 与 PDL

GDN preparation 操作从同一归一化输入分出两条分支.
Origin stream 计算 FP8 QKV/Z 投影与 convolution.
一个 side stream 计算 BF16 decay/beta 投影与 gate.
Side stream 等待输入, 并在成功或失败时汇合到 origin.
Stream 记录让 tensor 存活到其消费者完成.
两条分支不共用可写的 FP8 workspace.

自有 pointwise CUDA 启动使用 programmatic stream serialization.
Producer trigger 允许依赖的 grid 开始.
Consumer 在读取依赖的 activation 或 state 之前等待.
自有 CUTLASS 启动入口启用同一 dependent-launch 协议.
PDL 不保证并发驻留.
CUDA Graph capture 记录 fork/join 与 dependent launch.

默认 allocator 为已汇合的 graph frontier 使用 `graph_capture_record_stream_reuse:True`.
显式 `PYTORCH_ALLOC_CONF` 优先, 其次是旧变量 `PYTORCH_CUDA_ALLOC_CONF` 的值.
包装脚本将所选值统一为 `PYTORCH_ALLOC_CONF`.
诊断开关见 [Development](development.zh.md).

## 阶段模型

`performance/` 将观测到的语义工作映射到官方计算与 HBM 上限.
Rust 记录请求提交, 首 token 和末 token 的时刻及其对应 step.
二进制包含 Cargo 输入与 Rust 源码的构建标记.
采集器验证该标记, 源码身份, checkpoint, 已加载库及各自独立的重复区间.
未知 GPU 工作会阻止验收.
参见 [时延模型](latency-model.zh.md).

## 扩展边界

新模型需要经过验证的加载, 语义合同和缓存 layout.
多 GPU 需要新进程所有权和 collective 合同.
其他 NVIDIA 后端需要设备专属 provider 和实测验收.
当前接口没有这些支持声明.

DSpark 运行时支持仅覆盖已校验的本地 checkpoint 架构.
不同草稿 checkpoint 需要新增加载, 语义, 数值和验收测试.
源码身份及测量范围见 [验收索引](acceptance.zh.md).
