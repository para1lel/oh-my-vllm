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
6. Python 构建 batch plan, 启动 target 或 MTP GPU 单元.
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
| `worker/decode_graph.py`, `graph_cache.py` | 持久输入, 事务式 capture, 有界 graph 条目和 replay. |
| `worker/serving.py`, `sampling.py`, `sampler.py` | Tokenization, XGrammar, 采样, detokenization 和输出计数. |
| `models/qwen.py` | Checkpoint 加载, 64 层 target, MTP 层, FP8 / 普通 projection 和 forward 单元. |
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
| `init` | `model_path`, `num_gpu_blocks`, `mamba_blocks`, `block_size`, `tensor_parallel_size`, `max_model_len`, `num_speculative_tokens`; 分配 worker 状态. |
| `ready` | `logical_num_blocks`, `mamba_blocks`; 返回实际设备容量. |
| `register` | `request_id`, `prompt_token_ids`; 注册离线输入. |
| `prepare` / `prepared` | `rpc_id`, `request_id`, `request` / 准备好的 `prompt_token_ids`; 创建服务状态. |
| `execute` | `rpc_id`, `step_id`, `scheduled`, `finished_request_ids`, `preempted_request_ids`, `num_batched_tokens`; 计算选定工作. |
| Scheduled row | `request_id`, `token_ids`, `num_computed_tokens`, `prefill_token_ids`, `fa_block_table`, `mamba_block_table`. |
| `execute_result` | `rpc_id`, `outputs`; 返回 output row. |
| Output row | `request_id`, `token_ids`, `num_accepted_draft_tokens`, `new_draft_token_ids`; 可选 `error`, `text`, `finish_reason`, `reasoning_tokens`. |
| `error` | `message`, 可选 `rpc_id` 和 `kind`; 区分 validation 和 internal 失败. |
| `abort` / `shutdown` | Abort 带 `request_id`; 释放请求状态 / 停止 worker. |

`prefill_token_ids` 在接纳或重计算时包含完整已接受历史.
普通解码保持增量; 草稿单独保留.
初始化错误不带关联 ID 和 kind; 执行错误的 kind 默认 internal.
缺失注册产生请求局部执行错误.
精确 wire 名称和默认值以 Rust 和 Python 协议定义为准.

## 调度与分配

调度器在 sequence 和 token 预算下先处理运行请求, 再处理等待请求.
对齐 prefill 在 784-token 边界保存可复用 GDN checkpoint.
32768-token prompt 分为 32144 和 624 token.
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

普通 GDN 状态为 FP32; MTP GDN 状态为 BF16.
状态 layout 为 `[slot,48,128,128]`; convolution 状态为 BF16 `[slot,10240,3]`.
参见 [ADR-003](decisions/ADR-003-mtp-state-slots.zh.md) 和 [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.zh.md).

## Target 计算与 attention

Loader 校验 head dimension 256, rotary dimension 64, theta 10000000 和 RMS epsilon `1e-6`.
FP8 activation 使用逐行 128-value scale.
最多 32 个 FP8 projection row 使用独立 TRT-LLM GEMM; 更大的 FP8 batch 使用 CUTLASS.
不带 scale 的普通 projection 使用 `F.linear`.
这避开已观察到的 FlashInfer CUTLASS 在 17 至 32 row 的不稳定性.
小批 BF16 vocabulary projection 使用 FlashInfer CuTe-DSL GEMM.

Residual / RMS 和 SiLU / FP8 融合保留已有 BF16 舍入点.
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

## Semantic IR 与 graph

`torch.library` 算子提供 reference, fake implementation, mutation schema 和 provider 注册.
选择使用静态 metadata, 不提前具体化 symbolic dimension.
固定 PyTorch lowering 在 provider 替换前保留语义节点.
生产 provider 失败报错; debug reference / eager 模式需显式选择.

4 个 fullgraph 单元覆盖 prefill, target decode, MTP draft 和四步 proposal.
主机规划, 逻辑分配, ZMQ, 采样和普通 PyTorch 算子保留在算子清单外.
手动 CUDA Graph 包含编译单元; 禁用 Inductor graph.
持久 cache 写入保留顺序; 当前没有 activation donation.
单次使用的 BF16 SiLU-to-FP8 rewrite 有等价测试.

Target graph 预算为 32 条目.
Draft / proposal 共享 32 条目, 下限分别为 16 和 4.
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
Prefill 使用编译单元, 不做手动 graph capture.
Dynamo 限制为 4096, 在 256 时警告.
这些限制不能证明无限 shape 变化下编译器内存有界.
参见 [开放审计工作](audit.zh.md).

## 扩展边界

新模型需要经过验证的加载, 语义合同和缓存 layout.
多 GPU 需要新进程所有权和 collective 合同.
其他 NVIDIA 后端需要设备专属 provider 和实测验收.
当前接口没有这些支持声明.

DSpark 设计使用 5 层 BF16 GQA 和 target feature `[5,19,33,47,61]`.
包含 confidence / Markov head, 7 个草稿和 8-token verification.
Draft block 7 和 training block 16 独立于 target block784.
这是扩展说明; 未提供 DSpark 运行时.
