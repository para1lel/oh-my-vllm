# GPU 缓存

Rust 通过 [Qwen 策略](../../../crates/scheduler/src/policy/qwen.rs)管理 token 身份, 预留, pin 和淘汰.
Python 管理[物理缓存](../../../python/oh_my_vllm/models/qwen38/cache.py).
KV 历史和 recurrent checkpoint 保存在 GPU 上.
没有 KV 主机存储或主机淘汰路径.
必要的 token, metadata 和 draft 读回传输遵循[执行规则](../../execution.zh.md).

新布局尚未完成 GPU, 全模型, HTTP 或 AgentX 验收.
[已记录结果](results.zh.md)只适用于记录中的源码和旧缓存布局.

## 存储

布局身份为 `qwen_radix_gdn8192_fp8kv128_v1`.
[布局计算](../../../python/oh_my_vllm/models/qwen38/cache_budget.py)将可用持久容量与其他存储分开.

| 数量 | 含义 |
|---|---|
| `C` | 可用的持久 GDN checkpoint 数 |
| `P = 64 * C` | 可用的持久 KV page 数 |
| 128 tokens | KV 页长 |
| 8192 tokens | 周期性 GDN checkpoint 间隔 |
| 每个池一个 null slot | 不计入可用容量 |
| `6 * N` | `N` 个请求槽位使用的私有 checkpoint slot 数 |
| `ceil(T / 8192) + N` | token 预算为 `T` 时的 scratch checkpoint slot 数 |

Target FA 层共享逻辑 page ID. MTP 另有一层使用相同格式的 FA.
各层具有固定地址的稠密 slab, 形状如下:

| 张量 | 类型 | 形状 |
|---|---|---|
| Codes | 原始 E4M3FN 字节, `uint8` | `[P+1, 2, 128, 4, 256]` |
| Scales | FP32 | `[P+1, 2, 128, 4]` |

每个 scale 对应一个 token, 一个 head 的一条 256 元素 K 或 V 向量.
Null page 零不是有效历史.
页表宽度为 `ceil(max_context / 128)`, 在 262144 tokens 时为 2048.
不保存完整 KV 历史的 BF16 副本.
Recurrent checkpoint 使用 FP32. 卷积和保存的 hidden 值使用 BF16.

[FA 算子](../../../python/oh_my_vllm/operators/fa_fp8)将量化融合到 K/V producer, 将解码融合到 attention 读取.
新行与复用行都使用相同的已存 codes 和 scales.
Q 和 P 使用 BF16. QK 和 PV 使用 BF16 操作数, softmax 和累加使用 FP32.
[格式参考](../../../development/operators/qwen_fp8_reference.py)规定舍入和 scale 计算.
Page copy 和 rollback 同时移动 codes 与 scales, 不重新量化.

[GDN 算子](../../../python/oh_my_vllm/operators/gdn_radix)使用独立于 KV page 的显式导出 slot.
其 64 行计算在显式导出位置及序列末尾分段.
[MTP 算子](../../../python/oh_my_vllm/operators/mtp_fp8)的 context 和 draft 使用同一 FP8 缓存.
旧 FA, GDN 和 MTP 算子仍保留在算子层.

## 独立的前缀状态

一棵 token radix tree 在 cache domain 内比较精确 token 值.
Checkpoint 附件与 page 附件具有独立的容量, 版本, 淘汰和 pin.
淘汰只解除选中附件, 不删除另一类附件或后代节点.

[前缀查询](../../../crates/scheduler/src/policy/qwen/lookup.rs)返回三个不同位置:

| 位置 | 含义 |
|---|---|
| `D` | 最深的 checkpoint 命中 |
| `K` | 连续 KV 前缀的末尾 |
| `R` | 不晚于 `K` 的可用恢复 checkpoint |

`min(D, K)` 不会产生一个恢复 checkpoint.
缺页之后仍有效的 page span 可以继续复用.
Replay 从 `R` 开始遵循完整层序.
复用区间跳过 K/V 投影, 量化和写入, 但继续计算 hidden, GDN, query, attention 和 FFN.
完整 prompt 命中使用保存的 hidden 值, 配合当前请求的 output head 和 sampler.
不会再次计算最后一个 prompt token.

分配在预留目标位置前 pin 全部选中源.
Checkpoint 池, page 池, 私有状态, 导出和写时复制 page 构成一次预留.
预留失败会释放临时资源.
写时复制保护共享的部分 page, 直到完成结果通过验证.

## 成功会话尾

[会话注册表](../../../crates/scheduler/src/session.rs)在异步请求准备前记录准入顺序.
一个 `SessionId` 至多有一条成功尾.
尾值包含其 `CacheDomain`, 所以新 domain 中的成功尾会替换同一会话的旧尾.
新准备失败会保留旧成功尾, 不回退准入顺序.
较早请求即使晚完成, 也不能替换较新准入请求的尾.

Responses 分支继承根会话和缓存 namespace, 各自保持独立的活跃 pin.
匿名请求使用有容量边界的缓存, 不建立永久会话目录.
尾元数据和未 pin 的缓存状态都有容量限制.

[服务身份契约](../../serving.zh.md#缓存身份)规定 HTTP 字段.

## 精确收尾

Parser 在请求进入 `Finalizing` 前固定已接受端点 `A`.
[收尾策略](../../../crates/scheduler/src/policy/qwen/completion.rs)将已计算端点 `F` 与该端点比较.

| 条件 | 收尾工作 |
|---|---|
| `F < A` | 只计算尚未执行的已接受后缀 |
| `F = A` | 导出或复用端点 |
| `F > A` | 恢复已接受的验证端点和 KV 有效末尾, 或从可用的较早 checkpoint 重放 |

收尾不采样, 不增加输出 token, usage, RNG 推进或 grammar 改动.
Proposal 状态不能替代已接受状态.
普通解码也在精确端点保存 output head 的 hidden 值.
MTP context 保持原有规则, token `p` 配合 target hidden `p-1`.

[输出 workspace](../../../python/oh_my_vllm/operators/output/workspace.py) 将 output head hidden 值与 MTP context hidden 值分开.
Output head 使用原有 640 线程采样归一化.
旧的小行数 hidden 导出使用不同的归约, 未经数值证明不能替代 output head 的 hidden 值.
MTP 存储增加一条独立的宽度 5120 BF16 向量, 即每个 checkpoint bundle 增加 10240 字节.

HTTP 成功, SSE 最终标记和已存响应发布都等待精确收尾及完整回复验证.
Parser 和存储容量准备先于成功尾替换.
已提交工作的取消等待计算, 辅助 stream, rollback, 导出, copy 和全部参与 rank 的确认后, 才释放 pin.
[共享生命周期](../../cache-policy.zh.md)规定退休和 worker 替换.

## Cache 内存

缓存总字节上限保持在旧布局的预算水平.
Null slot, 私有 slot, scratch checkpoint, hidden 值和 FP32 scale 都计入此上限.
[Workspace 计划](../../../python/oh_my_vllm/models/qwen38/memory.py)还计入算子 workspace, metadata, 采样和 MTP 备份存储.
仅有 checkpoint slot 数不能限定 GDN 临时存储.
CUDA Graph, 分配器和 driver 峰值尚未实测.
新的可用容量和物理峰值均未验收.

`--mamba-blocks` 选择可用 checkpoint 数. `--num-gpu-blocks` 选择可用 page 数.
选定值必须满足 `P = 64 * C` 和字节预算.
选项冲突会报错.

Ready 报告可用, null, 私有和 scratch 数量, pool bytes, 布局身份及格式指纹.
Rust 在 Ready 验证后构造逻辑池.
[Wire 契约](../../adapters.zh.md#执行协议)规定版本和替换检查.
