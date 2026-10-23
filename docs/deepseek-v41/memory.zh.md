# DeepSeek V4.1 内存计划

## 1. 范围与依据

CPU [规划器](../../development/model/deepseek_v41/memory.py) 提供存储计划, 不构成运行时容量验收.
它使用独立的[张量布局](../../development/model/deepseek_v41/layout.py), 不使用 checkpoint 头部或索引总量.
[源代码契约](model.zh.md) 绑定 `LOCAL-V41:inference/model.py` 及其相关源文件.
[状态契约](state.zh.md) 提供已接受状态与恢复规则.
[架构](design.zh.md) 定义四 rank 策略.
Rank IDs 是逻辑组 IDs, 不是物理 GPU 索引.

计划覆盖 64 个已准入请求, 每个请求最多包含 1048576 个输入/输出 token.
DSpark 有五个草稿行, 目标验证最多使用六行.
较小的 kernel 分片不会减少已准入请求数.
本计划不包含 GPU 计算, 分配测量, 吞吐量测量或数值验收.

## 2. 输入与执行

`plan_memory(config, strategy)` 返回可序列化为 JSON 的字典.
`strategy` 为 `TP4`, `TEP4` 或 `DEP4`.
配置包含以下字段:

| 字段 | 含义 |
|---|---|
| `model` | 完整且受支持的 checkpoint 配置 |
| `device_capacity_bytes` | 按 rank 顺序排列的四个正设备容量 |
| `device_unavailable_bytes` | 在预算前扣除的四个不可用量 |
| `host_pinned_capacity_bytes` | 可用的共享 pinned-host 预算 |
| `host_reservations` | `kv_offload`, `uploads` 和 `transfer_leases` 的显式预留量 |
| `reservations` | `graphs`, `staging`, `communication`, `chunks` 和 `media` 的四 rank 字节列表 |
| `limits` | 可选的限制覆盖值 |

默认限制为 `requests=64`, `context_tokens=1048576`, `draft_tokens=5`, `prefill_chunk_rows=128` 和 `index_tile_positions=4096`.
上下文将输入和输出一起计数.
请求数必须为 1-64.
上下文必须为 1-1048576.
草稿容量为 0 或 5, 因为源代码块一次计算五行.
零选择普通 decode, 但仍加载所有草稿权重.

分块行数必须为 1-67108864.
索引 tile 必须为 1-1048576.

未知字段, 不受支持的模型形状, 无效所有者, 无效 scale 块以及非四 rank 拓扑均会报错.
字节运算检查无符号 64 位溢出.
预算差值可以为负.
预算不足会产生假的预算结果, 不会缩小请求限制.

命令从 `PLAN_CONFIG` 读取完整规划器配置:

```bash
scripts/with-env.sh python -m development.model.deepseek_v41.memory \
  --config "$PLAN_CONFIG" --strategy TP4 > "$PLAN_OUTPUT"
```

命令将完整计划写入标准输出.
不会导入模型源代码或 GPU runtime.

## 3. 张量放置

精确名称规则为每个预期张量指定放置方式.
未映射张量会报错, 而不是默认复制.
输出给出存储形状, 存储字节, 分片轴, 半开分片范围, 所有者 rank 和逐 rank 常驻字节.
范围针对存储轴, 也适用于打包 FP4 字节和 scale 张量.

| 张量族 | TP4 | TEP4 | DEP4 |
|---|---|---|---|
| 路由专家 | 每个专家有四个张量分片 | 一个所有者存储完整专家 | 一个所有者存储完整专家 |
| 共享专家 | 张量分片 | 张量分片 | 副本 |
| Query 扩展, 分组输出投影, 索引 query | 张量分片 | 张量分片 | 副本 |
| 词表 embedding/head 和 Markov embedding/head | 词表行分片 | 词表行分片 | 副本 |
| Vision, aligner, 图像标记权重 | Rank 0 | Rank 0 | Rank 0 |
| Engram 表和 scale | 一份共享 pinned 存储 | 一份共享 pinned 存储 | 一份共享 pinned 存储 |
| Engram 投影和 norm 权重 | 设备副本 | 设备副本 | 设备副本 |

专家中间维分片切分 `w1` 和 `w3` 的输出行, 以及 `w2` 的输入列.
完整专家所有权使用连续范围: 每个 rank 有 96 个目标专家或 32 个草稿专家.
共享专家遵循稠密策略, 不遵循路由专家所有权.
Vision 在 rank 0 运行.
向请求所有者传输特征使用单独预算.

Attention 布局遵循源代码类.
`wq_b` 和 `wo_a` 切分输出行.
`wo_b` 切分输入列.
`attn_sink` 切分 head.
索引 `wq_b` 和 `weights_proj` 切分输出 head.
Latent 投影 `wq_a`, `wkv`, compressor 权重和索引 key 生产者使用副本.

Query/KV norm 向量, 路由 gate, hyper-connection 矩阵以及小型标量/向量参数使用副本.
它们不除以四.
FP8 矩阵分片遵守 32 通道 scale 块边界.
FP4 归约分片遵守 32 个逻辑输入通道和每字节两个值的规则.
FP4 输出行分片保留完整的逐行 scale.
小型 BF16 矩阵不会增加 FP8 块填充.

全部三个 DSpark 块遵循与目标块相同的 attention, 稠密和专家规则.
`main_proj`, 其 norm, 最后一个草稿 norm 以及置信度投影使用副本.
草稿词表 embedding/head 权重引用目标权重.
计划不增加重复张量.

## 4. 权重字节与格式

存储字节和计划常驻字节分开计算.
计划保留路由专家的打包 FP4, 并让大多数稠密矩阵使用带 UE8M0 scale 的 FP8.
它遵循源代码的显式参数转换: 分组 `wo_a` 转为 BF16.
其 checkpoint scale 在加载时消耗.
Ratio-2 compressor 投影, 词表 head, Markov head 和置信度投影转为 FP32.
加载临时空间及重叠的源/目标分配不属于常驻权重字节.

以下为根据类推导的字节数, 不是分配测量值:

| 策略 | Rank 0 权重 | Rank 1-3 各自权重 |
|---|---:|---:|
| TP4 | 79217294664 | 78246757704 |
| TEP4 | 79217294664 | 78246757704 |
| DEP4 | 88392951624 | 87422414664 |

共享 Engram 表包含 196613849600 字节.
其 scale 包含 6144182800 字节.
权威 pinned 总量为 202758032400 字节, 不是四份副本.
Engram 投影留在设备上, 并计入 rank 总量.

[Engram 加载器](../../python/oh_my_vllm/models/deepseek_v41/engram.py)将编码张量流式写入一个共享文件, 不展开权重.
它在封存前检查完整 shard 身份, 流式缓冲区最大为 64 MiB.
协调进程必须在加载前设置 CPU NUMA 策略.
源文件页缓存与流式缓冲区及共享存储分开计算.

[注册所有者](../../python/oh_my_vllm/operators/engram/lookup.py)映射已封存文件.
在支持时, 它使用只读 CUDA 注册.
否则, 支持 HMM 的系统使用 `mlock`, 并设置 host 放置及 GPU 访问建议.
按页对齐的映射必须满足进程级 pinned 预算总量.
Workspace 所有者保持映射, 直到 Eager 完成和 graph 释放.
服务分配约束和模型集成仍待完成.

## 5. 常驻状态

TP4 和 TEP4 在所有 rank 上调度相同请求槽.
尽管 query head 分片, 它们的单一 latent KV 向量仍使用副本.
DEP4 将稳定请求槽 `s` 分配给 rank `s%4`, 不进行隐式迁移.
在 64 个请求时, 每个 DEP4 attention 所有者存储 16 个请求.
路由专家所有者即使 attention 分区为空, 仍可接收工作.

缓存计划使用拟采用的打包格式, 而不是参考实现中反量化的 BF16 缓存分配.
打包计数不包含 kernel 特定的行填充.
主全局 KV 每行使用 288 字节: FP4 值及每 16 通道一个 E4M3 scale.
索引 key 每行使用 68 字节: FP4 值及每 32 通道一个 UE8M0 scale.

源 2, 8 和 14 存储 `floor(context/2)` 行.
源 20 存储 `context` 行.
消费者不分配重复全局存储.

在组合限制下, 每个 TP4/TEP4 rank 的全局状态为 59726888960 字节, 每个 DEP4 rank 为 14931722240 字节.

以下容量针对默认限制下的每个所拥有请求:

| 状态 | 计划容量 |
|---|---|
| 已接受目标窗口 | 40 层, 每层 128 行, 每行 528 字节 |
| 已接受草稿窗口 | 3 层, 每层 128 行, 每行 528 字节 |
| 部分 compressor | 3 个源, 2 个槽, 宽度 512 的 FP32 值和 gate 分数 |
| 目标窗口 shadow | 40 层, 每层 6 行 |
| 草稿窗口 shadow | 3 层, 每层 6 行 |
| 全局 shadow | 15 个组合主 KV/索引 key 行 |
| 草稿私有 KV | 3 层, 每层 5 行 |
| 目标特征 | 3 个宽度 5120 的均值残差输入, BF16 |
| 特征 shadow | 6 个目标特征行 |
| 提案分布 | 5 个完整 FP32 词表行 |

窗口行包含 512 个 FP8 值和 16 个 UE8M0 scale.
每个源的部分 compressor 最多有一个有效未完成 token.
两个分配槽遵循源代码缓冲形状.
单独的部分状态快照保护跨压缩边界的拒绝恢复.
Shadow 不替代完整已接受窗口, 也不授权截断 replay.

计划还计入 token 历史, 惩罚计数器, Engram 历史/快照, Markov 特征, token 映射, hash 常量和两份共享 rotary 表.
Grammar 状态, 逻辑标识, prefix 共享和 offload 记录仍须分配契约.

## 6. 分配上限与未决边界

`plan_memory` 按所供预留量与缓冲下限中的较大值计费.
其 `fits_supplied_budgets` 结果仅检查已计存储.
[分配封装](../../development/model/deepseek_v41/allocation.py) 通过 `plan_allocation(plan, **budgets)` 消费一次该计划.
它保持常驻权重/缓存字节不变, 并检查完整默认限制.

各设备上限取设计上限与对应规划器计费量中的较大值.
Host `kv_offload`, `uploads` 和 `transfer_leases` 预留也继续作为下限.
更大的预留会增加显式空闲容量和总量.

输出给出一份设备缓冲表, 逐 rank 上限以及独立的 host arena.
缓冲记录给出形状, 元素大小和数量.
每个 arena 给出对齐, 生命周期, 复用条件以及经过检查的子项总和.
不受支持的限制, 无符号 64 位溢出, 子项溢出或启动预算不足均会报错.

完整 64 请求步骤包含 384 个目标行, 320 个预测器行, 2304 条目标路由边和 960 条草稿路由边.
它不施加 16 请求分片.
4 GiB compute 上限包含 1 GiB kernel 临时空间.
64 MiB staging 上限容纳两个 Engram bank.
通信, 状态, graph, native, media, 填充和启动存储分别计费.
源缓冲细节由可调用接口的输出给出.

提供四个 `device_reserve_bytes` 值.
设备启动必须适合输入计划中的容量减不可用字节, 再减此独立预留后的预算.
为 host 总量提供 `host_capacity_bytes`, `host_unavailable_bytes` 和 `host_reserve_bytes`.
Pinned 启动还必须适合输入计划的可用 pinned 预算.
启动为每个设备增加 128 MiB, 为 host 增加 512 MiB pinned 存储和 512 MiB pageable 存储.
这些缓冲与运行时存储重叠存在.

`pageable_reservations` 有六个上限: `control`, `request_payloads`, `media`, `processes`, `registration` 和 `kernel_io`.
将 `request_state_max_bytes` 和 `response_max_bytes` 绑定到现有全局 CLI 预算.
`request_payloads` 上限计入它们对齐后的总和.
例如, 16 GiB 和 8 GiB 是设计设置, 不是默认值或逐请求限制.
此封装不增加 API payload 或 grammar 限制.

Native, graph 和 decoder 额度是上限, 不是已证明的物理边界.
Grammar parser 存储和 `deep_copy` 字节边界尚未解决.
请求/响应物理副本和 media 准备峰值也没有已证明的边界.
封装列出这些缺口, 并设置 `runtime_feasibility_verified=False`.
它不实现运行时分配保护, 也不完成 Phase 0.

Ready 前, 运行时必须强制执行 arena 上限并拒绝不受支持的分配.
在宣称服务容量前, [容量流程](validation.zh.md#容量检查) 必须验证峰值分配, 精确继续以及同时准入.
