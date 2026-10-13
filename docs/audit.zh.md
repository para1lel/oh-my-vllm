# 审计状态与剩余工作

源码 `030f60f` 的审计覆盖项目 Rust, Python, 内核, 测试, 脚本, 基准和开发工具.
共识别 51 项问题.
关闭项索引保留有用合同及修复身份.
逐日整改经过保留在 Git 历史.
性能证据只适用于实测源码, 见 [验收](acceptance.zh.md).

## 服务控制 token 与重复

这些发现不属于原始源码审计的条目.
`SRV-11` 和 `SRV-12` 的协议修改处于验证阶段, 严重性为 P1.
`SRV-11` 涉及 grammar/parser 按字符串结束思考, 却没有产生结束 Python 思考计数的 token ID.
`SRV-12` 涉及错误工具起始格式绕过原 grammar trigger 及其 schema 或函数策略.
修复前已复现失败的 CPU 回归.
[ADR-016](decisions/ADR-016-canonical-serving-control-tokens.zh.md) 规定合同及功能保留范围.

`SRV-13` 仍未关闭, 严重性为 P1. 已记录的持久 MTP4 请求重复正文, 未能满足客户端进展要求.
其中的伪标签是合法文本, 并非重复的思考通道切换.
Target 条件概率测量支持将引用文本中的结束分隔符视为重复输出的可能原因.
它们没有复现原始采样路径.

[调查摘要](../bench/evidence/2026-10-13-constrained-decoding-investigation.json) 记录源码身份, 原始哈希, 测量和限制.
协议修复不能关闭该条目, 也不能证明长程 Agent 验收.

## PY-04: Graph 显存与 shape 变化

状态: partial/open, 严重性 P2.
Graph family 使用有界 graph 条目缓存和共享 pool, 配合接纳, 淘汰, 空闲显存保护和 capture 恢复.
压力驱逐可以移除某个 family 的最后一个 graph 及 pool handle. 完成同步, 垃圾回收和 allocator 缓存释放后再次捕获.
Resident-graph36k-to40k 转换在空闲显存少于 4 GiB 时完成.
普通 / MTP4 同形状 262144 的多次执行也完成, 无 graph eviction 或 recapture.
这些 probe 不覆盖跨形状 262k eviction/recapture 或全部 late-capture 显存条件.

后续测试需在最大 context 和实测显存压力下强制跨 shape eviction 和 recapture.
记录前后 key, counter, 持久状态一致性, 显存峰值和进程清理.
实现前审查工作负载, 确保在支持的显存范围内.

## PY-06: 主机同步

状态: partial/open, 严重性 P2.
Metadata 和 readback 已批量处理, 但执行链仍有依赖型主机等待.
Pinned-D-to-H ABBA 实验未发现明显的完整调用收益.
完整 step overlap 候选证明正确性和实测收益之前, 保持当前已验证路径.
Trace 或异步 API 调用本身不能证明 overlap 或吞吐改善.

## KRN-06: Streaming store 候选

状态: open, 优化候选已拒绝, 严重性 P2.
KRN-06 旧候选为原生 residual 输出使用 streaming store.
其测量未显示稳定的完整调用净收益. 项目拒绝了该替换方案.

当前带条件限制的原生 residual 和 DSpark hidden RMS store 路径继续使用.
只有新假设和完整操作测量支持时才对新候选作出决定.

## 其他证据限制

Worker 现在将自有 Dynamo variant 限为 256, 每个 code object 的 variant 限为 64, code object 限为 32.
自有 Inductor module 另有 256 项上限. 已捕获 graph 在 graph-cache 上限内保留所需执行对象.
Provider selection, rewrite record 和 compile counter label 也有固定上限.
回归测试检查淘汰, 以及淘汰后 graph 重放的正确性.

自有 FX source 和 FakeTensor metadata 同时受数量与字节上限约束.
Compiler retirement 在安全主机边界启动垃圾回收.

源码 `706f1a6` 使用四个 agent, 通过 MTP4 和 DSpark 一小时资源和内存检查.
结果覆盖观测到的输入 shape. 无限变化的 shape 仍需进一步测量.
见 [验收](acceptance.zh.md) 和 [持续测试](testing.zh.md#长时-agentic-服务).

## 服务资源修正

Prefix block 只有在校验后的 worker 输出提交其状态后才进入查找.
失败的待提交 block 不能成为共享前缀. RPC 输出 ID 在错误存储修改前完成校验.
准入 lease 限制排队请求, 存活请求, 编码状态预留和已完成的 stream drain.
字节计数不包含 JSON allocator 额外开销. RSS 单独接受内存审计.

Worker 失败后使对应逻辑 KV 失效, 并在 HTTP 保持可用的情况下重建 worker.
受影响的 stream 停止并报告失败. 进程出生记录保护清理过程, 避免 PID 复用导致误杀.
Allocator-OOM 采集器检查故障阶段, 缓存失效, 恢复后的成功服务与进程清理.
源码 `706f1a6` 通过十五项阶段负载, 两项一小时 agentic 采集和八项 allocator-OOM 恢复阶段.
[验收索引](acceptance.zh.md) 给出源码身份, 原始 SHA, 审查和限制.

阶段时延, 算子, 边界和服务证据保留各自的实测源码范围.
[验收索引](acceptance.zh.md)保留源码范围及全部原始失败记录.
历史十五行阶段时延采集在 `706f1a6` 上通过.
新的长 prefill 调度与持久会话负载处于验收阶段.
源码和审查范围见验收索引.

## 已关闭问题索引

以下 48 项有记录的修复.
当前测试和历史验收的适用范围不同.
实现背景查看相应提交, 当前验证流程见 [测试](testing.zh.md).

| ID | 严重性 | 已修复合同 | 修复提交 |
|---|---|---|---|
| SRV-01 | P0 | 请求失败隔离 | `d0d286a`, `96e4ecc` |
| SRV-02 | P0 | Worker 生命周期 | `487f8de` |
| SRV-03 | P0 | HTTP 错误分类 | `d0d286a`, `d33b844` |
| SRV-04 | P0 | 长度输出和工具完成 | `d0d286a` |
| EVD-01 | P1 | 重复和 spread 门槛 | `487f8de` |
| EVD-02 | P1 | 实际配置观察 | `487f8de` |
| KRN-01 | P1 | Grouped query 范围 | `eb5ff62`, `96e4ecc` |
| KRN-02 | P1 | KV slot 范围 | `96e4ecc` |
| KRN-03 | P1 | 空 split merge | `96e4ecc` |
| KRN-04 | P1 | Flat index 范围 | `eb5ff62`, `952a01d`, `312c54b` |
| KRN-05 | P1 | Q/K 合同 | `487f8de` |
| PY-01 | P1 | FA page 所有权 | `487f8de` |
| PY-02 | P1 | FP8 scale packing | `eb5ff62` |
| SCH-01 | P1 | 事务式输出校验 | `487f8de` |
| SCH-02 | P1 | 引用和空闲队列保护 | `487f8de` |
| SCH-03 | P1 | 未对齐 prefill 进度 | `42197f7` |
| SCH-04 | P1 | 容量接纳 | `487f8de` |
| SCH-05 | P1 | 内部 null block | `42197f7` |
| SRV-05 | P1 | RPC 关联和 timeout | `96e4ecc` |
| SRV-06 | P1 | 单向消息和准备隔离 | `d0d286a`, `96e4ecc` |
| SRV-07 | P1 | 服务输出所有权 | `96e4ecc` |
| PY-03 | P2 | Graph 接纳和淘汰 | `5772834`, `2837c49` |
| PY-05 | P2 | 紧凑 MTP metadata | `17413e5` |
| PY-07 | P2 | 缓存 penalty 和 top-k 选择 | `503ea27` |
| SRV-08 | P2 | 后台准备 | `efdef09` |
| SRV-09 | P2 | 增量工具解析 | `c082937` |
| SRV-10 | P2 | 按时间控制流式背压 | `b1761e9` |
| SCH-06 | P2 | 优先级抢占 | `67deb85` |
| SCH-07 | P2 | 增量前缀哈希 | `5a3b703` |
| KRN-07 | P2 | 可观察对齐复制 | `f4aacca` |
| KRN-08 | P2 | Fast/generic dispatch counter | `40e3e57` |
| KRN-09 | P2 | CUDA 错误观察 | `94c3473` |
| KRN-10 | P2 | Launch 配置和 page-table 复用 | `37f8cc9`, `231f066` |
| EVD-13 | P2 | Native cache 审计 | `895d57b`, `487f8de` |
| EVD-03 | P3 | 统一 pytest discovery | `6da6427` |
| EVD-04 | P3 | 实际 GQA 参考 | `4ed62ab` |
| EVD-05 | P3 | 独立 graph 预期 | `4ed62ab` |
| EVD-06 | P3 | 冻结参考身份 | `895d57b` |
| EVD-07 | P3 | 上下文和 HTTP 证据 | `bd8e21e`, `8dfc97b` |
| EVD-08 | P3 | 取消所有权测试 | `6da6427` |
| EVD-09 | P3 | 完整算子输出门槛 | `ea0aa4e` |
| EVD-10 | P3 | 确定性 fixture seed | `6da6427` |
| EVD-11 | P3 | 文本 cache 和根目录审计 | `487f8de` |
| EVD-12 | P3 | 编译器 / 模块来源 | `f34b661`, `9648012` |
| MNT-01 | P3 | 有效 CUDA factory 参数 | `be84176` |
| MNT-02 | P3 | Dtype, epsilon 和空输入合同 | `04540bd` |
| MNT-03 | P3 | 数值 dispatch 合同 | `0394727` |
| MNT-04 | P3 | 协议和 shutdown 维护 | `96e4ecc`, `ff1f944` |


## 修改规则

可行时先添加失败回归, 然后修复并独立审查.
保持数值容差和有效门槛不变.
Device 实现修改后再次执行受影响的正式算子用例.
声称性能验收持续成立前再次执行受影响框架工作负载.
更新本状态和 [交接](handoff.zh.md), 包括剩余限制.

## 自有 CUDA 算子审计

Opus 5.5 检查了自有 CUDA 入口, Python 包装器, 语义操作和生产调用点.
另一名 agent 对照源码中的保护, 分派, 图恢复和测量, 核查并修正了审计条目.
[算子审计](../bench/evidence/2026-10-10-owned-cuda-audit.json) 给出完整清单, 源码摘要, 修正和验证范围.

这份审计使用 CUDA 头文件拆分前的 `a6b9bd2` 和 `a5cc0ef`.
当前头文件布局的检查使用其他记录.

原生 CUDA 模块有 25 个 FFI 导出. `groupwise_fp8` 模块另有一个导出.
`variant_launch_count` 统计主机分派. 其余 25 个导出执行 GPU 计算.
`operators/` 中的 27 个 `__global__` 定义不包含 CUTLASS 模板实例.
审计分组统计语义操作, FFI 导出, kernel 定义和编译变体.

| 条目 | 契约 | 更多验证 |
|---|---|---|
| FP8 GEMM 工作区 | GEMM 代码在初始化和执行前检查所需工作区大小. | 修改配置后保留容量保护. |
| FP8 scale 与累加 | 选定的 624 到 2496 行投影使用 K-major scale. 主循环按 K-group 顺序执行 FP32 累加. | 一起检查 tile, 流水级数与 scale 布局. |
| GDN 复制与 PDL | 选定视图在分块 GDN prefill 前使用按输入 stride 执行的 V 复制. 消费者调用未提供 PDL 启动参数. | 改变触发位置的测试前检查两个启动入口. |
| DSpark 草稿注意力 | 七个草稿行均读取完整的已提交 context 和全部七个草稿 KV 行. 草稿块采用双向注意力. | 保留参考掩码和数学注意力 scale `1/sqrt(128)`. |
| 自有 decode 注意力 | 四个未分组 query 使用两个 shared-memory buffer 传输下一 tile. 其他选定路径使用一个 buffer. | 替换前比较完整注意力操作. |
| 图与 stream 生命周期 | 捕获代码恢复持久状态. 调用者 stream 在 `finally` 中记录对 GDN 分支的依赖. | 在内存压力测试通过前保留 `PY-04`. |
| 分派计数 | 计数器统计主机调用, 而非图重放. | 分组记录验证与重放计数. |

另一名 agent 检查 12 个初始审计条目后, 未发现新的生产缺陷.
修正后的审计给出已有保护, 错误表述, 外部候选, 以及尚无适用测量的性能修改建议.
本次审计后, 生产实现的选择保持不变.

### 算子调优

1. 对 624 到 2496 行的 FP8 投影, 先取得对应源码指令的观测, 再测试流水级数或寄存器配置.
2. 对 packed FP32 候选, 测量完整 RMS/SiLU, 量化, scale 处理和 GEMM 操作.
3. 修改 PDL 接入前, 一起检查 GDN 生产者与消费者的启动契约.
4. 选择候选前, 用本版本源码完成算子, 框架, 容量和服务测试.

外部 packed 候选把两个 FP32 值组合成 `float2`, 供 `__ffma2_rn` 使用.
它在 18 项中保持输出相等. 十项速度测试通过, 八项失败.
七个 GEMM 形状覆盖九个完整操作, 这些操作在测量中用时较短.
只有同一 GEMM 形状的所有已测输出路径均通过, 这个诊断选集才使用该形状.
这些路径包括不带 gate 的输出和带 gate 的输出.

已测行数为 624, 1248 和 2496. 这些测试不覆盖 two-SM tile 或大行数 MN scale.
进入生产前, 用本版本源码检查这些选定的诊断结果.

Profiler 测量标明源码, 形状, kernel 和重放.
Eligible warps 的单位是每个活跃 scheduler 周期的 warp 数.
Scoreboard 采样不能给出某条指令占用的运行时间百分比.
缓存可以满足 TMA 请求, 因而 TMA 流量和 HBM 流量是不同的测量.
没有其他测量时, 低 occupancy 不能确定延迟原因, 也不能说明另一种 tile 更快.
