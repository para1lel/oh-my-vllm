# 审计状态与剩余工作

源码 `030f60f` 的审计覆盖项目 Rust, Python, 内核, 测试, 脚本, 基准和开发工具.
共识别 51 项问题.
关闭项索引保留有用合同及修复身份.
逐日整改经过保留在 Git 历史.
性能证据只适用于实测源码, 见 [验收](acceptance.zh.md).

## PY-04: Graph 显存与 shape 变化

状态: partial/open, 严重性 P2.
Graph family 使用有界 graph 条目缓存和共享 pool, 配合接纳, 淘汰, 空闲显存保护和 capture 恢复.
Resident-graph36k-to40k 转换在空闲显存少于 4 GiB 时完成.
同形状 262144 的多次执行也完成, 无 graph eviction 或 recapture.
这些 probe 不覆盖跨形状 262k eviction/recapture 或全部 late-capture 显存条件.

后续测试需在实测显存压力下触发真实 eviction 和 recapture.
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

Dynamo 的 4096 限制和 256-variant warning 不能证明无限 shape 变化下编译器内存有界.
Log/cache 审计不能排除静默内存内重新编译.

当前成对比较, 算子, 边界和服务证据保留各自的实测源码范围.
[验收索引](acceptance.zh.md)保留源码范围及全部原始失败记录.
15 个阶段时延负载的采集与模型验证正在进行.

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
