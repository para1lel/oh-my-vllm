# 项目技术术语表

技术术语只用于定义的项目含义.
语法和通用词仍遵循 ASD-STE100 规则.
技术条目不授权同词的其他含义或词性.
机器源文件为 [terms.json](terms.json).
产品专名, 源码标识符和精确协议文本保持原样.

| 术语 | 词性 | 项目含义 |
|---|---|---|
| `inference` | 名词 | 产生输出 token 的模型计算. |
| `token` | 名词 | 输入或输出中的一个 vocabulary ID. |
| `checkpoint` | 名词 | 模型权重 / 配置, 或 token 边界保存的 recurrent state. |
| `prefill` | 名词 | 处理 prompt 或重计算历史的模型计算. |
| `decode` | 动词 | 从已接受前缀计算下一 target 输出. |
| `scheduler` | 名词 | 选择 token 工作和逻辑 slot 的 Rust 组件. |
| `scheduling` | 名词 | 在 token 和 sequence 预算下选择请求工作. |
| `cache` | 名词 | 为继续计算或前缀复用保存的 KV 或状态. |
| `cache` | 动词 | 保存已计算数据以供后续复用. |
| `KV` | 名词 | Attention key 和 value. |
| `FA` | 名词 | 针对适用 token 前缀的 full attention. |
| `GQA` | 名词 | KV head 少于 query head 的 grouped-query attention. |
| `GDN` | 名词 | GatedDeltaNet recurrent 计算及状态. |
| `MTP` | 名词 | 带草稿 proposal 和 target verification 的 multi-token prediction. |
| `MTP4` | 名词 | 提出 4 个草稿 token 的 MTP 模式. |
| `draft` | 名词 | 未验证 token proposal 或提供它的模型. |
| `bonus` | 名词 | 已接受 speculative prefix 后的 target token. |
| `slot` | 名词 | Group 内 tensor 的逻辑块或状态位置. |
| `page` | 名词 | 保存 784 个 target token 的持久 attention 存储. |
| `subpage` | 名词 | Target page 内 16 个 token 的 native attention view. |
| `batch` | 名词 | 一起计算的请求或 token row. |
| `batching` | 名词 | 在一个 step 中组合活跃请求工作. |
| `logits` | 名词 | 采样前模型对输出 vocabulary 的分数. |
| `sampling` | 名词 | 按模型分数和配置约束选择 token. |
| `sample` | 动词 | 按配置的分布选择 token. |
| `grammar` | 名词 | JSON 答案或函数参数的 token 级约束. |
| `mask` | 名词 | 采样前使用的允许 token 信息. |
| `tensor` | 名词 | 具有 device 和 layout metadata 的类型化多维数组. |
| `dtype` | 名词 | Tensor 元素表示, 例如 BF16 或 FP32. |
| `stride` | 名词 | Tensor 某维相邻元素之间的存储偏移. |
| `metadata` | 名词 | 与 tensor value 分离的配置和 shape/state 描述. |
| `provider` | 名词 | 语义算子的已注册实现. |
| `IR` | 名词 | 语义 GPU 算子的 intermediate representation. |
| `DSL` | 名词 | 用于项目语义算子的 domain-specific language. |
| `schema` | 名词 | 数据字段或算子 mutation 的显式合同. |
| `mutation` | 名词 | 改变持久 tensor 存储的算子 effect. |
| `donation` | 名词 | 将已证明临时值的存储复用于其他输出. |
| `graph` | 名词 | 带持久输入以供 replay 的 CUDA launch 序列, 或编译器算子表示. |
| `capture` | 动词 | 将 GPU launch 记录到 CUDA Graph. |
| `replay` | 动词 | 使用当前输入启动先前 capture 的 CUDA Graph. |
| `lower` | 动词 | 将语义 IR 节点替换为选定 provider 算子. |
| `compile` | 动词 | 将模型或内核代码转换为可执行 GPU 程序. |
| `compilation` | 名词 | 将模型或内核代码转换为可执行程序. |
| `allocate` | 动词 | 为计算预留逻辑 slot 或 tensor 存储. |
| `allocation` | 名词 | 逻辑 slot 或 tensor 存储的预留. |
| `preempt` | 动词 | 释放请求逻辑块, 将已接受历史重新排队以重计算. |
| `preemption` | 名词 | 逻辑缓存压力下请求退出当前运行集合. |
| `evict` | 动词 | 删除可复用 cache 或 graph 条目以释放容量. |
| `rollback` | 名词 | 拒绝后恢复 computed progress 或 speculative state. |
| `validate` | 动词 | 按数据, schema 或算子合同比较输入, 拒绝无效输入. |
| `validation` | 名词 | 按数据, schema 或算子合同比较输入并拒绝无效输入. |
| `verify` | 动词 | 接受 / 拒绝 speculative token, 或对照独立参考比较输出. |
| `verification` | 名词 | Speculative 接受决策或独立输出比较. |
| `configure` | 动词 | 设置软件运行参数或环境选择. |
| `configuration` | 名词 | 显式软件参数和环境选择. |
| `tokenize` | 动词 | 将文本转换为 vocabulary ID. |
| `detokenization` | 名词 | 将 vocabulary ID 转换为输出文本. |
| `serialize` | 动词 | 将数据编码为 msgpack 或 JSON 字节. |
| `dispatch` | 名词 | 主机选择和 launch 内核实现. |
| `quantization` | 名词 | 使用显式 scale 和舍入转换为 FP8 值. |
| `projection` | 名词 | 使用普通或 FP8 权重的模型 linear 计算. |
| `normalization` | 名词 | 按定义的 norm 和 epsilon 缩放. |
| `recurrent state` | 名词 | GDN 跨 token 计算使用的状态. |
| `convolution` | 名词 | 带持久历史的模型 causal convolution. |
| `alias` | 名词 | Tensor view 间共享存储, 或相同行为的 API 参数别名. |
| `runtime` | 名词 | 活跃推理实现及软件环境. |
| `worker` | 名词 | 负责 device 计算和请求准备的 Python 子进程. |
| `process` | 名词 | 具有独立生命周期的操作系统程序实例. |
| `process` | 动词 | 通过定义的软件操作处理请求数据或 token row. |
| `baseline` | 名词 | 用作性能分母的冻结原始测量. |
| `throughput` | 名词 | 实测工作负载每秒保留的输出 token 数. |
| `TTFT` | 名词 | 从提交到调用方收到首个保留输出 token 的时长. |
| `roofline` | 名词 | 用于分析可达性能的计算和带宽界限. |
| `warmup` | 名词 | 测量重复前不计入测量的完整工作负载. |
| `spread` | 名词 | 单项实测指标的最大减最小值, 除以中位数. |
| `NRMSE` | 名词 | 均方根误差除以参考均方根幅值. |
| `fixture` | 名词 | 受控测试输入或 scripted worker. |
| `harness` | 名词 | 构造用例并采集正式输出 / 计时结果的程序. |
| `provenance` | 名词 | 结果对应的源码, 编译器, 模块和原始哈希身份. |
| `profile` | 动词 | 采集 device 执行 trace 或 counter 用于诊断. |
| `redact` | 动词 | 从派生证据删除主机身份, 原始数据另行保留. |
| `export` | 动词 | 从未改变的原始记录写入可移植派生证据. |
| `repository` | 名词 | 由 Git 管理的项目源码和跟踪文档. |
| `submodule` | 名词 | 固定版本的第三方 Git 源码树. |
| `lockfile` | 名词 | 为可复现安装固定依赖版本的文件. |
| `RPC` | 名词 | Worker 通道上的关联请求 / 回复操作. |
| `IPC` | 名词 | 本地进程间传输 endpoint. |
| `SSE` | 名词 | 用于增量输出的 HTTP server-sent event. |
| `backpressure` | 名词 | 有界 stream 队列满时暂停生成. |
| `timeout` | 名词 | 请求或 RPC 允许的最长等待. |
| `TTL` | 名词 | 保存响应在完成后的有效期. |
| `passage` | 名词 | 用于页面或阅读跳转的 Twine story 单元. |
| `companion` | 名词 | 英文 Markdown 原文在同目录的完整中文译文. |
| `update` | 动词 | 修改软件状态, 配置或对应权威文档. |
| `store` | 动词 | 将数据保存在 device memory 或有界响应存储中. |
| `decode` | 名词 | 逐步计算 target output token 的增量推理阶段. |
| `capture` | 名词 | 记录 CUDA launch 并生成可复用 graph 的操作. |
| `replay` | 名词 | 使用当前输入执行已记录 CUDA Graph 的操作. |
| `profiling` | 名词 | 对 device trace 或 counter 的诊断测量. |
| `lowering` | 名词 | 在 IR 中将 semantic node 替换为所选 provider 的操作. |
| `store` | 名词 | 向 device storage 写入数值的内存操作. |
| `update` | 名词 | 修改配置或持久状态的软件操作. |
| `pin` | 动词 | 固定软件版本或设备身份, 以复现执行. |
| `run` | 动词 | 启动软件命令或程序的执行. |
| `pass` | 动词 | 产生满足全部指定条件的软件测试或门槛结果. |
| `fail` | 动词 | 对软件测试或门槛的一个或多个条件给出不满足要求的结果. |
| `test` | 动词 | 执行软件验证用例, 将行为与指定条件比较. |
| `check` | 动词 | 通过软件校验发现对指定规则的违反. |
| `independent runtime` | 名词 | 推理执行, 环境和编译缓存均不依赖 vLLM 的项目运行时. |
| `independent reference` | 名词 | 使用不同于候选实现的计算方法得到的预期算子输出. |
| `original evidence` | 名词 | 可移植导出前, 完整且字节不变的实测产物. |
| `original record` | 名词 | 可移植导出前, 一份完整且字节不变的实测记录. |
| `original hash` | 名词 | 可移植导出前, 完整实测产物的 SHA-256. |
| `original data` | 名词 | 可移植脱敏或导出前的完整实测数据. |
| `exact division` | 名词 | 得到正确 round-to-nearest 输出的 FP32 除法. |
| `exact scale division` | 名词 | 将 maximum 除以 FP8 range, 得到正确 round-to-nearest 结果的 FP32 除法. |
| `FP32 normal value` | 名词 | 编码 exponent 为 1 到 254 的有限 FP32 数值. 此类不包含零和 subnormal value. |
| `DSpark` | 名词 | 可选 7-token 草稿模式, 使用 5 层 BF16 GQA, target feature 和 Markov / confidence head. |
| `target feature` | 名词 | 一个 target 输入位置的 BF16 target 层输出, 位于下一 normalization 前. 临时验证行也有这些输出. |
| `Markov head` | 名词 | 使用 checkpoint 权重的草稿 projection, 用此前草稿 token 的分数修正每行 base logits. |
| `confidence head` | 名词 | 使用 checkpoint 权重的草稿 projection, 产生条件草稿 confidence, 用于选择草稿数量. |
| `cumulative confidence` | 名词 | 截至当前草稿位置的条件草稿 confidence 乘积. |
| `probability distribution` | 名词 | 配置采样变换及 mask 后的非负 vocabulary 概率, 总概率质量为 1. |
| `conditional distribution` | 名词 | 给定已接受历史及此前推测草稿后, 一个位置的 vocabulary 概率. |
| `stochastic sampling` | 名词 | 从配置的条件 vocabulary 概率中随机选择 token. |
| `hierarchical paired-bootstrap` | 名词 | 先重采样轮次, 再在选中轮次内重采样配对样本, 估计吞吐增益的单侧置信下界. |
| `probability` | 名词 | 单个 vocabulary 结果或接受事件的非负概率质量, 取值为 0 到 1. |
| `confidence bound` | 名词 | 指定置信水平下, 从重采样测量数据计算的统计界限. |
| `greedy sampling` | 名词 | 选择配置模型分数的最大值, 并按确定的 token ID 顺序处理相同分数. |
| `native MTP4` | 名词 | 通过 target checkpoint 自带 MTP 层进行的 4-token 推测. |
| `backbone` | 名词 | 共享 vocabulary head 及 Markov 修正前的 5 层 DSpark attention / MLP. |
| `context injection` | 名词 | 投影已接受 target feature, 并将各层 KV 追加到 DSpark context 存储. |
| `base logits` | 名词 | Markov head 修正前, 共享 target head 产生的草稿 vocabulary 分数. |
| `Markov step` | 名词 | 以此前草稿 token 为输入, 计算一个 proposal row 的草稿 head. |
| `median` | 名词 | 测量样本排序后的中间值, 偶数样本时取两个中间值的平均数. |
| `load` | 动词 | 将 checkpoint tensor 或可执行代码读入推理运行时内存. |
| `initialize` | 动词 | 将运行时对象, device 存储和请求状态设置为规定的初始值. |
| `return` | 动词 | 将软件函数, RPC 或 HTTP 操作的输出数据传给调用方. |
| `commit` | 动词 | 使接受的 token 历史, grammar 或 cache 状态成为后续引擎步骤使用的状态. |
| `reset` | 动词 | 将指定软件计数器, 请求状态或 cache metadata 恢复为初始值. |
| `forward` | 动词 | 通过验收代理传输客户端 HTTP 请求和上游回复, 保持 body 不变. |
| `overwrite` | 动词 | 在保存此前计算值的 device 地址写入新的 tensor 值. |
| `generate` | 动词 | 通过选定推理和采样路径计算模型输出 token. |
| `concatenate` | 动词 | 沿指定维度连接 tensor 值, row 顺序不变. |
| `conditional draft confidence` | 名词 | 给定此前草稿 token 和当前 hidden row, confidence head 在一个 proposal 位置产生的分数. |
| `normalize` | 动词 | 按规定 norm 缩放 tensor 值, 或将 vocabulary 概率缩放为总概率质量 1. |
| `share` | 动词 | 不同软件执行路径使用相同物理模型权重, tensor 存储或 graph 内存 pool. |
| `bidirectional attention` | 名词 | 读取已接受 context KV 及全部 7 个临时 proposal KV row 的草稿 attention. |
| `synthetic token IDs` | 名词 | 按固定索引公式生成的基准输入 ID, 不包含自然语言文本或 tokenization. |

| `DAG` | 名词 | 语义操作及其数据依赖构成的有向无环图. |
| `HBM` | 名词 | 保存参数与持久数据的 GPU 高带宽内存. |
| `SFU` | 名词 | 执行指定原生特殊函数指令的 GPU 资源. |
| `TMEM` | 名词 | 保存 Tensor Core 操作数与累积结果的 SM100 tensor memory. |
| `PDL` | 名词 | 通过 producer trigger 和 consumer 数据等待实现的 CUDA programmatic dependent launch. |
| `CUDA stream` | 名词 | 通过显式跨 stream 依赖事件连接的 GPU 命令队列. |
| `swizzle` | 名词 | 改变 cache 局部性或 memory-bank 访问的 tile 遍历或地址排列. |
| `occupancy` | 名词 | 活跃 GPU warp 数与硬件 warp 容量之比. |
| `REDUX` | 名词 | 执行资源契约单独规定的 PTX warp reduction 指令. |
| `feedback interval` | 名词 | 两个 token 反馈边界之间的语义工作, 使用一次理想 cache 抵扣. |

## 维护

只有批准词典无法准确表达相同技术含义时才新增技术名词或动词.
提供单一定义, 词性, 中文翻译和显式词形.
对照实际源码及 Rule1.5 或 Rule1.12 审查术语.
相同动作可用通用批准措辞表达时, 使用 keep, examine 或 make sure.
代码标识符可独立于正文术语使用库定义名称.
