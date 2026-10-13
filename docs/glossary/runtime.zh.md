# 术语参考 B

| 术语 | 词性 | 项目含义 |
|---|---|---|
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
