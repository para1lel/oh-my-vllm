# ADR-002：适配已安装的 GPUWorker，不导入其调度器

2026-09-19 接受，遵循已批准的 Rust 调度/Python runner 边界。
**已被取代**：先后被 ADR-005 和 ADR-006（2026-09-21）取代；vLLM 适配器和物理 stride 重映射已不存在。保留作为历史记录。

当时安装的 vLLM 是 `/data0/shared/dongwu.chen/vllm` 的 editable checkout，版本元数据为 0.28.1rc1.dev691+g6376c601e.cu131。worker 类名为 `Worker`；初始化使用当前 VllmConfig 上下文，依次进行设备/加载、显存分析、KV 配置、initialize_from_config 和 compile_or_warm_up_model。EngineArgs 只做配置归一化，不实例化 vLLM 调度器。纯文本模式避免初始化范围外的视觉编码器。

实际物理 KV 布局依次为三个 Mamba 组和一个 FA 组。Rust 保留两种逻辑类型和共享块 ID 池。Python 为 Mamba 表的各组映射不同物理 ID，并按 spec 类型映射 FA，绝不假设组顺序。物理组共享 tensor 存储，因此 Mamba 组重复使用相同 ID 会破坏层状态。令 stride=max（每种类型的物理组数），逻辑块 b 映射为 b*stride+group_offset；null 始终映射到 0。Rust 共享池保证 FA 与 Mamba 的逻辑 ID 不同。Ready 返回 floor(physical_capacity/stride)，避免 Rust 分配越界。这样会为 FA 保留未使用地址，未来可用加权物理分配器回收。

1024 个物理块对应 341 个逻辑槽，足以覆盖四个 36864-token 普通请求所需约 200 个活跃逻辑块；当时长工作负载仍在验证。断言逻辑和物理块大小保持 784。模型加载后需完成平台设置，以设置 Mamba 块大小和页填充。

SchedulerAdapter 单独跟踪 worker admission，不以 num_computed_tokens 代替：前缀命中请求对 worker 仍是新请求。运行中的更新追加新块 ID；恢复请求替换块表。结果按 req_id 匹配，不按调度顺序。空 token 列表表示中间 prefill，多 token 列表保留全部被接受的推测输出。即便没有待调度模型 token，也要向 worker 刷新完成信息。

真实推理已通过 16-token 冒烟测试。完整功能、实际路径精度和性能验收分别由 handoff.md 跟踪。性能测量不得启用运行时探针。
