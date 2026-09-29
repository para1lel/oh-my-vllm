# ADR-008：语义算子 IR 与 GPU 前向编译

状态：用户于 2026-09-29 确认设计；实现与验收进行中。

## 决策

以 PyTorch `torch.library` custom op 为项目自有 CUDA 和关键 FlashInfer 计算建立不透明语义节点。每个算子由可执行的 PyTorch 参考、显式 fake shape/dtype 及修改 schema 定义。生产 provider 独立注册，使用静态元数据能力检查和有序优先级。参考仅是显式选择的调试 provider，不自动回退。eager 与 compiled 执行共用 provider 选择；首次选择后冻结注册配置。

以 fullgraph torch.compile 编译 prefill、target decode、MTP draft 和四步 proposal；一个小型 FX backend 在调用 Inductor 前降低语义节点。现有手工 CUDA Graph 捕获已编译的 GPU 单元。Rust 调度、Python 主机计划及持久缓存分配留在编译单元外。保留受检查的调用点清单；包括 `F.linear` 在内的普通 PyTorch 运算不列入语义算子范围。

可用时优先采用公开 `torch.library` 和 `torch.compile` 接口。受限的 FakeTensor、FX 元数据与 Inductor backend lookup 调用使用 PyTorch 2.14.0 内部接口，需要固定版本回归测试。缓存写入须显式且有序；持久 KV/GDN buffer 不能捐赠。provider 选择仅依赖阶段、shape、dtype、layout 和设备，不依赖张量值。编译或 provider 失败须报错，不自动回到 eager。

## 结果与验证

模型调用路径使用语义名称，而非直接 CUDA/FlashInfer 入口。基础 FP8 quantization 与融合 FP8 linear/SiLU quantization 各有语义名称。计划型 attention 通过存活的 CPU handle 引用主机 plan 状态，同时将动态 decode 表/长度和原生子页元数据作为显式图输入。编译后的 target、draft、proposal 闭包在手工捕获前预热，捕获恢复仍具事务性。

经过精确输出等价测试的后期 FX 改写，仅在相邻、单次使用、BF16 输入且生产 provider 一致时融合 `silu_mul -> fp8_linear`。编译器适配层关闭 Inductor CUDA Graph，并将 PyTorch 2.14.0 的两个重编译上限提高至 4096。每个前向单元每编译 256 个变体发出警告。已测形状族之外的长期编译缓存增长及逐出后重捕获仍是限制；4096 不是显存或主机内存容量保证。没有证明安全的临时目的地时，不启用 activation 捐赠。

`tests/test_ir_*` 检查注册、契约、provider 选择、静态覆盖和真实模型 fullgraph/CUDA Graph 执行。完整的现有正确性、最大上下文、算子及 12 组框架门禁仍是验收条件；仅 torch.compile 成功不能豁免其他门禁。

设计参考：[vLLM IR](https://docs.vllm.ai/en/latest/design/vllm_ir/)。本项目实现自己的受限 DSL，不依赖 vLLM runtime。
