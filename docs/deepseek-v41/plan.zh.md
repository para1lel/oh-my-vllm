# DeepSeek V4.1 分阶段实施计划

> 代理执行者: 使用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`. 保留阶段门槛和独立审查, 不再等待维护者逐阶段批准.

**目标:** 增加四 GPU DeepSeek 推理, 不损失现有 Qwen 功能.

**架构:** 模型适配器提供数学行为, 状态布局和编解码器. Rust 控制调度和状态事务. Python 执行原生算子.

**技术栈:** Rust, Python, ZMQ DEALER, msgpack, CUDA C++ 或 CuTe DSL, 以及原生 NCCL.

**规格:** [已批准架构](design.zh.md). 从阶段 0 开始执行.

## 全局约束

- 从基线 `b99701a` 在 `nightly` 工作.
- 本会话启动的代理使用 GPT-6.1 Sol 和 `high` effort.
- 保留 Qwen 普通解码, MTP4, 262144 tokens 和单张 B200 配置.
- 增加 DeepSeek TP4, TEP4, DEP4, 五草稿 DSpark, 64 个活跃请求和每请求 1048576 tokens 总上限.
- Engram 权重保存在 pinned host memory.
- 保持 Rust 状态所有权, DEALER/msgpack 和整个 Execute 的 Eager/Graph 规则.
- 不添加外部算子库或 vLLM 依赖.
- 新增测试前说明该测试及现有覆盖的缺口.
- 英文文档少于 2000 words.
- 提供完整中文翻译.
- 使用[测量策略](validation.zh.md), 不新增 README 结果表.

## 审查重点

- Rank 在 collective 期间故障: 整组失败, 不进行部分状态提交. 阶段 2 负责此检查.
- 被拒绝草稿跨压缩边界: 恢复全部已接受状态. 阶段 5 和 6 负责此检查.
- 取消后传输仍在进行: 保留主机及设备存储. 阶段 3 和 5 负责此检查.
- 六十四个长请求竞争内存: 限制分配, 不隐藏串行化. 阶段 5 和 8 负责此检查.
- 模型媒体或工具语法不同: 使用其编解码器, 不引入 Qwen 回归. 阶段 1 和 7 负责此检查.

## 工作方式

每个阶段产出可独立检查的可运行软件, 或冻结的前置契约.
每阶段编码前, 为其接口编写简短的实施清单.
新行为没有现成检查时, 先编写失败的行为检查, 再实施.

阶段 1 的契约抽取可以在阶段 0 参考生成期间进行.
它不得修改正在使用的参考源码, 启用未经验证的 DeepSeek 执行, 或移除阶段 0 的门槛.
冻结的阶段 0 数学和验收契约是阶段 3 算子的前置条件.

不把这份路线图作为一次巨型实现变更.

每个阶段都执行独立审查.
修正问题.
运行适用 hooks.
更新交接.
提交选定文件.
推送工作分支.

明确管理 worker, 服务, GPU 锁和外部 artifacts 的所有权.
停止为该阶段启动的全部进程.

## 阶段 0: 冻结模型与验收契约

**文件:** 新增 `docs/deepseek-v41/model.md` 及其翻译. 新增 `development/model/deepseek_v41/identity.py`.
**输入:** 本地 checkpoint 和固定上游源码.
**输出:** 带版本的身份 manifest, 以及经过批准的数学, 精度, 状态和测量契约.

- [x] 不加载全部张量, 验证全部 48 个分片, 张量名称, offsets, shapes, scales 和文件身份.
- [x] 对比本地权重与固定 DeepInfer revision 及其不同的默认 checkpoint.
- [ ] 明确 CED, CSA2, FP4, Engram 哈希, 提示编码, 媒体编码和 DSpark 概率规则.
- [ ] Kernel 调优前冻结独立参考输入, 完整词表输出, 数值门槛和算子性能门槛.
- [x] 规划逐层流式参考计算, 而非完整的 FP64 权重副本.
- [x] 取得并行定义的批准.
- [x] 解决验证计划中的 HTTP 策略假设.
- [ ] 为组合容量目标计算逐 rank 及主机预算.

**门槛:** 不得有影响阶段 3 算子的未解决权重映射或验收规则.
**测试理由:** 现有身份检查只接受一种 Qwen 配置, 无法发现 DeepSeek 格式不匹配.

## 阶段 1: 抽取模型契约并保持 Qwen 行为

**文件:** 新增 `crates/model-contract/src/lib.rs`, 对应 Cargo manifest 和 `python/oh_my_vllm/models/{api,registry}.py`.
修改 `Cargo.toml`, `crates/zmq-worker/src/{main,protocol}.rs` 和 `python/oh_my_vllm/models/qwen38/inference.py`.
分小步将模型专属代码移入 `python/oh_my_vllm/models/qwen38/`.
**接口:** 定义 `ModelDescriptor`, `CacheSchema`, `ParallelPlan`, `SpeculationSpec` 和 `ModelAdapter`.
Descriptor 绑定身份, 上下文上限, 能力, 格式和 schema 版本.
Adapter 提供准备, 执行, 结果验证, 状态选择和资源关闭.

- [x] 为未知架构, 不支持格式和能力不匹配增加 registry 拒绝检查.
- [ ] 将固定 Qwen 维度, tokenizer 规则, 媒体位置和 MTP4 选择移出共享控制代码.
- [ ] 保持 Qwen 当前的 API 校验和默认值.
- [ ] 执行现有 Rust, 服务, 约束, 执行和冻结的 Qwen 数值检查.

**门槛:** Qwen 经 adapter 运行, 已接受输出和状态语义不变.
**测试理由:** 现有检查未覆盖模型族之间的选择.

## 阶段 2: 增加类型化缓存组和四 rank 控制

**文件:** 修改 `crates/kv-cache/src/{group,coordinator,snapshot}.rs`, `crates/scheduler/src/{lib,cache,output}.rs` 和协议.
新增 `crates/zmq-worker/src/group.rs` 和 `python/oh_my_vllm/operators/runtime/distributed.py`.
扩展 `scripts/with-gpu.sh` 和 `development/serving/device.py`.
**接口:** 在阶段 1 契约中增加 `StateTransaction`, `RankReady`, `RankResult` 和 group epoch 字段.

- [ ] 泛化关联状态预留, 不改变 Qwen copy-on-write 行为.
- [ ] 增加带版本的 rank 就绪, 容量汇总, 完成校验, 截止时间和整组关闭.
- [ ] 按稳定顺序获取全部四张 GPU 的锁, 否则等待前释放所有已取得的锁.
- [ ] 获取锁后重新检查全部选定设备. 不修改其他用户的时钟或进程.
- [ ] 测试空 rank, 不均匀调度, 过期结果, 启动故障和 collective 中途故障.
- [ ] 验证 decode graph capture 和 replay 的固定 collective 顺序与缓冲区生命周期.

**门槛:** 四 rank 合成执行不能发布部分结果或泄漏进程和锁.
**测试理由:** 单 worker 生命周期检查无法发现分布式提交或 collective 故障.

## 阶段 3: 加载权重并实现原生基础算子

**文件:** 在 `python/oh_my_vllm/` 下新增 `models/deepseek_v41/{config,weights}.py`.
增加算子包 `fp4`, `moe`, `engram`, `hyper_connection`, `compression` 和 `sparse_attention`.
扩展 `development/native_build.py` 和 `development/operators/`.
**接口:** 权重计划使用身份 manifest 和并行计划. 算子采用现有 API/plan/workspace 模式.

- [ ] 为量化, 路由, 压缩及 Engram hash/lookup 边界编写小型固定参考 fixtures.
- [ ] 实现原生 FP8/FP4 解包, scale 解释, 投影, 路由, hyper-connections 和稀疏注意力基础算子.
- [ ] 加载 rank 拥有的权重切片, 不完整展开为 FP16 或 FP32.
- [ ] 分配带有界暂存的 pinned Engram 存储, 与 host KV 分开计费.
- [ ] 检查共享注册, NUMA 放置, 内存压力, 取消和传输完成所有权.
- [ ] 执行数值检查, 算子延迟检查, 并对改变的内存路径运行 Compute Sanitizer.

**门槛:** 算子符合已批准参考, 且不超过启动预算.
**测试理由:** Qwen fixtures 不覆盖 FP4, UE8M0, 稀疏路由或主机端模型权重.

## 阶段 4: 在全部并行策略运行普通解码

**文件:** 新增 `models/deepseek_v41/{model,parallel,cache}.py` 和原生通信计划.
在 `development/model/deepseek_v41/` 新增全模型捕获及参考适配器.
**接口:** 使用阶段 1 descriptors, 阶段 2 transactions 和阶段 3 operators.
通过通用 adapter 产出目标 logits 和已接受状态.

- [ ] 减少 decoder 工作量前, 先实现完整短上下文 CED/CSA2 目标计算.
- [ ] 先实现 TP4.
- [ ] TP4 之后实现 TEP4.
- [ ] TEP4 之后实现 DEP4.
- [ ] 明确共享专家, 输出 heads, 视觉及后续 DSpark 权重的放置和通信.
- [ ] 测试不均匀专家负载, 零 token rank, top-k 同分和请求顺序变化.
- [ ] 在 Eager prefill 和 graph decode 中, 将全部 129280 个端点 logits 与冻结参考比较.

**门槛:** 三种策略均可通过离线和 HTTP 路径提供普通文本推理.
**测试理由:** 算子一致性不能单独证明完整分布式模型组合正确.

## 阶段 5: 达到长上下文与 64 请求容量

**文件:** 扩展 DeepSeek cache adapter, Rust 类型化 groups, scheduler 和 worker offload 路径.
**接口:** Cache schema 指定共享 producers, 持久状态, 未完成组, rollback 和 rank 放置.

- [ ] 实现 chunked prefill, 有界 index workspaces, 压缩共享 KV 和私有推测状态.
- [ ] 实现前缀复用, 压力驱动 offload, 精确恢复和安全的部分前缀续算.
- [ ] 选择减少工作量的路径前, 将 decoder replay 与完整计算对照验证.
- [ ] 未经单独批准和数值门槛约束, 不用近似有界 replay 替代精确恢复.
- [ ] 覆盖 400000, 800000 和 1048576-token 边界, 并预留输出容量.
- [ ] 每种策略使用普通解码测试 64 个活跃长请求.
- [ ] 测试内存压力下取消及反复前缀驱逐和恢复.

**门槛:** 逐 rank 内存计费与分配一致, 普通解码容量用例完成.
最终 DSpark 容量检查属于阶段 6.
**测试理由:** 现有 32769-token Qwen 检查不能证明压缩缓存边界或四 rank 长上下文容量.

## 阶段 6: 增加五个草稿 token 的 DSpark

**文件:** 新增 `models/deepseek_v41/dspark.py` 和 `operators/dspark/`.
扩展共享推测解码元数据, scheduler 预算, constraints 和 traces.
**接口:** Proposal 结果包含 token IDs, proposal probabilities, confidence, positions 和私有状态引用.
Acceptance 将提交长度与新 proposals 分开返回.

- [ ] 实现三个预测层, 最多提出五个 token, 而非五个循环 MTP 层.
- [ ] 先实现定长验证, 最多六个目标行.
- [ ] 定长 acceptance 通过参考检查后, 再加入置信度调度验证.
- [ ] 保持精确概率修正, 除非明确批准其他有界误差规则.
- [ ] 测试各位置拒绝, 全接受, EOS, stop strings, 输出上限, schema masks 和传输恢复.
- [ ] 测试部分验证和 batch 重排下的 RNG 与 penalty 历史.
- [ ] 每种策略使用 DSpark 执行最终组合容量检查.

**门槛:** DSpark 在三种策略中工作, 不改变已批准的采样或状态契约.
**测试理由:** Qwen MTP4 无法发现 DSpark block, confidence 或六行验证错误.

## 阶段 7: 完成 API 和媒体功能对齐

**文件:** 增加 DeepSeek prompt, parser, media 和 vision adapters. 扩展 `tests/test_service.py` 和现有约束检查.
仅为按模型选择的行为修改 `crates/zmq-worker/src/serving/`.
**接口:** 保持共享请求/响应对象. 模型 adapters 提供编解码器和能力校验.

- [ ] 完成验证计划中的功能清单.
- [ ] 对工具, 推理, 图像和会话续接, 将编码与固定的本地参考 tokens 比较.
- [ ] 实现经过验证的视频帧顺序和时间戳, 不采用 Qwen 位置假设.
- [ ] 通过 HTTP streaming, 离线请求, 取消和 worker replacement 覆盖普通及 DSpark 路径.

**门槛:** DeepSeek 不缺少任何已声明与 Qwen 对等的功能.
**测试理由:** 文本推理不覆盖模型专属工具语法或多模态历史.

## 阶段 8: 测量, 审查与交付

**文件:** 泛化 `development/serving/{benchmark,agentx,device,trace,accounting,workload}.py`.
新增 `development/serving/profiles/deepseek_v41.py`.
扩展现有 driver validation tests.
将批准的结果写入验证计划指定的独立文档.

- [ ] 验证精确的用例枚举, 四 rank 身份, 提示长度, 输出长度和请求关联.
- [ ] 仅在正确性门槛通过后执行批准的 HTTP 和 AgentX 矩阵.
- [ ] Qwen 回归测量与 DeepSeek 结果分开.
- [ ] 报告实测延迟, 吞吐, 接受率, 内存, 失败和资源身份, 不复制 Qwen efficiency 数值.
- [ ] 遵循工作方式中的阶段交付流程.

**门槛:** 结果绑定交付源码, native modules, 权重, tokenizer, 拓扑和测量配置.
**测试理由:** 现有 drivers 固定 Qwen 用例大小, 单张 GPU 和四个 AgentX 会话树.
