# Tile 执行实施计划

> 代理执行时: 使用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development` 逐任务实施.

**目标:** 将全部 GPU tensor 计算迁移到依赖 tile 的 MegaKernel 执行, 并通过最终 HTTP 性能门槛.

**架构:** 统一计划接口提供有限 task graph, 运行时 metadata 和状态权限.
专用 persistent kernel 可以共享其依赖契约.
Rust 保持服务调度和逻辑 cache 所有权.

**技术栈:** Rust, Python, ZMQ DEALER, msgpack, CUDA C++/CUTLASS 及允许的 CuTe DSL.

**规格:** [Tile 执行设计](tile-execution.zh.md).
本计划提供实施步骤. 范围和验收以设计为准.

## 全局约束

- 使用 `nightly`, 单张 NVIDIA B200 和 Qwen3.8-27B-FP8.
- 保持 48 层 GDN, 16 层 FA, 块大小 784 和 262144-token 总长度上限.
- 保持模型值, 数值容差, 私有状态规则及 HTTP/离线功能.
- 最终目标覆盖全部 GPU tensor 计算, 也包含初始化和准备阶段.
- 迁移期间不设性能门槛. 最终全部 24 种 HTTP 配置必须达到至少 50%.
- 构建工具与运行时导入保持分离.
- 本工作中的 subagent 使用 GPT-6.1 Sol, high effort.
- 新增每个测试前说明必要性. 尽量复用独立参考.
- 每轮交付前取得独立审查.
- 每轮交付前运行适用 hooks.
- 每轮交付前更新两种语言 README 表格.
- 只提交选定文件.
- 每轮交付后推送工作分支.
- 报告放在仓库外. 使用后停止临时进程.

## 审查重点

- Epoch 复用和满队列不得发布旧数据或阻止执行终止.
- 物理调用切分不得增加参考工作量或虚构 MTP 工作.
- 位置 783, 784 和 785 必须在计划复用时遵守状态和 page 规则.
- MTP 拒绝和短尾部必须保持已接受历史, 约束, 状态槽及 RNG 一致.
- Vision 预处理和启动 tensor 准备不得遗漏在最终 task 覆盖审计之外.

## 任务 1: 将 HTTP 计费与旧算子工具分离

**文件:** 新建 `development/serving/accounting.py` 和 `development/native_build.py`.
修改该目录中的 `development/serving/roofline.py`, `trace.py`, `benchmark.py` 和 `test_roofline.py`.
将必要参考移至 `development/reference/`.
修改 `.pre-commit-config.yaml`, 安装命令及相关双语指南.

**接口:** `canonical_passes(events: list[dict], manifest: dict) -> list[dict]` 提供逻辑 pass.
`reference_work(passes: list[dict]) -> dict[str, float]` 提供字节量及按精度划分的工作量.
`reference_seconds(work: dict[str, float]) -> float` 使用设计规定的标称常量.
构建入口为 `python -m development.native_build`, 使用现有构建选项.

- [ ] 在 `test_roofline.py` 中增加一个参数化计费回归测试.
  它必须检测额外物理切分, task 数量变化, padding 或状态复制造成的计费膨胀.
  它还必须拒绝虚构 proposal 和不一致的已接受位置.
  根据 benchmark manifest 检查重复/缺失请求 ID 和不同到达顺序.
  现有测试绑定旧校准机制, 不能证明新不变量.

- [ ] 运行 `scripts/with-env.sh python -m unittest development.serving.test_roofline`, 观察新增测试失败.

- [ ] 实现确定的参考调度, 使用 32768-token 预算, decode 优先级和最多 4 个运行请求.
  使用 benchmark 的请求序号, 按序分配余量, 并使 prefill chunk 与 cache 边界对齐.
  重放记录的 MTP 依赖, 不创建新的 proposal 序列.

- [ ] 将 benchmark manifest 绑定到产物身份.
  将客户端请求序号映射到响应/trace ID, 模式及配置的输入/输出长度.
  参考分组前拒绝重复 ID, 成员不完整和长度不一致.

- [ ] 将参考数学工作量与实现选择的算法分离.
  明确标量工作及省略项. 排除可消除的中间流量和重复工作.

- [ ] 用固定标称吞吐和计费版本替代校准输入.
  保持硬件, 模型, 可执行文件, 源码和计时身份.

- [ ] 删除前先迁出 native builder, 独立数值参考和状态 fixture.

- [ ] 移除旧算子计时/审计命令, 公开实现比较工具, 峰值校准及未使用的辅助工具.
  在同一变更中更新导入, hooks 和命令.

- [ ] 运行计费及已迁移参考的测试. 更新全部 24 行 README 数据, 不混合计费版本.

- [ ] 取得独立审查. 运行适用 hooks. 提交选定文件. 推送工作分支.

## 任务 2: 构建 task 引擎及其资源契约

**文件:** 新建 `python/oh_my_vllm/kernels/tile_plan.py`, `mega.py`, `cuda_backend/mega.cu` 和 `cuda_backend/tile_runtime.cuh`.
新建 `development/tile/test_engine.py`, 并在 `development/tile/` 中添加 task fixture.
更新 native builder 和 loader.

**接口:** `mega.py` 提供 `prepare(kind, capacity, bindings) -> Plan`, `execute(plan, metadata) -> Completion` 和 `close(plan) -> None`.
`Plan` 拥有 graph descriptor, task 资源, 稳定缓冲区和 native lease.
`StepMetadata` 包含 epoch, query 长度, page table, 读写槽和有效行.
`Completion` 提供 epoch, 错误状态和已完成输出.

- [ ] 增加一个引擎协议测试, 使用小型独立 producer/consumer graph.
  覆盖多输入依赖, epoch 重复, 缓冲区复用, 队列容量失败及无效 metadata.
  数值参考不能测试队列发布或执行进展.

- [ ] 运行 `scripts/with-env.sh python -m unittest development.tile.test_engine`, 在实现前观察失败.

- [ ] 实现不可变拓扑, 运行时 task 范围, 有界队列及数据/状态/生命周期依赖.

- [ ] 实现发布顺序, 异步完成, 错误状态及有限终止.
  检查整个 launch grid 的 occupancy 和进展, 而不只检查单一 task 类型.

- [ ] 将失败接入 worker 移除和 epoch 清理流程. 不复用部分修改的状态.

- [ ] 运行 CPU 协议检查, 并通过 `scripts/with-env.sh scripts/with-gpu.sh` 运行 GPU fixture.
  对 GPU 协议 fixture 使用现有 step deadline 和 Compute Sanitizer.

- [ ] 取得独立审查. 运行适用 hooks. 提交选定文件. 推送工作分支.

## 任务 3: 完成普通 decode

**文件:** 在 `cuda_backend/tile_tasks/` 中添加语言模型和 tensor task 实现.
修改 `worker/model.py`, `worker/inference.py`, `worker/sampling.py` 及 FFN/FA/GDN wrapper.
在 native 接口中将词表投影与采样分离.

**接口:** Decode `Plan` 消费现有模型绑定和 step metadata.
它提供 hidden 结果, logits 及 candidate cache/状态写入.
独立采样消费这些 logits, 不重复执行词表投影.

- [ ] 改造现有数值和状态检查, 将完整新 decode 路径与独立参考比较.
  覆盖支持的 batch 容量, 无效行, page table 变化及位置 783/784/785.

- [ ] 通过 task 执行实现 embedding, 全部 64 层, 最终归一化和词表投影.
  各层共享有限 task 实现. 明确精度和状态权限.

- [ ] 通过 CUDA Graph 提交 decode 计划, 保持地址稳定并更新 metadata.

- [ ] 运行改造后的检查及普通 HTTP 配置.
  正确性通过后删除被替换的 decode 实现.

- [ ] 取得独立审查. 运行适用 hooks. 提交选定文件. 推送工作分支.

## 任务 4: 完成 prefill

**文件:** 在 `cuda_backend/tile_tasks/` 中添加 prefill task 实现.
修改 `tile_plan.py`, `worker/model.py` 和 FA/GDN/FFN wrapper.

**接口:** Prefill `Plan` 提供 prompt hidden 行, 必要 checkpoint 和端点结果.
它使用与 decode 相同的完成和状态权限契约.

- [ ] 改造现有 FA 和 GDN 参考, 覆盖可变长度, chunk 边界和非连续 page table.

- [ ] 实现大型投影, attention, GDN chunk, metadata 和归约 task 类别.
  保持规定的 FP32 live carry 和 BF16 checkpoint 行为.

- [ ] 按模型契约检查分块执行, prefix reuse 和末行状态.

- [ ] 运行改造后的检查和输出 64 个 token 的 prefill 敏感 HTTP 配置.
  删除被替换的 prefill 计算路径.

- [ ] 取得独立审查. 运行适用 hooks. 提交选定文件. 推送工作分支.

## 任务 5: 完成 MTP 和采样

**文件:** 在 `cuda_backend/tile_tasks/` 中添加采样和 MTP task 实现.
修改 `worker/speculation.py`, `worker/draft_chain.py`, `worker/sampling.py`, `kernels/sample.py` 和 `kernels/mtp.py`.

**接口:** 采样计划消费 logits 和采样 metadata.
MTP 计划提供 proposal, 接受数量, 选定状态槽和 RNG 结果.
CPU 约束状态只在取得已接受 token 结果后改变.

- [ ] 改造固定种子采样及 MTP 参考, 覆盖 0 至 4 个提案接受和短尾部.
  保持 penalty, grammar, 随机, greedy 及混合 batch 覆盖.

- [ ] 实现 context, proposal, verification, acceptance, gather, 排序和随机选择 task.

- [ ] 在服务契约使用 CPU 决策的位置保持该边界.
  有限提交可以为 CPU 决策返回, 但不能因此把 GPU tensor 工作留在旧路径.

- [ ] 运行改造后的参考, 约束测试和全部 18 种 MTP4 HTTP 配置.
  删除被替换的采样和 MTP 实现.

- [ ] 取得独立审查. 运行适用 hooks. 提交选定文件. 推送工作分支.

## 任务 6: 完成 vision

**文件:** 在 `cuda_backend/tile_tasks/` 中添加 vision task 实现.
修改 `worker/media.py` 和 `kernels/vision.py`.

**接口:** Vision 计划消费已解码媒体 tensor 和 shape metadata.
它提供有序语言 embedding, 保持现有图片/视频位置语义.

- [ ] 改造 vision 参考, 覆盖像素缩放, 帧隔离, patch 顺序, 不完整帧组和 spatial merge.

- [ ] 通过 task 依赖实现 GPU 预处理, 位置, 全部 27 层, merge 和 embedding gather.

- [ ] 使用推理 worker 检查图片和视频服务路径, 不使用 placeholder.
  纯文本 HTTP 性能表不能证明 vision 正确性.

- [ ] 改造后的检查通过后, 删除被替换的 vision 实现.

- [ ] 取得独立审查. 运行适用 hooks. 提交选定文件. 推送工作分支.

## 任务 7: 补齐准备阶段覆盖并完成最终验收

**文件:** 在 `cuda_backend/tile_tasks/` 中添加准备和辅助 task 实现.
修改 `worker/weights.py`, `worker/device.py`, `worker/cache.py` 及 native 注册/构建列表.
在 `development/tile/test_engine.py` 中增加覆盖检查.

**接口:** 准备计划使用相同引擎处理权重转换, scale, 表和 tensor 初始化.
覆盖记录将每个运行时 GPU 计算入口映射到 task 类别和正确性检查.

- [ ] 完成权重准备, 激活表, 索引传输, 初始化及辅助 tensor task.

- [ ] 对照 native 入口和执行 trace 审计启动, warmup, capture, replay 及错误路径.
  普通 cuBLAS/CUB/CUDA tensor 调用不能作为最终绕过路径保留.
  CUDA 传输和资源管理操作继续作为基础设施.

- [ ] 增加覆盖回归测试, 拒绝未分类的运行时 tensor 计算入口.
  现有输出比较无法检测仍能产生正确结果的旧计算路径.

- [ ] 删除未使用的算子代码, 注册, 构建输入和过时指南.

- [ ] 运行全部适用正确性检查, hooks 和完整 24 种 HTTP 比较.
  调整 task 算法和资源布局, 直到每行均在规定精度下达到 50%.

- [ ] 最终产物绑定同一源码/计费身份. 更新两种语言 README 表格.

- [ ] 取得整体验证审查. 修正问题. 提交选定文件. 推送工作分支. 停止临时进程.

## 计划更新

首个 decode 里程碑后, 研究或测量可以改变 task 实现, kernel 类别和迁移顺序.
这些选择变化时, 更新本计划和中文翻译.
保持最终覆盖范围, 数值契约和 HTTP 门槛不变.
