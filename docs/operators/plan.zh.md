# 算子重构实施计划

> Agent 指令: 选择执行方式后, 使用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`. 按顺序完成复选框步骤.

**目标:** 将各算子的接口, 计划, 临时存储和 CUDA 代码放入同一个包.

**架构:** 使用已确认的[设计](design.zh.md). 源码迁移期间保留 7 个 native 模块. 按依赖顺序改变接口.

**技术栈:** Python, Rust, TVM FFI, CUDA 13.1, CUTLASS 4.8.0, 以及单张 NVIDIA B200.

**规格:** [算子结构设计](design.zh.md).

## 全局约束

- 使用 `main` 和[贡献流程](../../CONTRIBUTING.zh.md).
- 保持[运行时契约](../requirements.zh.md), [模型精度](model.zh.md)和[执行规则](../execution.zh.md).
- 推理代码不得导入开发工具或执行编译.
- 使用各算子族自己的类型. 不引入算子基类或 backend registry.
- 显式传入权重. 持久状态所有权保留在 worker 和 Rust 中.
- 录制前提供具名借用输出. 重放期间保持地址固定.
- 顺序执行的层复用临时存储. Graph 释放前保留全部录制资源引用.
- 保持冻结的短输入参考及其数值门槛. 接口变更前使用链接的[验证计划](validation-plan.zh.md).
- 应用固定的 microbench 延迟倍率门槛. 使用全部 24 组 HTTP 配置进行集成检查.

## 审查重点

| 条件 | 必要行为 | 任务 |
|---|---|---|
| Include 跨越包目录 | 源码摘要包含全部传递依赖 | 1 |
| Buffer 大小相同但录制的 launch 不同 | Plan 签名防止错误复用 Graph | 3 |
| 多层权重及录制失败 | Owner 保持存活, 直到按顺序释放 Graph | 3 |
| Stop 截断和三个 RNG 域 | 有效数量控制状态选择, counter 保持隔离 | 5 |
| Vision 首末层及奇数帧数 | 全模型端点和媒体顺序保持正确 | 6 |

## 文件与接口映射

任务 1 前, 在当前运行时上完成验证任务 V1–V3.
经审查的 fixture 和倍率门槛控制后续迁移.

以下 Python 包路径相对于 `python/oh_my_vllm`.
开发, 测试和工具路径相对于仓库根目录.

新包根目录为 `python/oh_my_vllm/operators`.
在 `fa`, `gdn`, `ffn`, `output`, `mtp` 和 `vision` 中分别创建 `api.py`, `plan.py` 及 `workspace.py`.
CUDA 阶段放入各算子族的 `cuda` 目录.
共享资源代码放在 `runtime` 中. 共享计算放在 `primitives` 中.

各算子族定义自己的 `Weights`, `Plan`, `Workspace`, 输入, view 和输出.
保留现有 native 契约中的张量 shape 和数据类型.
Plan 和 view 使用 frozen data class.
准备阶段使用现有 `Device` 分配接口.
构建 plan 时不得分配 GPU 存储或执行 GPU 工作.

该命名约定不引入基类:

```python
prepare(plan: Plan, device: Device, weights: tuple[Weights, ...]) -> Workspace
workspace.outputs: Outputs
workspace.graph_lease()  # Existing context-manager lifetime contract.
```

准备阶段创建存储和 native context, 并执行各算子族自己的 warmup.
其权重 tuple 包含该算子 workspace 使用的全部层绑定.
Worker 组合各算子族的 `plan.reuse_key`, 并独立检查资源身份.
Plan 提供启动容量计算所需的存储描述. 使用计数分配器的构造函数不再选择实现.

## 任务 1: 迁移源码及其工具

**依赖:** 验证任务 V1–V3. **结果:** 现有计算从新目录运行.

**文件:** 将 `kernels/native.py` 移到 `operators/runtime/native.py`, 将 `worker/device.py` 移到 `operators/runtime/device.py`.
将 `kernels/specification.py` 移到 `operators/specification.py`, 将 kernel 目录指令移到 `operators`.
各 wrapper 移到对应算子族的 `api.py`. 将 `sample.py` 移到 `output/api.py`.
将 `device.cu` 移到 `runtime/cuda/device.cu`, 其余 6 个编译单元移到对应算子族目录.
将 `sample.cu` 放入 `output/cuda`.

**接口:** 本任务保持 native 模块名和导出函数不变.
在 `runtime/native.py` 中增加 `MODULE_SOURCES: dict[str, Path]`.
`development/native_build.py` 和 `source_fingerprint(name)` 使用此映射.

- [ ] 检查验证任务 V3 的变更前 manifest 和测量结果.
- [ ] 为 `development/model/test_identity.py::test_native_identity_tracks_only_transitive_dependencies` 增加兄弟目录 include 和 include 环.
- [ ] 检查传递依赖修改会改变摘要, 无关修改不会改变摘要.
- [ ] 将算子族前缀的 CUDA header 随算子族迁移. 共享 header 按现有使用者和职责迁移.
- [ ] 对相对于算子源码根目录的依赖路径及文件内容一起计算摘要.
- [ ] 修改带引号 include, builder include 根目录, 全部推理/开发/测试导入及源码链接.
- [ ] 更新 `scripts/format_cuda.py` 和 `.pre-commit-config.yaml`, 使 CUDA 的 800 行限制继续生效.
- [ ] 将 `tests/test_execution.py` 适配到新的 Device 导入, 保持失败断言不变.
- [ ] 使用现有短输入 FP64 端点及全部 7 个重建模块完成下方交付流程.

## 任务 2: 增加全模型验证输入

**依赖:** 任务 1. **结果:** 数值证据覆盖后续任务将改变的路径.

**文件与接口:** 使用[验证计划](validation-plan.zh.md)中的验证任务 V4.

- [ ] 在 FA/MTP 变化前增加 MTP 和长上下文端点.
- [ ] 在 vision 变化前增加媒体端点.
- [ ] 将参考生成和产物身份与推理变更分开审查.
- [ ] 在推理行为不变的条件下完成交付流程.

## 任务 3: 应用 FA 计划和 MTP 组件接口

**依赖:** 任务 1, 以及任务 2 的 MTP/长上下文检查. **结果:** FA 及其 MTP 使用者采用显式权重和具名存储.

**文件:** `operators/fa/{api,plan,workspace,components}.py`, `operators/mtp/{api,plan,workspace}.py`, 以及 `worker/{model,weights,metadata,speculation,draft_chain}.py`.
修改 `tests/test_execution.py` 和 `tests/test_constraints.py` 的 draft-chain fixture.

**接口:** FA 定义以下函数:

```python
make_plan(*, rows: int, max_queries: int, max_context: int,
          min_context: int, variant: str = "full", first_token: int = 0) -> Plan
run(weights: Weights, workspace: Workspace, inputs: Inputs,
    batch: BatchView, state: CacheView) -> Outputs
```

`Inputs` 包含 `value` 和 `residual`. 完整 FA 输出具有相同的具名字段.
`BatchView` 包含 `positions`, `sequence_rows` 和 `lengths`.
`CacheView` 包含 `cache`, `slots` 和 `tables`.
`components.py` 中的私有组件类型为 MTP 提供 projection 和 attention 的存储契约.

- [ ] 新增 `test_fa_plan_separates_recorded_launches`, 覆盖变体, 行数 32/33 和 128/129, 以及分派边界 1024, 12544 和 32768.
- [ ] 检查存储大小相同但 launch 条件不同的情况. 检查兼容元数据更新不需要新 plan.
- [ ] 使用假 device 和多层权重增加 `test_operator_workspace_keeps_outputs_and_graph_owners`.
- [ ] 检查固定输出身份, 录制期间无分配, 顺序复用 scratch, 以及 Graph 销毁后才释放资源.
- [ ] 实现前运行新检查, 显示尚缺的行为.
- [ ] 将分派和准备决策从 `worker/model.py` 移到 FA plan.
- [ ] 用显式组件类型替换 MTP 的 buffer 索引访问和复制的 FA 布局.
- [ ] 为旧 FFN/output 组件保留具名私有适配器, 直到任务 4 和 5 将其移除.
- [ ] 使用 FA/MTP microbench 用例和 V4 全模型端点完成交付流程.

## 任务 4: 应用 GDN 和 FFN 契约

**依赖:** 任务 3. **结果:** 全部 target 层使用显式准备后的权重和可复用算子存储.

**文件:** `operators/{gdn,ffn}/{api,plan,workspace}.py` 和 `worker/{model,inference,weights,metadata,cache,speculation}.py`.

**接口:** FFN 使用 `run(weights, workspace, inputs) -> Outputs`, 采用算子族自己的类型和具名 `value`/`residual` 字段.
GDN 使用 `run(weights, workspace, inputs, batch, state) -> Outputs`.
其 view 使用现有 `Schedule` 字段及 recurrent/convolution 张量.
输入还为可选 token 和 media 张量提供名称.

- [ ] 移除 `gdn.metadata[2]` 访问前, 为 hidden 导出提供由 worker 持有的具名 write-slot view.
- [ ] 将 GDN chunk 选择, 元数据准备和 FFN 分派移到各算子族 plan.
- [ ] 在 `worker/model.py` 和 `worker/inference.py` 中用 plan 存储描述替换启动计数构造函数.
- [ ] 从使用者中移除可变权重绑定和 FFN 私有 buffer 访问.
- [ ] 将任务 3 的生命周期断言应用到 GDN/FFN 准备路径.
- [ ] 使用 GDN/FFN microbench 用例及 V4 的短输入, chunk 边界和长上下文端点完成交付流程.

## 任务 5: 拆分 output 和 MTP acceptance

**依赖:** 任务 4. **结果:** 输出计算和候选接受具有已确认的语义边界.

**文件:** `operators/{output,mtp}/{api,plan,workspace}.py`, 其 CUDA 源码, `operators/runtime/{device.py,cuda/device.cu}`, 以及 `operators/primitives/cuda`.
修改 `worker/{model,sampling,speculation,draft_chain,inference}.py`, `development/model/capture.py` 和 `tests/test_constraints.py`.

**接口:** 在相应 `api.py` 中定义 `output.hidden`, `output.distribution`, `output.sample`, `mtp.context`, `mtp.draft` 和 `mtp.accept`.
各接口在适用时接收显式算子族权重, 已准备的算子 workspace, 以及具名 input/view 类型.
输出按用途提供 `hidden`, `logits`, `probabilities`, `tokens` 和 `selected_probability` 字段.
Acceptance 字段为 `tokens`, `counts` 和 `candidate_slots`.

- [ ] 新增 `test_stop_truncation_controls_accepted_slot`: 将 4 个候选截断为 1 个 token, 检查有效 cache commit 数量.
- [ ] 检查在第一个返回 token 前触发 stop 时, 不能提交原始 candidate slot.
- [ ] 变更前运行该测试, 显示尚未覆盖的契约.
- [ ] 将 `final_hidden` 移到现有 sample native 模块. 通用 row gather 保留在 primitives.
- [ ] 将 `mtp_accept` 连同 vocabulary, CUB 和 Philox 依赖移到现有 MTP native 模块.
- [ ] 随这些导出一起修改函数列表, loader 调用和 manifest.
- [ ] 将 V2 的 distribution-only 路径移到新 API. 兼容时保留 sample 的融合路径.
- [ ] 词表投影使用较少行时, hidden 导出仍须接受全部输入行.
- [ ] 移除临时组件适配器, 数值 capture 使用具名 logits.
- [ ] 使用 output/MTP microbench 用例, V4 端点, RNG 检查和状态选择检查完成交付流程.

## 任务 6: 应用 vision 契约

**依赖:** 任务 5 和 V4 的媒体端点. **结果:** Vision 自主管理计划和临时存储, 媒体语义保持不变.

**文件:** `operators/vision/{api,plan,workspace}.py`, 其 CUDA 源码, 以及 `worker/media.py`.

**接口:** `run(weights: Weights, workspace: Workspace, inputs: Inputs) -> Outputs` 返回具名 `hidden`.
Plan 给出 input, transformer 和 merge 阶段条件, grid/frame 容量, 以及 native context 所有权.

- [ ] 将构造函数验证和 native-context 准备移到 vision plan/准备路径.
- [ ] 媒体 decode, 请求顺序和 position 构建保留在 worker.
- [ ] 使用固定 vision microbench 用例检查首末层和奇数帧数.
- [ ] 使用 CPU plan 验证检查无效 grid 和 65536 行容量.
- [ ] 使用 V4 媒体端点和代表性媒体计时完成交付流程.

## 各任务的交付流程

- [ ] 运行验证计划中适用的验证命令和全部 hooks.
- [ ] 事实变化时更新 `operators/specification.py` 和双语[服务边界](fusion.zh.md).
- [ ] 根据已验证测量结果更新两份 README 的性能表.
- [ ] 获取另一位 agent 的代码和文档审查.
- [ ] 修正发现的问题.
- [ ] 更新双语 handoff, 不写入测试日志.
- [ ] 仅提交任务文件, 使用 Conventional Commits 标题.
- [ ] 保持 hooks 开启并推送 `main`.

任务 6 后, 移除未使用的兼容文件和旧 kernel 包.
检查推理, 开发工具和测试中的导入及直接 buffer 访问.
后续融合变化作为独立变更评估, 并执行各自的完整验证.
