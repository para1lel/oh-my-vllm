# 算子验证实施计划

> Agent 指令: 选择执行方式后, 使用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`. 按顺序完成复选框步骤.

**目标:** 为 kernel 变更提供固定 PyTorch 参考 fixture 和理论延迟验收门槛.

**架构:** 开发适配器调用完整生产操作. 数值测试加载仓库 fixture. 性能测试将延迟倍率与冻结的逐用例门槛比较.

**技术栈:** Python, NumPy, 用于显式生成 fixture 的 PyTorch FP64, TVM FFI, CUDA events, 以及单张 NVIDIA B200.

**规格:** [Microbench 设计](microbench.zh.md)和[算子结构设计](design.zh.md).

## 全局约束

- 推理代码不得导入开发依赖包.
- 遵守[模型契约](model.zh.md), [GPU 所有权规则](../execution.zh.md)和[贡献流程](../../CONTRIBUTING.zh.md).
- 保持冻结的短输入全模型参考及其 SHA256 不变.
- 数值 fixture 存入仓库. 计时样本, 测试报告和主机设置保存在仓库外.
- 实现每个新行为测试前说明原因. 使用少量参数化测试.
- 每个边界最多使用 5 个数值用例和 5 个性能用例.
- 缺失 fixture, 经批准的门槛或可比较的主要测量结果时, 不得通过验收.

## 审查重点

| 条件 | 必要行为 | 任务 |
|---|---|---|
| 预期数组缺失, 改变或被重新生成 | 常规测试直接失败, 不运行 PyTorch | V1 |
| 输出看似正确但状态写入错误 | 检查指定写入及未修改的 guard | V2 |
| 采样与递推状态 | 符合固定参考及精确 counter/slot 契约 | V2 |
| 热缓存权重或膨胀的工作量计费 | 理论分母保持保守且固定 | V3 |
| 局部 kernel 正确但组合错误 | 保留全模型和 HTTP 验收 | V4 |

## 固定 shape 选择

以下元组是完整用例选择, 不是用于构造笛卡尔积的取值集合.
`B` 为请求数, `T` 为总 query 行数, `Q` 为每个请求的 query 数, `L` 为每个请求的上下文长度.
宽度使用模型契约中的 checkpoint 维度.

| 文本用例 | 数值 `(B,T,Q,L)` | 性能 `(B,T,Q,L)` |
|---|---|---|
| 单请求 decode | `(1,1,1,784)` | `(1,1,1,32768)` |
| 批量 decode | `(4,4,1,784)` | `(4,4,1,131072)` |
| MTP verification | `(4,20,5,784)` | `(4,20,5,131072)` |
| Checkpoint prefill | `(1,784,784,784)` | `(1,784,784,784)` |
| 大 prefill | `(1,1568,1568,1568)` | `(4,32768,8192,32768)` |

FA 和 GDN 使用这 5 个用例.
FFN 和 `output.hidden` 使用其中的 `T` 值, 不附带无用的 attention 元数据.
各 fixture 使用一致的起始 position 和 page/state slot.
GDN 用例检查全部指定 checkpoint, 并包含恢复状态后的 decode 用例.

| 其他边界 | 固定数值用例 | 固定性能用例 |
|---|---|---|
| `output.distribution`, `output.sample` | 行数 1 greedy, 1 stochastic, 4 stochastic, 5 stochastic, 20 stochastic | 相同的 5 个用例 |
| `mtp.context` | `(T,B)=(1,1),(4,1),(8,2),(16,4),(784,1)` | 前 4 个用例加 `(32768,4)` |
| 单次 `mtp.draft` | Batch 1 query-only, 1 full, 2 full, 4 query-only, 4 full | 相同的 5 个用例 |
| 四步 draft 链 | Batch 1, 2, 4 | 相同的 3 个用例 |
| `mtp.accept` | Batch 1 全接受, 1 首次拒绝, 2 部分接受, 4 混合拒绝, 4 含未激活后缀 | 相同的 5 个用例 |
| 各 vision 边界 | Grid/frame 组合 `(16x16,1),(28x28,1),(14x14,3)` | 相同的 3 个用例 |

存在历史上下文时, MTP context 数值用例使用长度 784, 性能用例使用长度 32768.
784 行 MTP prefill 从位置 1 开始, 结束时的 context length 为 785.
Draft 用例分别使用相同的上下文长度.
层索引从 0 开始.
数值用例中, FA 使用 checkpoint 层索引 3, GDN/FFN 使用层索引 0, vision 输入/内部/merge 使用层索引 0/1/26.
Manifest 记录完整权重张量名.

每个用例使用一个固定的数据/mask/penalty 配置.
采样 fixture 通过这些配置覆盖有效 mask, penalty, 截断和 bonus 行为.
不要在数值集合上增加 seed 或层扫描.

每个 FA/GDN/FFN 性能用例按模型顺序, 对适用的 target 层执行一次固定遍历.
内部 vision 性能用例按顺序使用层索引 1 到 25.
其他性能边界使用各自唯一的生产权重绑定.
用例 manifest 绑定有序权重/状态身份和操作次数.
计时针对完整遍历, 资源工作量也按完整遍历计算.
不得仅将倍率的一侧除以操作次数.

## V1: 建立 fixture 和适配器框架

**依赖:** 无. 源码迁移前, 从当前运行时开始.

**新建:** `development/operators/{__init__,__main__,cases,run,fixtures,test_contracts}.py`, `cases.json`, 以及 `adapters/{fa,gdn,ffn,output,mtp,vision}.py`.
修改 pre-commit 的 CPU 测试命令.
适配器最初将现有 buffer 接口转换为具名观测结果.
各公开算子迁移后删除相应转换.

**接口:** `Case` 记录 ID, 边界, shape, 模式, 权重名, fixture 路径和预期分派条件.
`Fixture` 包含具名输入, 初始状态, 预期输出, 精确副作用和已验证的 manifest.
`run.py` 提供:

```python
prepare(case: Case, device: Device, fixture: Fixture, *, model: Path) -> PreparedCase
PreparedCase.reset() -> None
PreparedCase.invoke() -> None
PreparedCase.observe() -> dict[str, numpy.ndarray]
PreparedCase.close() -> None
```

- [x] 在 `test_contracts.py` 中新增 `test_fixture_loading_is_strict_and_reference_free`.
- [x] 检查分片缺失, 摘要改变, 字段不完整, 以及常规测试中被禁止的 PyTorch 导入.
- [x] 新增 `test_case_limits_reject_shape_sweeps`, 检查用例超过 5 个或出现未列出的 shape.
- [x] 实现前先运行这些检查.
- [x] 实现用例, 固定地址 fixture, 具名观测结果和清理逻辑.
- [x] 将 CPU 检查加入现有 pre-commit 命令.
- [x] 请另一位 agent 检查框架.
- [x] 提交框架.

## V2: 生成并提交数值 fixture

**依赖:** V1. 接口迁移前完成.

**新建:** `development/operators/reference/{__init__,__main__,fa,gdn,ffn,output,mtp,vision}.py`, `numerics.py`, `limits.json`, 以及 fixture 目录.
使用现有 reference 依赖锁文件.
修改 `test_contracts.py`.

修改 `python/oh_my_vllm/operators/output/{api,workspace}.py` 和 `python/oh_my_vllm/operators/output/cuda/sample.cu`, 完成必要的准备步骤.
提供 `output.distribution(weights, workspace, inputs, settings)`, 用于词表投影和分布准备.
它返回 `Outputs`, 已准备的概率张量位于 `Outputs.probabilities`.
它不选择 token, 也不接受 RNG 参数.
增加 native `sample_distribution` 导出, 并更新函数列表和 manifest.
保留现有融合采样路径.

**接口:** 各算子族提供 `reference(case, inputs, state, weights) -> dict[str, numpy.ndarray]`, 直接使用 PyTorch 公式.
`numerics.compare(actual, fixture, limits) -> NumericalResult` 返回字段指标和精确契约失败项.
生成器写入新的暂存目录, 不覆盖已跟踪 fixture.

- [x] 新增 `test_numerics_rejects_wrong_state_and_missing_limits`, 包含错误 checkpoint, guard 写入和缺失预算.
- [x] 使用零权重投影, 单可见 key attention, 以及可解析的 recurrence/acceptance 示例检查参考公式.
- [x] 实现前先运行这些检查.
- [x] 在该边界数值验收或性能校准前, 实现 distribution-only 路径.
- [x] 实现直接 PyTorch 参考和完整字段比较.
- [x] 在 reference 环境中生成固定输入和预期数组.
- [x] 请另一位 agent 检查公式, 数值预算, checkpoint 身份和完整 fixture manifest.
- [x] 将经过审查的 NumPy 分片, manifest, 源码和数值门槛一起提交.
- [x] 在不含 PyTorch 的推理环境中运行数值验收.

## V3: 实现理论延迟验收

**依赖:** V1 和 V2. 在源码迁移到 `main` 前冻结经审查的门槛.
迁移后运行最终性能验收. 按[最终交付流程](plan.zh.md#最终交付流程)统一提交并推送一次.

**新建:** `development/operators/{accounting,timing}.py`.
修改 CLI, 门槛和 `test_contracts.py`.
使用 `development/serving/accounting.py` 的速率值和公式辅助函数, 不改变 HTTP 计费行为.

**接口:** `accounting.work(case) -> ResourceWork` 给出语义资源总量.
`accounting.lower_bound(case, device_spec) -> float` 返回秒数.
`timing.measure(prepared, *, count=MINIMUM_SAMPLES, events_factory=Events, repeats=None) -> list[Sample]` 记录原始 GPU/host 样本.
`timing.check(samples, lower_bound, max_ratio) -> PerformanceResult` 给出倍率, 区间和判定.

- [x] 新增 `test_accounting_ignores_implementation_work`, 检查额外 scratch 和重复 launch 不能增加理论计费.
- [x] 新增 `test_ratio_gate_excludes_reset_and_fails_uncertainty`, 使用假 events 和固定计时样本.
- [x] 检查 cache 常驻与流式访问计费, reset 排除, 固定门槛失败和未决区间.
- [x] 实现前先运行这些检查.
- [x] 按 microbench 设计实现计费和测量.
- [x] 使用固定的 5 次会话流程, 根据当前正确实现校准建议门槛.
- [x] 请另一位 agent 检查资源数量, 缓存策略, 实测倍率和建议门槛.
- [x] 将经批准的门槛和来源摘要与测量工具一起准备好, 用于最终交付.
- [x] 不提供旧构建或旧计时输入, 运行性能验收.

## V4: 保留集成验收

**依赖:** V2. 受影响的算子迁移前完成相应端点检查.

**修改:** `development/model/{capture,compare,__main__,test_identity,test_logits}.py` 和现有 worker 行为测试.
新建 `development/model/{cases,reference_cases}.py`, 并为新增端点建立仓库 fixture.
为模型 CLI 增加 `--case`. 默认值保持现有短输入用例行为.
保持 `reference.py`, `reference.npz` 和现有短用例身份检查不变.

- [x] 数值 capture 使用具名 logits 和全部 7 个 native 模块身份.
- [x] 在相应执行路径变化前, 新增 `mtp-restored`, `image-video` 和 `context-32769` 全模型用例.
- [x] 分别比较 target 加 MTP 计算, media 到 language 计算和长上下文 decode 后的完整词表端点.
- [x] 新增全模型参考同样遵守仓库 fixture 和显式生成规则.
- [x] 保持现有短输入用例门槛不变.
- [x] 保留执行生命周期, stop 截断, constraints, service 和 Rust 检查.
- [x] 使用不变的工作负载设置运行 24 组 HTTP 配置.
- [x] 根据已验证 HTTP 产物更新两份 README 的性能表.
- [x] 请另一位 agent 检查集成变更.
- [x] 将集成变更纳入最终交付.

## 命令与验收

使用以下算子命令进行验证.
使用前设置本地主机的模型/环境路径及新的仓库外报告路径.

```bash
scripts/with-env.sh python -m unittest development.operators.test_contracts
OH_MY_VLLM_CONDA_PREFIX="$REFERENCE_ENV" scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.reference --model "$MODEL_ROOT" --output "$FIXTURE_STAGING"
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators numerics --model "$MODEL_ROOT" --report "$NUMERICAL_REPORT"
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators calibrate --model "$MODEL_ROOT" --report "$CALIBRATION_REPORT"
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators performance --model "$MODEL_ROOT" --report "$PERFORMANCE_REPORT"
scripts/with-env.sh pre-commit run --all-files
```

只有显式 fixture 生成使用 reference 环境.
校准在仓库外报告中提出门槛建议, 不能改变已跟踪门槛.
常规命令加载仓库用例定义, fixture 和经批准的门槛, 不允许覆盖 shape.
数值失败, 倍率失败, 结果缺失及未决结果均以非零状态退出.

使用现有[短输入 logits 命令](development.zh.md#数值命令)和 [HTTP 命令](../performance.zh.md#测量命令).
Device 内存或同步改变时, 在计时之外运行 Compute Sanitizer.
每次运行后停止所有进程并释放 Graph owner, 失败时也一样.
