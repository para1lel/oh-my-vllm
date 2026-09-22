# 自定义 kernel 开发

生产自定义 kernel 使用 TileLang 0.1.14。独立库内部仍可使用 Triton。TileFoundry 是 agent 开发工具，不是服务依赖。已有正确性测试和冻结 vLLM 验收门槛继续作为标准，不设置与项目旧 Triton kernel 的对比门槛。

## 复现工具环境

在项目根目录、现有 oh-my-vllm 环境中运行：

```bash
git submodule update --init --recursive 3rdparty/TileFoundry
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python setuptools-scm==10.2.3 vcs-versioning==2.4.1
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python --no-build-isolation -r requirements/development.txt
uv pip check --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python
scripts/with-env.sh tilefoundry --help
```

先安装构建元数据工具：即便锁定文件里也有这些依赖，`--no-build-isolation` 仍要求它们在生成 editable 元数据前已安装。

gitlink 固定我们的 para1lel/TileFoundry fork，基于上游 main b5aaa442161940694e710c0d00c32b5c884faae2。其 oh-my-vllm-integration 分支是项目仅 main 策略的唯一例外，不向上游提 PR。本地适配移除 Transformers 版本上限（保留下限），并为 RoPE 添加 HIR sin/cos 求值、打印和成本分类；后者不提供 TileFoundry 设备代码生成。

保持 fork 小巧：简单不兼容问题本地修复，重大变化先讨论。不维护 TileLang 或 OR-Tools fork，改用兼容的已发布版本。

运行时使用 apache-tvm-ffi 0.1.12、protobuf 6.33.6，以及已有 Torch2.14、Transformers5.17、FlashInfer0.6.18.post1。requirements/runtime.txt 固定运行依赖；requirements/development.txt 添加 editable fork 及工具。editable 安装要求源码目录持续可用。

## Agent 工作流

1. 选择算子前阅读生产包装器和已有测试。保留支持的 shape、packed stride、dtype、返回状态、别名规则和数值舍入边界。
2. 在 development/kernels/semantics.py 表达完整逻辑行为。development/kernels/twins.py 通过薄适配层调用生产代码。快照和更新后的逻辑缓存是输出，不隐藏为副作用。
3. 用 TileFoundry 分析算术量、显存和 roofline 假设。提供的单个逻辑 CTA 是分析拓扑，不是实际 TileLang 放置方式。HIR 成本估计不是实测 GPU 时延。
4. 实现并调优 TileLang kernel。保留公开 Python 接口、调度器所有权、持久 784-token 页及 MTP 排除位置零的契约。可改变内部 tile、融合及临时布局。长度不决定算法/布局时采用动态 token 维度。
5. 临时算子实验放在仓库外，随后运行已有必需测试和实际模型验收。及时停止任务自有 GPU 程序。不提交微基准脚本、生成 kernel、报告、缓存内容或实验记录。
6. 每个里程碑审查并修复问题，然后在 main 提交。主仓库更新 submodule gitlink 前先推送 fork 改动。

TileFoundry 分析缩小搜索空间后，使用 TileLang 开发期 `AutoTuner`。显式提供合法页表、长度、packed stride 和递归状态；随机整数元数据不是有效输入生成器。对图回放路径显式选择 `cudagraph` 计时后端。保持原参考容差，不允许以一定比例元素失配作为豁免。有状态 kernel 每轮正确性试验都重置状态，计时不能测量不断变化的递归。搜索代表性 batch、上下文范围及普通/MTP shape，包括边界和 ragged 情况。考虑 tile 大小、线程数、split 数、布局和 pipeline；参数 autotune 不能代替更好的内存访问或融合设计。

入选候选必须按完整算子复测：包含 split reduction、临时分配和 launch 开销，并考虑重复输入的缓存效应。随后验证实际模型正确性和完整性能矩阵。生产中使用确定、已验证的 shape 分派，不在服务启动或推理期间搜索。临时调优脚本、缓存产物和算子级结果留在仓库外。参见 [TileLang autotuning 指南](https://www.tilelang.com/programming_guides/autotuning.html)。

示例（分析输出明确放在 /tmp）：

```bash
scripts/with-env.sh tilefoundry analyze development/kernels/semantics.py:Operators.silu /tmp/oh-my-vllm-silu-analysis.txt --compute-cost --memory --roofline
PYTHONPATH="$PWD:$PWD/python" scripts/with-gpu.sh scripts/with-env.sh tilefoundry check development/kernels/twins.py:TileLang.silu --inputs random --out output --fn allclose --atol .03 --rtol .03
scripts/with-gpu.sh scripts/with-env.sh pytest -q tests
```

检查 tuple 输出时，为每个返回 tensor 选择明确判定条件。代表性 HIR shape 有助于研究语义，但不能替代已有 FP64 参考、不规则元数据、packed 视图、图回放、页隔离、最大上下文或实际模型检查。不能为适应 TileLang 或旧 Triton 实现而放宽容差。

## 验收与性能

保留文档规定的全部服务、agentic、最大上下文和正确性检查。使用冻结 EngineCore 的 12 组对比：吞吐至少 95%、TTFT 至多 1.1 倍，并满足既有稳定性和来源审计。捕获/测量前预热所有待测 shape。逐运行的 TileLang 和第三方 Triton 缓存根目录防止并发编译污染审计；FlashInfer 也必须完整预热，正式重复期间保持不变。

任一组失败时，分析实际模型、定位主导开销、跨所需 shape 做针对性优化后重测。用 TileFoundry 成本/显存分析指导实验，再在实际模型运行中验证收益。完成前全部 12 组必须通过吞吐、TTFT 和稳定性门槛；性能或正确性失败都不能仅凭解释豁免。未来模型/后端可添加语义模块和 TileLang 特化，无需改变 Rust 调度；多 GPU 和本地 DSpark 仍属未来架构工作。
