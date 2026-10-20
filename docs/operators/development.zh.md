# Kernel 开发

使用[模型契约](model.zh.md), [数值检查](evaluation.zh.md)及 [GPU 执行规则](../execution.zh.md).
推理 worker 加载原生模块. 占位实现仅适用于 `--model stub`.

## 工具

使用 CUDA 13.1. 主机路径保存在忽略的 `LOCAL.md` 中.
将 `CUDA_HOME` 设为 CUDA 安装目录.
安装 kernel 开发包:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

推理绑定位于 `requirements/kernels.txt`.
开发包不是 worker 依赖.
将 `CUTLASS_ROOT` 设为外部依赖目录.
获取指定头文件:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

构建仅接受干净的 Git revision `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
将 `OH_MY_VLLM_KERNEL_CACHE` 设为外部构建目录.
构建原生模块:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build
```

推理 loader 不导入开发工具, 也不使用编译器.
它根据构建 manifest 检查源码和二进制 identity.
每个二进制在外部 cache 中具有不可变副本和 manifest.

## 数值命令

运行 CPU 证据和计费检查:

```bash
scripts/with-env.sh python -m unittest development.model.test_identity development.model.test_logits development.serving.test_roofline
```

对照使用仓库内冻结的 [FP64 reference](../../development/model/reference.npz).
将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `LOGITS_REPORT` 设为仓库外的新 JSON 路径.
运行全模型对照:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --report "$LOGITS_REPORT"
```

命令还将推理 logits 保存到同名的 `.npz` 文件.
使用 `--mode prefill` 对照短提示的末行.
[评估流程](evaluation.zh.md)规定输入, 精度, 指标及范围.

只有明确变更参考时, 才需要生成新文件.
在独立环境中安装 `requirements/model-reference.lock`.
使用[官方 PyTorch 包](https://pytorch.org/get-started/previous-versions/).
推理 worker 不使用 PyTorch.
将 `LOGITS_REFERENCE` 设为仓库外的新 `.npz` 路径.
用参考环境生成文件:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference --model "$MODEL_ROOT" --output "$LOGITS_REFERENCE"
```

生成器不覆盖文件.
只有 checkpoint, 输入, 生成器和数值策略的审查全部完成后, 才更新冻结文件及其 SHA256.

## 算子命令

将 `OPERATOR_REPORT` 设为仓库外的新目录.
在推理环境中运行全部固定数值用例:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators numerics --model "$MODEL_ROOT" --report "$OPERATOR_REPORT"
```

该命令加载仓库内 fixture 和批准的门槛, 不使用 PyTorch.
fixture 缺失, identity 改变或未通过门槛都会导致错误退出.

明确更新 fixture 时, 将 `OPERATOR_FIXTURES` 设为仓库外的新暂存目录.
在参考环境中生成固定数组:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.reference --model "$MODEL_ROOT" --output "$OPERATOR_FIXTURES"
```

将 `OPERATOR_LIMITS` 设为仓库外的新 JSON 路径.
按已批准的[精度策略](precision.zh.md)提议门槛:

```bash
scripts/with-env.sh python -m development.operators.budgets --fixtures "$OPERATOR_FIXTURES" --output "$OPERATOR_LIMITS"
```

更新仓库内 fixture, 用例哈希和[门槛](../../development/operators/limits.json)前, 检查数组, 清单, 生成器 identity 和提议门槛.

将 `OPERATOR_PERFORMANCE` 设为仓库外的新目录.
按已批准的倍率门槛运行全部固定性能用例:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators performance --model "$MODEL_ROOT" --report "$OPERATOR_PERFORMANCE"
```

[资源与计时策略](performance.zh.md)定义测量的操作和理论下界.
失败或无法判定的用例会导致错误退出.

初始校准时, 将 `OPERATOR_CALIBRATION` 设为仓库外的新目录.
采集五个完整 session:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators calibrate --model "$MODEL_ROOT" --report "$OPERATOR_CALIBRATION"
```

此命令在外部报告中提议门槛.
它不修改仓库门槛.
将批准的性能条目复制到门槛文件前, 审查完整报告及其来源记录.

完整推理计时使用 [HTTP benchmark](../performance.zh.md#测量命令).
比较变更前后的测量时, 使用相同的 `--prompt-seed` 值.
HTTP 产物记录此 seed 和执行策略.
