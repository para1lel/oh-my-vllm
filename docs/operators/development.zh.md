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

在独立开发环境中安装 `requirements/model-reference.lock`, 用于 PyTorch FP64 参考.
使用[官方 PyTorch 包](https://pytorch.org/get-started/previous-versions/).
推理 worker 不使用 PyTorch.
将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `LOGITS_REFERENCE` 设为仓库外的 `.npz` 路径.
用该环境生成独立参考:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference --model "$MODEL_ROOT" --output "$LOGITS_REFERENCE"
```

命令拒绝已有输出路径.
数值验收前将其 SHA256 记录到对照策略.
将 `LOGITS_REPORT` 设为仓库外的新 JSON 路径.
对照还将实际 logits 保存到同名的 `.npz` 文件.
用推理开发环境运行完整模型对照:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --reference "$LOGITS_REFERENCE" --report "$LOGITS_REPORT"
```

使用 `--mode prefill` 对照短提示的末行.
[评估流程](evaluation.zh.md)规定输入, 精度, 指标及范围.
命令没有计时或效率门槛.
性能验收使用 [HTTP benchmark](../performance.zh.md#测量命令).
当 trace 或 counters 能定位瓶颈时, 使用 Nsight Systems 和 Nsight Compute.
其 artifacts 保存在仓库外.

比较变更前后的 HTTP 测量时, 使用相同的 `--prompt-seed` 值.
HTTP 产物记录此 seed 和执行策略.
