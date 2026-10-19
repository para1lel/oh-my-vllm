# Kernel 开发

使用[模型契约](model.zh.md), [数值检查](evaluation.zh.md)及 [tile 迁移计划](../tile-migration.zh.md).
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

执行 CPU 检查:

```bash
scripts/with-env.sh python -m unittest development.reference.test_contracts development.reference.test_reference development.serving.test_roofline
```

将 `MODEL_ROOT` 设为 checkpoint 目录.
运行小规模 GPU 参考:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.reference --model "$MODEL_ROOT"
```

使用 `--operator` 选择 `ffn`, `fa_attention`, `gdn_attention`, `vision_block`, `sample`, `mtp_context` 或 `mtp_draft`.
命令检查 eager 执行和 CUDA Graph replay.
它在每次检查前恢复可变状态, 并在使用后关闭原生 workspace.
它没有计时或效率门槛.

每个 provider 声明其数值抽样范围.
FFN 检查全部 residual 元素并抽样 output 行.
GDN 检查 private state 写入. FA 和 MTP 检查 KV 写入及未改变的 guard 存储.
Sampling 检查带种子的选择及接受状态 slot.
Vision 检查 raw patch, output 样本及 frame isolation.

这些检查不能证明其声明范围之外的元素精度.

性能验收使用 [HTTP benchmark](../performance.zh.md#测量命令).
当 trace 或 counter 可以识别瓶颈时使用 Nsight Systems 和 Nsight Compute.
其产物保存在仓库外.
