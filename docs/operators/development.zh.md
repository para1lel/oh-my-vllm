# 算子开发

使用[算子规格](model.zh.md), [fusion 计划](fusion.zh.md)和[评测流程](evaluation.zh.md).
FFN provider 是基础版本. 其余 serving provider 尚不可用.
Worker 仍使用占位后端.

## 工具

使用 CUDA 13.1, Nsight Systems 和 Nsight Compute.
工具路径保存在被忽略的 `LOCAL.md` 中.
将 `CUDA_HOME` 设为 CUDA 安装目录.

安装算子开发包:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

推理绑定包位于 `requirements/kernels.txt`.
开发包不是 worker 依赖.

将 `CUTLASS_ROOT` 设为仓库外的依赖目录.
获取指定版本的 CUTLASS 头文件:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

构建只接受来自干净依赖 checkout 的 Git 修订版本 `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
将 `OH_MY_VLLM_KERNEL_CACHE` 设为仓库外的构建目录.

构建原生模块:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels build
```

推理加载器使用这些模块, 不使用编译器或开发工具.
加载器拒绝与其 manifest 不一致的源码或二进制标识.

## 评测命令

获取负载计划:

```bash
scripts/with-env.sh python -m development.kernels plan
```

执行 CPU 契约检查:

```bash
scripts/with-env.sh python -m unittest development.kernels.test_contracts development.kernels.test_reference
```

将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `ARTIFACT_ROOT` 设为仓库外的产物目录.
测量 FFN provider:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode graph --output "$ARTIFACT_ROOT/ffn-graph.json"
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode eager --output "$ARTIFACT_ROOT/ffn-eager.json"
```

获取单个 shape 的 Systems trace 和 Compute counters:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels profile --model "$MODEL_ROOT" --operator ffn --rows 1 --mode graph --output "$ARTIFACT_ROOT/ffn-profile"
```

执行命令前, 选择尚不存在的 profile 目录路径.
Profiling 不提供验收计时.

使用 `--rows` 选择部分建模行数.
子集不足以满足算子覆盖率门槛.
缺失的 provider 会阻止完整实现的验收.

FFN 检查会确认所有输出元素都是有限值.
所有残差元素都必须满足 BF16 精确相等.

FP64 投影参考比较首行, 中间行和末行的全部列.
报告会注明抽样范围, 不提供其他输出行的容差证据.

原始报告和 profiler 文件保存在仓库外.
不在项目文档中记录测量结果.
