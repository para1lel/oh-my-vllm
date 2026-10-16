# 算子开发

使用[算子规格](model.zh.md), [fusion 计划](fusion.zh.md)和[评测流程](evaluation.zh.md).
原生 provider 提供全部七个 serving 边界. 完整负载验收是必要的.
推理 worker 加载原生模块. 占位实现仅用于 `--model stub`.

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
每个二进制文件在外部 cache 中都有不可变副本及构建 manifest.

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

使用 `--operator` 选择 `ffn`, `fa_attention`, `gdn_attention`, `vision_block`, `sample`, `mtp_context` 或 `mtp_draft`.
使用 `--rows` 选择部分建模行数.
子集不足以满足算子覆盖率门槛.
负载不完整或 provider 失败会阻止验收.

为每个算子和执行模式提供一份来自同一组冻结源码的完整报告.
使用报告路径执行完整验收审计:

```bash
scripts/with-env.sh python -m development.kernels audit --reports "$ARTIFACT_ROOT"/suite-*.json --output "$ARTIFACT_ROOT/acceptance.json"
```

审计检查当前源码, 不可变二进制, checkpoint 身份, 冷权重, 数值检查和全部建模 shape.
源码变更后, 旧报告不再符合验收条件.

FFN 检查会确认所有输出元素都是有限值.
所有残差元素都必须满足 BF16 精确相等.

FFN 的 FP64 投影参考比较首行, 中间行和末行的全部列.
每个 provider 在报告中注明其数值抽样范围.

GDN 检查全部输入投影及私有状态写入. Sampling 检查所有行和词表列.
FA 和 MTP 检查 KV 写入及未改动的保护存储. Vision 检查原始 patches, 输出抽样和帧隔离.
这些检查不为所述范围以外的元素提供容差证据.

原始报告和 profiler 文件保存在仓库外.
不在项目文档中记录测量结果.
