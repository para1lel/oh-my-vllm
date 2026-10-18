# 端到端要求

使用一张 NVIDIA B200 和[模型契约](operators/model.zh.md).
使用完整 Rust 服务和 Python 推理 worker.
占位输出不能作为推理证据.

## 配置

每个 batch 以并发 HTTP 请求开始.
输入和输出长度适用于每条请求.

| 模式 | Batch 大小 | 每条请求的输入 tokens | 每条请求的输出 tokens |
|---|---|---|---|
| 普通 decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

共 24 种配置. 每种配置必须单独通过.
64-token 输出使测量对 prefill 时间敏感.
保留 8192-token 输出, 用于长 decode 测量.

此比较使用 32768 的 token 预算和最多 4 条运行中请求.
使用没有缓存前缀的请求.
计时前加载模型.
计时前完成初始化.

在客户端创建请求线程之前开始计时.
在客户端读取并解析全部 JSON 响应且请求线程停止之后结束计时.
包含客户端提交, 服务端请求准备, 排队, prefill, decode, 约束, 采样, 进程通信和输出传输.
计时前构造精确长度的 prompt 文本.
计时不包含模型加载, 初始化和 trace 分析.
不能用首个 token 的时间代替完整延迟.

## 性能限制

每种配置的 roofline efficiency 必须至少为 50%.
实测 batch 延迟不得超过其记录的执行 trace 对应 roofline 时间的 2 倍.
使用选定 GPU UUID 的[实测峰值](calibration.zh.md)及[精度规则](operators/evaluation.zh.md#性能计算).

计算每个 serving 算子的必要工作量, 包括 MTP context, proposals, 验证和接受.
注意力计算使用完整序列长度.
保留实测 MTP 接受数量. 不假定固定接受率.

不能将 padding, 重复计算或不必要复制计为额外必要工作.
不能给下界加入实测启动时间或经验延迟系数.
将源码, 二进制, checkpoint, 设备, 配置和计时标识保存在外部产物中.
保留下界计算, 供审查使用.

## 执行

优先选择 decode 请求. 在 prefill 请求之间分配剩余 token 预算.
仅含 decode 请求的 batch 使用 CUDA Graph 执行.
包含 prefill 请求的 batch 使用普通执行.

在 target 和 proposal 计算中, 先应用约束 mask, 再采样.
投机拒绝后, 约束状态必须与已接受 token 历史一致.
使用模型契约中的 MTP4 接受和私有 cache 状态选择.

在能够缩短完整执行时间时, 对 producer 与 consumer 的工作进行融合.
可以对独立工作使用多个 CUDA streams, 但必须明确依赖关系.
推理 worker 不使用 PDL.

使用算子测量和完整 HTTP 测量来选择改动.
使用[算子负载](operators/evaluation.zh.md)进行算子验收.
算子测量不设最低效率要求.

## 测量命令

将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `PEAK_PROFILE` 设为选定 GPU UUID 的校准文件.
通过 `OH_MY_VLLM_GPU_UUID` 选择此 UUID.
将 `ARTIFACT_ROOT` 设为仓库外的新目录.
按[服务配置](serving.zh.md#启动)构建原生模块和 release 可执行文件.
执行完整比较:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --peaks "$PEAK_PROFILE" --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

命令为每种模式启动并停止服务.
它在计时前完成短初始化请求.
它通过 HTTP usage 字段检查 prompt 长度, 输出长度及未缓存的前缀.
默认 cache 容量为 1024 个 FA block 和 1024 个 GDN slot.
使用 `--blocks` 和 `--mamba-blocks` 修改这些容量.

使用 `--case ordinary:1:32768` 选择其两个输出长度.
使用 `--case ordinary:1:32768:64` 选择一个配置.
使用 `--output-tokens 64` 进行对 prefill 时间敏感的测量.
使用 `--output-tokens 32` 进行短运行.

不完整的配置集合或其他输出长度不能通过完整门槛.
配置重复, 配置缺失或任一配置失败都会阻止完整验收.
命令拒绝重复选择, 也拒绝与所选输出长度不匹配的 case.

设置 `OH_MY_VLLM_EXECUTION_TRACE` 后, Worker 写入执行轨迹.
轨迹路径必须位于仓库外, 且运行前不得已存在.
Benchmark 为其服务进程设置此变量.

## 下界计算

[计算实现](../development/serving/roofline.py)使用各矩阵运算的规定精度.
它包含完整的 causal attention 长度, FP32 GDN 工作量, 词表投影以及 MTP context 和 draft 工作量.
它包含接受算法的运算及实测接受的 token 数量.

每个 trace 事件保留其记录的 batch, prefill 分块, query 长度及 proposal 阶段.
计算保留独立调用. 计算不按验证轮次编号合并请求.
每个 prefill 分块都包含该次调用所需的权重和 cache 访问.
采样按已完成 prefill 的请求及记录的验证行计算.

计算排除中间张量复制和 padding.
没有指定吞吐模型的小运算不会增加下界.

每个 serving 算子的时间取其 HBM 时间, tensor-core 时间, FP32 时间和 FP64 时间中的最大值.
Tensor-core 时间按各自峰值累加 FP8, BF16 和 TF32 工作量.
当前 worker 在一个 CUDA stream 上顺序执行这些算子.
门槛按记录的调用次数累加这些算子的时间.
有依赖关系的调用之间不允许资源重叠.

全局资源估计用于分析, 不用于设置门槛.

仓库外的产物包含响应, usage, 执行 shape, MTP 计数, 下界分项及源码标识.
[产物检查](../development/serving/benchmark.py#L210)<!-- source: def verify_artifact_identity(arguments, expected): -->在源码, 可执行文件, 原生构建或 checkpoint 标识改变时阻止验收.

轨迹必须完整覆盖 prefill, 已接受位置必须连续, 输出数量必须与 HTTP usage 一致.
其唯一的 identity 记录必须匹配服务模式. 其 prompt 长度必须与 HTTP usage 一致.
每个 HTTP 响应必须标识被测 batch 中不同的请求.
同一工作计算多次或已接受的位置减少会阻止验收.
