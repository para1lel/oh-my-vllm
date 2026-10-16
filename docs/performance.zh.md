# 端到端要求

使用一张 NVIDIA B200 和[模型契约](operators/model.zh.md).
使用完整 Rust 服务和 Python 推理 worker.
占位输出不能作为推理证据.

## 配置

每个 batch 以并发 HTTP 请求开始.
输入长度适用于每条请求. 每条请求必须生成 8192 个 tokens.

| 模式 | Batch 大小 | 每条请求的输入 tokens |
|---|---|---|
| 普通 decode | 1, 2, 4 | 32768 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 |

共 12 种配置. 每种配置必须单独通过.
此比较使用 32768 的 token 预算和最多 4 条运行中请求.
使用没有缓存前缀的请求.
计时前加载模型并完成初始化.

从 HTTP 提交开始测量, 直到客户端收到该 batch 的全部输出.
包含请求准备, 排队, prefill, decode, 约束, 采样, 进程通信和输出传输.
不能用首个 token 的时间代替完整延迟.

## 性能限制

实测 batch 延迟不得超过其理论 roofline 下界的 `5 / 3` 倍.
使用 [B200 峰值和精度规则](operators/evaluation.zh.md#性能门槛).
计算完整模型的必要工作量, 包括 MTP context, proposals, 验证和接受.
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

构建完整 worker 期间停止算子优化.
[算子负载](operators/evaluation.zh.md)保留供后续算子分析使用.

## 测量命令

将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `ARTIFACT_ROOT` 设为仓库外的新目录.
按[服务配置](serving.zh.md#启动)构建原生模块和 release 可执行文件.
执行完整比较:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

命令为每种模式启动并停止服务.
它在计时前完成短初始化请求.
它通过 HTTP usage 字段检查 prompt 长度, 输出长度及未缓存的前缀.
默认 cache 容量为 1024 个 FA block 和 1024 个 GDN slot.
使用 `--blocks` 和 `--mamba-blocks` 修改这些容量.

使用 `--case ordinary:1:32768` 选择一个配置.
使用 `--output-tokens 32` 进行短运行.
短运行或不完整的配置集合不能通过完整门槛.

设置 `OH_MY_VLLM_EXECUTION_TRACE` 后, Worker 写入执行轨迹.
轨迹路径必须位于仓库外, 且运行前不得已存在.
Benchmark 为其服务进程设置此变量.

## 下界计算

[计算实现](../development/serving/roofline.py)使用各矩阵运算的规定精度.
它包含完整的 causal attention 长度, FP32 GDN 工作量, 词表投影以及 MTP context 和 draft 工作量.
它包含接受算法的运算及实测接受的 token 数量.

Prefill 分块按一次完整输入计算计数.
计算会在每个验证轮次中将相关请求的 shape 放在一起.
计算排除中间张量复制和 padding.
没有指定吞吐模型的小运算不会增加下界.

门槛使用总 HBM 时间, tensor-core 时间, FP32 时间和 FP64 时间中的最大值.
Tensor-core 时间按各自峰值累加 FP8, BF16 和 TF32 工作量.
这个全局下界允许不同资源之间重叠.
独立区域的估计用于分析, 不用于设置门槛.

仓库外的产物包含响应, usage, 执行 shape, MTP 计数, 下界分项及源码标识.
测量期间源码改变会阻止验收.
轨迹必须完整覆盖 prefill, 已接受位置必须连续, 输出数量必须与 HTTP usage 一致.
同一工作计算多次或已接受的位置减少会阻止验收.
