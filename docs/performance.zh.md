# HTTP 性能测量

使用一张 NVIDIA B200, 并遵守[模型契约](operators/model.zh.md).
使用完整 Rust 服务和 Python 推理 worker.
占位输出不能作为推理证据.
[GPU 执行规则](execution.zh.md)规定执行方式.

## 配置

每个 batch 同时启动 HTTP 请求.
输入和输出长度适用于每个请求.

| 模式 | Batch 大小 | 每请求输入 token | 每请求输出 token |
|---|---|---|---|
| 普通 decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

共 24 种配置.
64-token 输出使测量对 prefill 时间敏感.
保留 8192-token 输出用于长 decode 测量.
使用 32768-token 预算, 最多同时运行 4 个请求.
使用没有缓存前缀的请求.

使用 workload `serving-tools-v1` 及 checkpoint 和服务的默认采样参数.
当前默认值为 temperature 1.0, top-k 20, top-p 0.95, presence penalty 0.5, 并开启重复循环保护.
使用客户端序号作为随机 seed. 固定 seed 不会选择 greedy 采样.

关闭 reasoning.
设置 `tool_choice=auto`, 并使用 [benchmark 工具 schema](../development/serving/workload.py).
允许普通文本. 语法检查工具名称和 JSON 参数.
Target 和 proposal 采样使用与 serving 相同的语法及惩罚规则.

计时前加载模型.
计时前完成初始化.
计时前构造 prompt 文本.
客户端创建请求线程前开始计时.
客户端将全部响应读取为 JSON 且请求线程停止后结束计时.
包含提交, 请求准备, 排队, prefill, decode, 约束, 采样, 进程通信和输出传输.

不包含模型加载, 初始化和 trace 分析.
不能用首 token 时间代替完整延迟.

## 测量命令

将 `MODEL_ROOT` 设为 checkpoint 目录.
通过 `OH_MY_VLLM_GPU_UUID` 选择一个 GPU UUID.
将 `ARTIFACT_ROOT` 设为仓库外的新目录.
按[服务设置](serving.zh.md#启动)构建原生模块和 release 可执行文件.
运行完整比较:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

命令为每个模式启动并停止服务.
计时前完成短初始化请求.
它还在计时前为每个模式检查一次完整工具调用.
它检查 HTTP usage 中的 prompt 长度, 输出长度和无缓存前缀.
默认缓存容量为 1024 个 FA block 和 128 个 GDN slot.
使用 `--blocks` 和 `--mamba-blocks` 调整这些容量.

使用 `--case ordinary:1:32768` 选择两个输出长度.
使用 `--case ordinary:1:32768:64` 选择一种配置.
使用 `--output-tokens 64` 进行对 prefill 时间敏感的测量.
使用 `--output-tokens 32` 进行短测试.

命令接受配置子集和其他输出长度, 但拒绝重复配置.
请求失败或产物无效时以错误退出.

设置 `OH_MY_VLLM_EXECUTION_TRACE` 后, worker 写入执行 trace.
Trace 路径必须位于仓库外, 且运行前不能存在.
Benchmark 为服务进程设置此变量.

## 标称资源

使用 [NVIDIA 数据表第 8 页](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf)中的 HGX B200 dense 数值.
吞吐为 HBM 7.7 TB/s, FP8 4.5 PFLOP/s, BF16 2.25 PFLOP/s 和 TF32 1.1 PFLOP/s.
标量吞吐为 FP32 75 TFLOP/s 和 FP64 37 TFLOP/s.
不使用 sparse 数值, GB200 数值或实测校准值替代.

```text
group_seconds = max(
    group_bytes / HBM_rate,
    group_FP8 / FP8_rate + group_BF16 / BF16_rate + group_TF32 / TF32_rate,
    group_FP32 / FP32_rate,
    group_FP64 / FP64_rate,
)
reference_seconds = sum(group_seconds for each serial group)
efficiency = reference_seconds / HTTP_seconds
```

这是保守资源估算, 不是最小物理延迟证明.

## 下界计算

[参考计算](../development/serving/kernels.py)使用版本 `http-reference-v4`.
它使用当前 kernel 边界和精度规则, 不使用实测校准输入.

[Trace 验证器](../development/serving/trace.py)先检查完整 prefill, 连续接受位置, proposal 依赖和输出数量.
唯一 trace identity 必须匹配服务模式.
Prompt 长度必须匹配 HTTP usage.
Benchmark manifest 将客户端序号映射到唯一响应和 trace ID, 模式及输入/输出长度.
缺失 ID, 重复 ID, 未知 ID 或长度不一致都会阻止计费.

参考调度器按客户端序号接纳请求, 最多同时激活 4 个请求.
它先调度 decode, 再将剩余的 32768-token 预算分配给 prefill 请求.
整数余数分配给较早序号.
未完成的 prefill 停在 784-token 边界.
若还能容纳一个对齐 chunk, 未使用空间分配给较早的待处理请求.
最后的 prompt chunk 可以在边界之间结束.

物理到达顺序和调用边界不控制此调度.
MTP verification 消费上一轮 proposal, 并保留记录的接受结果.
Context 计算停在接受终点; 如果随后有新 proposal, 则再计算一行.
调度器不能创建 proposal, 也不能合并同一请求的不同步骤.

计算对每个 kernel 组使用资源时间的最大值.
随后累加串行组的时间.
Prefill 的计算时间与 decode 的访存时间相加.
融合 kernel 构成一个组.
若 consumer 等待前的必要工作可以重叠, 则 PDL producer 和 consumer 使用同一组.

FFN 的 65 至 1024 行使用一个组, 包含 normalization, 融合 upper 计算和 down projection.
每个 consumer 都可以在等待前读取常量权重和 weight scales.
Activation values 和 activation scales 在等待后读取.
其他 prefill 行数保留原有分组.

最多八行的 down16dp projection 在等待后才读取权重.
其计分工作保留为独立组.
层完成边界保持串行.

[分组定义](../development/serving/kernels.py)拆分 FA projection, heads, attention, gate 和 output 工作.
它也拆分 GDN convolution, recurrence 阶段及 projection.
FFN 阶段, MTP input 阶段和词表投影使用独立分组.
必要对象的内存额度归入首次使用它的组.
这些组使用固定的 canonical schedule. Trace launch 数量不改变这些组.

额外 tile, padding, launch, 重复计算或临时复制不能增加计分工作量.

产物包含响应, manifest 及其摘要, 所选 trace 摘要, 资源总量, 分组时间和接受计数.
Manifest 还记录采样默认值, 工具 schema, seed, 请求哈希和工具校验计数.
工具参数必须符合 schema. 因输出上限而停止的最后一次调用可以包含合法的参数前缀.

Manifest 将完整请求计时绑定到产物 identity.
该 identity 绑定源码, 可执行文件, 原生模块, checkpoint, tokenizer, 设备和计费版本.
[产物检查](../development/serving/benchmark.py)拒绝运行期间的变化.
测量产物只描述其记录的源码和计费版本.

## 内存

每个逻辑投影 pass 只计算一次矩阵权重, 由其 batch 共享.
FP8 矩阵每元素使用一字节, 每个 128-by-128 区域使用 FP32 scale.
BF16 矩阵每元素使用两字节.
每个 pass 只计算一次各归一化向量.
中间 hidden state, logits 和 activation buffer 不计内存分数.

FA 对每个请求和 pass 读取一次历史 KV.
新 query KV 可以直接交给 attention, 不再读取 HBM.
接受的 KV 只计一次写入, 使用 4 个 KV head, 宽度 256, K 和 V 均为 BF16.
MTP draft attention 从位置 1 开始. 首个 draft 读取缓存的 context endpoint.
后续 draft 计算新 KV.

当请求起点大于零时, GDN 读取一份 FP32 recurrent state 和一份 BF16 convolution state.
计算该 pass 中每个不同的 784-token checkpoint 和接受终点.
终点上的 checkpoint 只计一次.
Verification 前进接受的 proposal 数加一行.
每个 GDN 层有 48 个 128-by-128 recurrent head, 以及 10240 个各存储 3 个值的 convolution channel.

私有 candidate 写入和后续 state 复制不计分.
此省略不取消 private-state 正确性检查.

## 算术

矩阵乘法按规定 operand 精度计算, 每个 multiply-add 计两次操作.
计算涵盖全部 64 个 target 层和完整词表投影.
全部 verification 行都计入, 包括 proposal 被拒绝的情况.
FA 计算完整因果 attention pair, 每个 pair 和 query head 计三次 FP32 操作.
这些标量操作代表 score scaling, probability summation 和 probability normalization.

每请求 query 数大于 128 时, GDN prefill 使用 64-row chunk, 在 784-token checkpoint 处分割.
Preparation, state carry 和 output 构成三个串行组.
其乘积使用 BF16. 对角求逆和 state update 使用 FP32.
Beta 乘法进入 inverse matrix 后, W/U 三角乘积包含对角线.

短 prefill, decode 和 verification 使用 FP32 recurrence.
融合进 convolution 或 recurrence 的小行 BA projection 对 BF16 输入执行 FP32 算术.
其他 BA projection 使用 BF16 矩阵乘法.
修改这些固定的分派与融合规则时, 必须更新计费版本.

RMS normalization 每元素计四次操作, 每行另计两次标量操作.
Residual addition 每元素计一次操作.
FFN activation 每个 intermediate 元素计两次乘积.
GDN convolution 每 token 和 channel 计七次操作.

FA head normalization 使用宽度 256. Rotary product 使用规定的 64 维.
Attention output gate 每元素计一次乘积.
MTP context, draft 和词表投影遵循记录的因果位置.

## 省略项

采样, 惩罚, 语法处理, 排序, 整数索引, 随机数生成和 MTP acceptance 不计资源分数.
它们的运行时间仍计入 HTTP timer.
估算省略 sigmoid evaluation, transcendental instruction, datatype conversion, quantization 和 scale application 算术.
它还省略上述条目之外的 GDN head normalization, gate, input normalization 以及 prefix/ratio 构造.
省略的操作没有经验吞吐系数.

Trace 没有用于 embedding-region 去重的完整 input token ID.
因此 embedding 内存读取不计分.
Checkpoint hidden storage, MTP endpoint storage, metadata 和返回 token 流量也被省略.
这些保守省略可能降低报告效率.
当实现增加工作量时, 它们不能提高效率.

Vision 和初始化不在这些纯文本 HTTP 计时范围内.
这些计时不能检查它们的正确性.
修改这些假设必须更新计费版本, 并使旧性能证据失效.
