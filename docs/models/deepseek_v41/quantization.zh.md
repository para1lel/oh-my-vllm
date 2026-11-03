# 量化契约

Adapter 按发布的 [DeepSeek reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py) 执行 block quantization.
Reference source 的 SHA-256 为 `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
本契约适用于 activation, window KV, compressed KV, index key 和 query.
本契约不改变原始 weight value 或 scale.

## 数学

设 `A` 为一组中最大的 FP32 绝对值.
使用以下 normalization scale:

| Format | Normalization scale |
|---|---|
| FP8, 未取整 | `max(A, 1e-4) * (1 / 448)` |
| FP8, 已取整 | `max(A, 1e-4) * (1 / 448)` 向上取二次幂 |
| FP4, UE8M0 scale | `max(A, 6 * 2**-126) * (1 / 6)` 向上取二次幂 |
| FP4, E4M3 scale | `E4M3(max(A, 6 * 2**-9) / 6)` |

向上取二次幂使用 FP32 的 exponent 和 mantissa bit.
它不使用经过舍入的 `log2` 结果.
FP4 E4M3 scale 必须先舍入, 再归一化数值.
UE8M0 scale storage 向上舍入.
当保存的 scale 使用 UE8M0 时, 未取整的 FP8 normalization 仍使用原始 FP32 scale.

E4M3 和 E2M1 数值使用最近偶数舍入.
将归一化的 FP8 数值限制在 `[-448, 448]`.
将归一化的 FP4 数值限制在 `[-6, 6]`.
E2M1 magnitude code `0` 至 `7` 表示 `0, 0.5, 1, 1.5, 2, 3, 4, 6`.

偶数 channel 放在低 nibble.
负零保留其符号.
全零 FP4 group 保留非零 scale.

In-place 调用先将归一化数值舍入到所选格式, 再反量化.
结果转回输入 dtype.
Encoded 调用返回 E4M3 数值或 packed E2M1 byte, 并在不同的 storage 中返回 scale.
FP4 encoded storage 的宽度是逻辑宽度的一半.

FP4 projection 的 activation 使用 FP8, 而不是 FP4.
Native projection 使用每 32 个 channel 一组的 activation scale.
每个 128-channel activation scale 在四个 native group 中重复.
Reference FP8 projection 沿 row 和 column group 两个维度展开 weight scale.
FP32 scale 输入使用 reference projection path, 而不是 native UE8M0 byte interface.

Dense 和 grouped projection 使用 raw K32 dot product 和 FP32 accumulator.
FP4 weight 在每个 raw dot 前精确转换为 FP8.
FP4 weight scale 应用于每个输出行.
FP8 weight scale 应用于 32 行组.

每个 raw dot 从零开始.
执行 `scaled = mul_rn(raw_dot, activation_scale)`.
执行 `sum = fma_rn(scaled, weight_scale, sum)`.
不要在 dot product 之前将两个 scale 相乘.
仅将完成的总和舍入到所选 output type.

RMS normalization 在将 square 加入 accumulator 之前, 对每个 square 舍入.
使用完整逻辑 row 数和 width 选择 mean reduction layout.
Reduction 完成后, 将 sum 乘以 `FP32(1 / width)`.
不要用 division 替换此 multiplication.

在 FP32 中将每个 input 乘以 inverse RMS.
在 FP32 中将这个 FP32 result 乘以其 weight.
将 result 舍入到 BF16.

[Router](../../../python/oh_my_vllm/models/deepseek_v41/model.py) 在 score calculation 之前, 使用 FP32 input 和 weight 执行 projection.
Expert selection 之前不要将 router logits 舍入到 BF16.
每个 score 使用 `sqrt(x > 20 ? x : log1p(exp(x)))`.

六个 score 的 sum 为 `((s0 + s4) + s2) + ((s1 + s5) + s3)`.
三个 score 的 sum 为 `(s0 + s2) + s1`.
将每个 selected score 除以 `sum + 1e-20`.
将 quotient 乘以 `1.5`.
Corrected score 相等时, native routing 选择最小的 global expert ID.
对于这种 tie, 发布参考的 top-k 可能选择不同 ID.

TP4 对每个路由投影按输出通道分片.
在下投影前收集 BF16 激活乘积.
每个计算进程为其输出通道计算完整的 K32 和.
在 FP32 词元求和前, 将每个完整的专家输出舍入到 BF16.
按进程顺序收集这些 FP32 词元和.
在分组执行和逐个专家循环中保持相同操作.

不要将下投影的 K32 和拆成各进程本地的部分和.
多行收集操作在通道重排期间可能同时持有两个完整的收集结果张量.
临时存储估算必须计入这两个张量.

分组上投影和所有完整的下投影保留 BF16 输出.
分组接口还接受 FP32 输出, 不改变工作区大小.

Shared expert 在每个 rank 上保留完整权重.
在路由输出交换后添加其输出.
不对 shared output 执行跨 rank reduction.
Attention output projection 保留其 FP32 rank reduction 之前的 local BF16 舍入.

Sparse attention 按原顺序处理 selected index, 每个 block 包含 64 个 slot.
QK 和 probability-V product 使用 BF16 operand, 并以 FP32 累加.
Running maximum 从 `-1e30` 开始, 不包含 sink.
Maximum 改变时, 每个 block 对之前的 FP32 numerator 和 denominator 重新缩放.
未舍入的 exponential 计入 denominator.
与 V 相乘之前, 将这些 exponential 舍入到 BF16.

Denominator 先使用 XOR offset `128, 64, 32` 跨 warp 求和.
然后使用 XOR offset `2, 1` 跨四个 lane 求和.
Denominator 更新使用 `fma_rn(previous_sum, scale, block_sum)`.

所有 selected block 完成后, 将 sink exponential 加入 denominator.
将 numerator 除以 denominator.
将 output 舍入到 BF16.
保留 duplicate index 和 masked slot 的原始位置.
Empty row 和 fully masked row 返回零, 即使 finite sink exponential 下溢也如此.

BF16 table 且 query 最多四行时, native attention 并行计算 64-slot partial.
每个 partial 使用按顺序更新的 running maximum 对 probability 舍入.
每个输出通道保留原来的 FP32 numerator 和 denominator 更新顺序.
此路径必须满足冻结 endpoint 的误差限.

DSpark proposal query 共享完整的 draft-index block.
Target attention 保持其因果 index 规则.

在 Sinkhorn mixing 中, 第一次 column normalization 之前, 给 row softmax value 加上 `hc_eps`.
发布模型使用 `hc_eps=1e-6`.
对于四份 copy, row sum 使用 XOR offset `2, 1`.
Column sum 使用 XOR offset `8, 4`.

HC collapse 对每个 FP32 product 先舍入, 再从左向右求和.
HC expansion 先按该顺序对已舍入的 residual product 求和.
在 residual sum 之后, 加上单独舍入的 branch product.
在这些求和中, 对 FP32 multiplication 和 addition 独立舍入.

HC mixing 在四个 FP32 accumulator 中独立累加 squared input.
完整逻辑 row 数决定 reduction layout.
在 FP32 projection 之后应用 inverse RMS.

Native projection 按已发布的 FP32 reduction order 计算.
HC 单 row projection 使用 32 个交错 split dot.
其 caller 拥有的 32 MiB workspace 保存 split partial 和 chunk partial.
多 row step 从实测的已发布 table 选择 split chunk.
chunk 在每个 row 数范围中保持不变.

每个实测 table 属于一个 output shape.
HC table 有 24 个 output, width 为 20480.
Router table 有 384 个 output, width 为 5120.
DSpark router table 有 128 个 output, width 为 5120.
Compressor table 有 512 个 output, width 为 5120.
每个 table 覆盖 2 到 32768 的 row 数.

多 row projection 为每个 row, output 和 chunk 写入一个 partial.
独立的 reduction 按 chunk 顺序累加这些 partial.
未实测的 shape 或 row 数会以错误停止 call.

Router 和 compressor 的单 row projection 使用 32-lane warp reduction.
这个 workspace 大小是数值契约的一部分.

Ratio-1 compressor 使用 BF16 projection.
其多 row call 使用 SM count target 为八的 cuBLAS handle.
这会选择已发布的非 split-K kernel.

一次实测 sweep 覆盖 row 数 2, 4, 8, 16, 64, 256, 1024 和 4096.
Row 数 1024 和 4096 与 CUDA 13.0 image 完全一致.
Row 数 1, 3, 5 到 7, 9 到 15, 17 到 255, 257 到 1023, 1025 到 4095 和 4097 到 32768 未实测.

已知偏差: 部分小 row 数不能复现 CUDA 13.0 的 BF16 结果.
Row 数 2, 4, 8, 16, 64 和 256 分别有 3, 3, 6, 12, 23 和 113 个不同 output.
比较的 output 总数分别为 1024, 2048, 4096, 8192, 32768 和 131072.
每个不同 output 与发布结果最多相差两个 BF16 code unit.

Startup 在 limited handle 上运行固定的 BF16 probe.
不同的 probe pattern 会以错误停止 worker.
Probe 记录已接受的 native library layout.
Probe 固定已接受的偏差 pattern.
环境变为精确时它也会失败.

Probe 只检查第一个 device.
它不覆盖 rows == 1 的 default handle.
Driver 或 library 变化可能选择不同的 kernel.
DSpark markov head 使用 8-lane warp reduction.
Confidence head 使用单 row split-dot order.


## Target row group

Initial target prefill, single-token target call 和 MTP 保持现有的 row layout.
Nonzero multi-token target call 保持 batch-major `[batch, step, ...]` storage.
其 logical row count 为 request batch size.
HC, router 和 compressor FP32 projection 对每个 source step 使用这个 row count.
其 FP32 projection 直接写入 strided physical output storage.

RMS 按 logical row count 选择 mean reduction layout, 并处理所有 physical row.
Engram 使用 token logical row count 的四倍, 执行每份 copy 的 mean 和 dot reduction.
在其最终 residual addition 中, 对 FP32 multiplication 和 addition 独立舍入.
Engram embedding, KV projection, target attention 和 routed expert computation 保持批量执行.
Router projection 不使用 HC workspace.

## 执行

[Model quantizer](../../../python/oh_my_vllm/models/deepseek_v41/quantization.py) 选择 native BF16 path 或 reference computation.
[Native interface](../../../python/oh_my_vllm/operators/transform/api.py) 规定 shape, format, alignment 和 overlap rule.
它使用 caller 的 CUDA stream.
每个 warp 在 in-place write 之前读取一个完整 group.
Native quantization 不分配临时 device storage.

HC preparation 保留 FP32 residual 副本, mean 与 inverse statistics, 以及 operator workspace.
Native HC call 接受最多 32768 行.
Small-row workspace plan 保留 384-row 限制.

HC preparation 是 capture-safe 的.
在 Graph recording 之前准备 HC workspace.
在 Graph release 之前, 保持 captured workspace 存活.
Native manifest 记录 tensor runtime package 的 cuBLAS library image 和 header identity.

Tensor preparation 可以生成一个对齐的 contiguous input copy.
Encoded output 和 scale 使用不同的 storage.

Dense projection 不使用 device scratch storage.

Grouped projection 使用 device row offset, 不上传 host descriptor.
算子会在 device scratch 中压缩非空 group.
grouped launch 覆盖非空 group 和输出 column tile.
每个 block 遍历其 group row, 因此 launch 不使用 host row bound.

每个 device count 必须在零与准备的最大行数之间.
这些 count 的总和必须等于完整输入行数.
无效 device count 生成 NaN output, 且不访问 input 或 weight.

Prepared weight storage 必须保持有效, 直到执行完成或 Graph 释放.
Prepared cache entry 保留原始 weight 和 scale 的 owner, 而不只是其地址.
Source view 的 dtype, shape 或 stride 不同时, 使用不同的 prepared entry.
准备完成后, 不要改变 weight 或 scale.

Host metadata failure 在 launch 之前发生.
无效 device value 或 scale 产生无效 scale code.
Native attention 在接受 output 之前检查这些 code.
Input 必须产生 interface 规定的有限 reconstruction result.

Native attention 将 window KV 和 compressed KV 保存在两张 table 中.
Attention call 不量化这些 BF16 table.
算子也接受 encoded table 及其 scale array.
Reference path 在计算之前连接两张 BF16 table.

Native sparse attention 是 supported input 的默认路径.
设置 `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=0`, 为 short query 选择 reference attention.
除非发生 native status failure, 否则 long query 保持 native selection.
Attention status failure 禁用 native selection.
step boundary 检查一次累积的 status, 并在 worker 提交 cache state 之前报错.

## 验收

[Quantization 检查](../../../tests/test_deepseek_quantization.py) 使用独立的数值和 encoded byte 字面量.
它们检查 scale rule, 最近偶数中点舍入, 负零, exponent boundary 和 projection scale group.
CUDA case 检查 stream order, invalid metadata, empty history 和 temporary storage bound.
这些检查不能替代独立的 full-model endpoint fixture, 或批准的 HTTP 和 AgentX workload.

## Reference 执行

使用隔离的 development environment 执行发布的 TileLang reference.
不要将 TileLang 加入 inference dependency.
对于此 quantizer source, TileLang 0.1.8 可能在所有 input store 完成前复用 input shared storage.
在 reference 副本中, 每个 quantizer 写入 output shared storage 前都要使用 CUDA block barrier.
在 `act_quant_kernel` 和 `fp4_quant_kernel` 中, 紧接 `T.copy(y_local, y_shared)` 之前加入 `T.sync_threads()`.
不要改变 equation, cast 或原始 checkpoint file.

在每个 reference manifest 中记录 original source, patch, modified source, compiler 和 runtime identity.
生成 expected array 前, 使用 Compute Sanitizer Racecheck 检查 packed 和 in-place format.
除非重复执行的 target 和 DSpark endpoint 逐位一致, 否则不要冻结 array.
将此 backend 标识为带 synchronization patch 的 reference, 而不是未修改的发布版本.
不要导入 candidate module 来生成 expected array.
