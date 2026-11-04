# 投影运算

[Quantization contract](quantization.zh.md) 规定每个 K32 group 执行一次 raw dot 和一次 scale update.
Grouped FP4 path 保留原始 operator interface, BF16 expert rounding 和完整的 down-projection sum.

## 精确合并 scale

[Native scale update](../../../python/oh_my_vllm/operators/projection/cuda/fp8_k32.cuh#L32)<!-- source: __device__ __forceinline__ float fp8_scaled_add(float raw, uint8_t activation, uint8_t weight, --> 可以合并两个 scale 的指数.
每个有限 E4M3 乘积都是 `2**-18` 的整数倍.
Raw K32 dot 仍是这个值的整数倍, 且绝对值小于 `2**23`.
当 activation scale code 不大于 232 时, 该乘法是精确的, 对 subnormal 结果也成立.
最小 scale 指数是 -127, 因此最小的非零乘积步长是 `2**-145`.
允许的最大指数是 105, 因此结果不会溢出.

如果合并后的指数在 -149 到 127 之间, 对应的 2 的幂可由 FP32 精确表示.
用这个值执行一次 FMA, 得到的结果就与要求的 MUL 和 FMA 相同.
其他 scale 对保留原始操作.
无效 scale code 保留原始 NaN 行为.

## Native reduction layout

[FP16 tile conversion](../../../python/oh_my_vllm/operators/projection/cuda/half_k32.cuh#L19)<!-- source: __device__ __forceinline__ void half_store(uint16_t *destination, int row, int column, --> 保留 CUDA 13.4 FP8 MMA 的 reduction layout.
对于每个 K32 group, 第一条 K16 操作使用位置 `0,1,4,5,...,28,29`.
第二条使用位置 `2,3,6,7,...,30,31`, 并累加到第一条的结果中.
E4M3 和 E2M1 数值均可由 FP16 精确表示.
转换改变 storage 和指令选择, 不改变操作数数值或 K32 reduction order.

## Dense prefill 分块

[Dense prefill 内核](../../../python/oh_my_vllm/operators/projection/cuda/tcgen_k32.cuh#L15)<!-- source: constexpr int tcgen_rows = 128, tcgen_columns = 256, tcgen_slots = 2, tcgen_raw_stages = 3; --> 每个块使用 128 个输入行和 256 个输出列.
它适用于至少 128 行, 输出宽度大于 1024 且可被 256 整除, 输入宽度不超过 6144 的形状.
输入和权重地址必须按 128 字节对齐.
其他形状保留已有内核.

TMA 通过三个共享缓冲区加载原始 FP8 值, 每次加载一个 K128 块.
不同线程分别将数值转换成 FP16, 并为每个 K32 点积发出两次 K16 操作.
两个 TMEM 槽保留独立的原始点积.
输出线程按 K32 顺序对每组执行一次 MUL 和一次 FMA.
调用之间不常驻 FP16 权重副本.

在 Graph 捕获前, 在调用线程和设备上准备各输出类型与缩放布局.
捕获期间可以为新量化存储编码 TMA 描述符.
内核按值接收这些描述符, Graph 所有者保持输入和权重存储有效.

## 共享专家投影

Shared expert 在每个 rank 上保留完整的 FP8 权重和 scale.
没有 hooks 时, 其原始 gate 和 up Linear 模块共享一份量化输入及其 scale.
替换模块, forward 重写, hooks 和编译调用保留原始执行方式.

TP4 和 TEP4 对每个完整的 prefill chunk 只计算一次 shared expert.
Routed expert 保留各自的行分批.
其原生 gate 和 up 投影共享一份 FP8 输入和 UE8M0 K32 缩放因子.
Down 投影单独量化 BF16 激活乘积.
DEP4 仅对 owner 的自有行计算 shared expert.
每个投影保留 raw K32 dot 和有序 FP32 scale 累加.

在路由输出交换后添加 shared 输出.
不要对 shared 输出执行跨 rank reduction.
路由 FP32 和在最终 BF16 转换前接收此 BF16 shared 输出.

## 路由 prefill 波次

`OH_MY_VLLM_DEEPSEEK_MOE_PREFILL_ROWS` 为支持的 Eager target prefill 选择 1024, 4096 或 8192 个 token 行.
四 rank TP4 和 TEP4 默认使用 8192 行.
其它配置使用 1024 行.
较大的投影波次减少未填满的专家 tile 数和设备元数据准备操作数.
将此环境变量设为 1024 可减少临时存储.
Router 调用保留原有的行分组和 source-step 布局.

TP4 的 gate, up 和 SwiGLU 计算使用投影波次.
Product gather, down 投影, 本地求和与输出 gather 的每一步最多处理 1024 个 token 行.
每次 collective 调用使用原有路径的消息维度和先后顺序.

TEP4 的 gate, up, SwiGLU, down 和本地求和使用投影波次.
其路由 FP32 reduction 保留原有的 1024 行切片和消息顺序.
每个专家输出先舍入到 BF16, 然后按专家 ID 顺序进行本地 FP32 求和.

模型持有一个 [operator workspace](../../../python/oh_my_vllm/models/deepseek_v41/moe_prefill.py#L29)<!-- source: class PrefillWorkspace: -->.
其 target 层在调用流上复用这些缓冲区.
首次受支持的 Eager 调用按所选波次和配置的每个 token 路由数预留存储.
缓冲区形状或设备变化会替换缓冲区, `_apply` 会清除缓冲区.

只有具有 CUDA BF16 输入, 原生 FP4 专家, 576 或 2304 的 gate 宽度及 SwiGLU limit 10 的调用会使用此路径.
Decode, Graph, DEP4 和不受支持的专家绑定保留原有路径.

## 本地专家求和

官方 MoE 按专家 ID 递增顺序累加各本地专家的输出.
每个专家输出先舍入到 BF16.
[本地求和](../../../python/oh_my_vllm/models/deepseek_v41/kernel.py#L221)<!-- source: def ordered_expert_sum(values, sorted_routes, output, topk): --> 对每个 token 和路由槽使用一行投影输出.
投影行按专家 ID 排序.
CUDA kernel 对这些行索引排序, 然后按此顺序将其 BF16 值累加到 FP32.
未使用的路由槽不提供数值.

TP4, TEP4 和 DEP4 在输出交换之前使用此求和.
映射对每个 token 最多使用六个 INT32 项.
其存储量与输出宽度无关.
Graph replay 用当前路由重建映射.

## 固定容量投影与求和

`projection.grouped_project_padded(encoded, scales, weight_ptrs, scale_ptrs, counts, output, scratch, max_rows, status)` 使用设备计数确定的前缀.
`counts` 是按专家 ID 排列的 CUDA INT32 `[G]`.
每个计数非负且不超过 `max_rows`; 总和不超过输入容量.
输入使用编码后的 FP8 值和 UE8M0 K32 缩放因子.
调用方持有准备好的 FP4 权重和缩放因子存储, 直到执行完成或 Graph 释放.
仅持有指针数组不能保留这些存储的引用.
有效输出前缀保留严格投影入口的运算顺序, 输出类型为 BF16 或 FP32.
前缀之后的每一行输出都为零, 因此 SwiGLU 可以读取完整缓冲区.
单独的内核清零这些行.
现有工作区查询将容量作为行数, 并要求 `capacity <= G * max_rows`.
没有更小的上界时, 使用 `max_rows=max(1,capacity)`.

`routing.expert_sum_padded(values, sorted_routes, counts, output, route_rows, status)` 使用相同的有效 assignment 前缀.
`values` 包含 BF16 专家输出; `sorted_routes` 包含按专家顺序排列的 INT64 词元及槽位索引.
调用方提供 INT32 `route_rows[token_capacity,topk]`, 每个词元有一到六个槽位.
内核按投影行顺序累加数值, 并写出 FP32 输出.
没有 assignment 的词元行输出为零, 包括未使用的容量尾行.
无效计数, assignment 超容量, 无效路由或重复槽位会设置错误位.
求和不读取有效前缀之后的数值或路由.

两个调用均使用调用方持有的 CUDA INT32 `status[1]`, 不做主机读取.
每次 Execute 开始时清零一次 status.
错误在连续调用和 Graph 重放之间保持.
投影计数无效或已有错误时, 投影输出全零, 不读取输入或权重.
status 非零时, 求和输出全零, 不读取 values.
在缓存提交之前的步边界检查 status.
任何错误均拒绝该步.
零容量调用仍检查 counts.
严格的 `grouped_project` 入口保留计数总和必须相等的要求, 以及无效计数产生 NaN 的行为.
TEP4 Graph 和 DEP4 dispatch 的调用方接入仍由各自工作完成.

## 批量 target 算术

实验性的批量 CUDA target 要求发布配置: 宽度 5120, 四个 HC 副本, FP8 稠密权重和 FP4 路由权重.
Attention 要求 64 个宽度为 512 的 head. Index 评分要求 32 个宽度为 128 的 head.
路由要求 384 个 expert, 每次选择六个 expert.
Attention window 有 128 个槽位, index selection 最多有 512 个槽位.

投影 scale 必须使用 UE8M0 K32 分组. Native sparse attention 必须启用, 且此前没有 status 失败.
不支持的配置设置在绑定 batch 输入和计算模型之前报错.
CPU 参考执行和不使用 batch 的 target 保留现有路径.

TP4 和 TEP4 batch 为每个请求的 index head 权重, index key 和 `wo_a` 保留原来的 BF16 矩阵形状.
Index 评分先完成一个请求的 score 分块, rank 求和和选择, 再开始下一个请求.
每个请求的 query 分块预算为 256 MiB, 与 batch 大小无关.
每次 index score collective 保留单独请求调用的形状和 FP32 求和.
所有 rank 均按请求顺序, 再按 query 分块顺序执行, Graph 捕获期间也相同.
MoE 和 attention 输出 collective 保留现有消息大小.

## 数值检查

[Shape check](../../../development/model/check_quantized_gemm.py) 将 BF16 和 FP32 输出与不可变的 native baseline 比较.
用例覆盖 TP4 和完整 expert 的 shape, 空 expert, 不均匀 count, 以及 1 到 32768 之间选定的 expert row 数.
它与 native baseline 的比较是 arithmetic regression check, 不是独立的 frozen model reference.
[Literal check](../../../tests/test_projection_precision.py) 还检查稀疏 K32 数值, 以及 expert count 改变后的 Graph replay.
将原始记录保存在仓库外.
